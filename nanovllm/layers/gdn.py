# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Tri Dao, Albert Gu.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from https://github.com/state-spaces/mamba/
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Union
from einops import rearrange
import triton
import triton.language as tl
from nanovllm.layers.convolution import QKVParallelConvolution
from nanovllm.layers.linear import QKVDAGParallelLinear
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from config import NanoConfig
from nanovllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    conv: torch.Tensor,
    hidden_state: torch.Tensor,
    linear_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = hidden_state.shape
    D = num_heads * head_dim
    assert conv.stride(-1) == 1 and hidden_state.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )



class MixerGatedDeltaNet(nn.Module):
    def __init__(
        self, config: NanoConfig,
    ):
        super().__init__()
        self.config = config

        self.d_model = config.d_model
        self.expand_v = 2

        self.n_heads = config.num_heads
        self.d_head = int(self.d_model ) // self.n_heads

        self.key_dim = self.n_heads * self.d_head
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = self.d_head
        self.head_v_dim = self.d_head * self.expand_v
        self.silu = nn.SiLU()

        self.qkvgba_proj = QKVDAGParallelLinear(
            hidden_size=self.d_model,
            head_size=self.d_head,
            total_num_heads=self.n_heads,
            bias=False,
        )
        self.qkv_conv = QKVParallelConvolution(
            conv_size=config.d_conv,
            head_size=self.d_head,
            num_heads=self.n_heads,
        )

        self.A_log = ColumnParallelLinear(
            hidden_size=self.n_heads,
            output_size=1,
            dtype=torch.float32,
            bias=False,
            )
        self.dt_bias = nn.Parameter(
            torch.zeros(self.n_heads, dtype=torch.float32),
        )
        self.apply(self._initialize_weights)
        
        self.linear_cache = torch.tensor([[]])

    def forward(self, hidden_states, cache=None):
        """
        hidden_states: (b, l, d)
        Returns: same shape as hidden_states
        """
        context = get_context()

        qkv, gate, beta, A = self.qkvgba_proj(hidden_states)  # (b, l, D)

        # split proj into q, k, v, b, a
        q, k, v, conv_cache = self.qkv_conv(
            qkv,
            cache=conv_cache,
        )
        h_cache, q_conv_cache, k_conv_cache, v_conv_cache = None, None, None, None
        if cache is not None:
            h_cache, q_conv_cache, k_conv_cache, v_conv_cache = cache

        q, k = map(
            lambda x: rearrange(x, "b t (h d) -> b t h d", d=self.head_k_dim), (q, k)
        )
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)
        beta = b_proj.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(A.float() + self.dt_bias)

        if context.is_prefill:
            o, h_cache = chunk_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                initial_state=h_cache,
                output_final_state=(cache is not None),
                cu_seqlens=None,  # for varlen training
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )  # (b t h d) where d is head_v_dim
        else :
            o, h_cache = fused_recurrent_gated_delta_rule(
                q=q.bfloat16(),
                k=k.bfloat16(),
                v=v.bfloat16(),
                g=g,
                beta=beta,
                initial_state=h_cache,
                output_final_state=(cache is not None),
                cu_seqlens=None,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )  # (b t h d) where d is head_v_dim

        # gate
        g = self.g_proj(hidden_states).view(
            o.size(0), o.size(1), o.size(2), o.size(3)
        )  # (B, L, H, D)
        o = o * F.silu(g)

        return o

    def get_empty_cache(self):
        return (
            None,
            None,
            None,
            None,
        )  # (h_cache, q_conv_cache, k_conv_cache, v_conv_cache)