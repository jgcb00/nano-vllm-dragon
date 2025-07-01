import math
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import Srelu
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm, HeadWiseRMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
        swa: bool = False,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        if swa :
            self.attn = SwaAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.num_kv_heads,
            )
        else:
            self.register_buffer(
                "lambda_init", torch.tensor(0.8 - 0.6 * math.exp(-0.3 * layer_depth))
            )
            self.lambda_q1 = torch.nn.Parameter(
                torch.zeros((head_dim,), dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k1 = torch.nn.Parameter(
                torch.zeros((head_dim,), dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_q2 = torch.nn.Parameter(
                torch.zeros((head_dim,), dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.lambda_k2 = torch.nn.Parameter(
                torch.zeros((head_dim,), dtype=torch.float32).normal_(mean=0, std=0.1)
            )
            self.softmax_scaler = nn.Parameter(
                torch.ones(self.n_heads, dtype=torch.float32)
            )
            self.attn = DiffAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.num_kv_heads,
            )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class DragonGatedDeltaNet(nn.Module):
    pass

class DragonMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert (
            hidden_act == "srelu"
        ), "Dragon models were trained with squared ReLU (SReLU) activation."
        self.act_fn = Srelu()

    def forward(self, x):
        x = self.p_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class DragonDecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        swa: bool = False,
        layer_depth: int = 0,
        kv_source=None,
    ) -> None:
        """
        swa: whether to use local attention/SWA for this block, or global
        kv_source: layer to get KV from, if any
        """
        super().__init__()

        self.premixer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        if swa:
            self.attn_group_norm = HeadWiseRMSNorm(
                n_heads=config.num_attention_heads,
                d_head=2 * config.hidden_size // config.num_attention_heads,
                eps=config.rms_norm_eps,
            )
        else :
            self.attn_group_norm = HeadWiseRMSNorm(
                n_heads=config.num_attention_heads // 2,
                d_head=4 * config.hidden_size // config.num_attention_heads,
                eps=config.rms_norm_eps,
            )

        

        self.lin_attn = DragonGatedDeltaNet

        self.lin_attn_group_norm = HeadWiseRMSNorm(
            n_heads=config.n_heads, 
            d_head=2 * config.hidden_size // config.num_attention_heads,
            eps=config.eps_rmsnorm
        )

        self.out_proj = RowParallelLinear()

        self.premlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.mlp = DragonMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.register_buffer(
            "layer_norm_scaling", torch.tensor(1 / math.sqrt(layer_depth + 1))
        )

        # Caution, this can cause differences in the output because of different rounding errors
        self.premlp_norm.weight.data.mul_(self.layer_norm_scaling)
        self.premixer_norm.weight.data.mul_(self.layer_norm_scaling)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else:
        #     hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.premixer_norm(hidden_states)

        y_attn = self.attn(positions, hidden_states)
        y_lin_attn = self.lin_attn(positions, hidden_states)

        y_attn = self.attn_group_norm(y_attn).view(y_attn.size(0), y_attn.size(1), -1)
        y_lin_attn = self.lin_attn_group_norm(y_lin_attn).view(
            y_lin_attn.size(0), y_lin_attn.size(1), -1
        )

        x = x + self.out_proj((y_attn + y_lin_attn) / 2)

        hidden_states = self.mlp(self.premlp_norm(hidden_states))
        return hidden_states


class DragonModel(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [DragonDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DragonForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.model = DragonModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
