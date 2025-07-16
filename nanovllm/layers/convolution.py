import torch
from torch import nn
from fla.modules import ShortConvolution
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

class ColumnParallelConvolution(nn.Module):

    def __init__(
        self,
        input_size: int,
        conv_size: int,
        activation="silu",
        bias: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.input_size_per_partition = input_size
        self.conv_size = conv_size
        self.tp_dim = divide(input_size, self.tp_size)

        self.conv = ShortConvolution(
            hidden_size=divide(input_size, self.tp_size),
            kernel_size=conv_size,
            activation=activation,
        )  
        self.conv.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor, cache : torch.Tensor) -> torch.Tensor:
        return self.conv(
            x,
            mask=None,
            cache=cache,
            output_final_state=True,
            seq_idx=None
        )
    

class QKVParallelConvolution(ColumnParallelConvolution):
    
    def __init__(
        self,
        conv_size: int,
        head_size: int,
        num_heads:int,
        activation="silu",
        bias: bool = False,
    ):
        self.head_size = head_size
        self.num_heads = num_heads
        input_size = 4 * num_heads * head_size
        super().__init__(
            input_size=input_size,
            conv_size=conv_size,
            head_size=head_size,
            num_heads=num_heads,
            activation=activation,
            bias=bias,
        )
        self.tp_num_heads = divide(num_heads, self.tp_size)
        self.qk_size = self.tp_num_heads * self.head_size
        self.v_size = 2 * self.tp_num_heads * self.head_size
    
    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str
    ):
        param_data = param.data
        assert loaded_shard_id in ["q_conv1d", "k_conv1d", "v_conv1d"]
        if loaded_shard_id == "q_conv1d":
            shard_size = self.qk_size
            shard_offset = 0
        elif loaded_shard_id == "k_conv1d":
            shard_size = self.qk_size
            shard_offset = self.qk_size
        else:
            # v_conv1d
            shard_size = self.v_size
            shard_offset = 2 * self.qk_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)
        
    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (b, l, D)
        Returns: same shape as x
        """
        y, cache = super().forward(
            x,
            cache=cache,
        )
        q, k, v = y.split(
            [self.qk_size, self.qk_size, self.v_size],
            dim=-1
        )
        return q, k, v, cache