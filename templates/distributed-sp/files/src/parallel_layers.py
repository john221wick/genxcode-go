"""Column/Row parallel linears (same as TP template)."""

import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group, bias=False):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        assert out_features % self.tp_size == 0
        self.out_per_rank = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.kaiming_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(self.out_per_rank)) if bias else None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group, bias=False):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        assert in_features % self.tp_size == 0
        self.in_per_rank = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.kaiming_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
        return out
