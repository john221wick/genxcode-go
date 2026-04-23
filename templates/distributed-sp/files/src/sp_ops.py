"""Sequence parallelism primitives.

In TP, the attention and FFN are split across ranks, but LayerNorm
and dropout operate on the full hidden dim — so they're replicated.
SP fixes this by sharding the sequence dimension for these ops.

At TP boundaries:
  - Before TP region: all-gather along sequence dim (each rank gets full seq)
  - After TP region: reduce-scatter along sequence dim (back to sharded)

This means each rank only holds seq_len/tp_size tokens for the
non-TP operations, saving activation memory.
"""

import torch
import torch.distributed as dist


class _ScatterToSequenceParallel(torch.autograd.Function):
    """Scatter full sequence to SP shards (forward), all-gather (backward)."""

    @staticmethod
    def forward(ctx, input, tp_group):
        ctx.tp_group = tp_group
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        B, T, C = input.shape
        assert T % world_size == 0
        chunk_size = T // world_size
        return input[:, rank * chunk_size : (rank + 1) * chunk_size, :].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        tp_group = ctx.tp_group
        world_size = dist.get_world_size(tp_group)
        gathered = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=tp_group)
        return torch.cat(gathered, dim=1), None


class _GatherFromSequenceParallel(torch.autograd.Function):
    """All-gather SP shards to full sequence (forward), scatter (backward)."""

    @staticmethod
    def forward(ctx, input, tp_group):
        ctx.tp_group = tp_group
        world_size = dist.get_world_size(tp_group)
        gathered = [torch.empty_like(input) for _ in range(world_size)]
        dist.all_gather(gathered, input.contiguous(), group=tp_group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        tp_group = ctx.tp_group
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)

        B, T, C = grad_output.shape
        chunk_size = T // world_size
        return grad_output[:, rank * chunk_size : (rank + 1) * chunk_size, :].contiguous(), None


def scatter_to_sp(input, tp_group):
    """Full sequence -> SP shard (keep only local chunk along seq dim)."""
    return _ScatterToSequenceParallel.apply(input, tp_group)


def gather_from_sp(input, tp_group):
    """SP shard -> full sequence (all-gather along seq dim)."""
    return _GatherFromSequenceParallel.apply(input, tp_group)


class _ReduceScatterToSequenceParallel(torch.autograd.Function):
    """Reduce-scatter: sum partial results and scatter along sequence dim."""

    @staticmethod
    def forward(ctx, input, tp_group):
        ctx.tp_group = tp_group
        world_size = dist.get_world_size(tp_group)

        B, T, C = input.shape
        assert T % world_size == 0
        chunk_size = T // world_size

        output = torch.empty(B, chunk_size, C, device=input.device, dtype=input.dtype)
        input_list = list(input.chunk(world_size, dim=1))
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tp_group = ctx.tp_group
        world_size = dist.get_world_size(tp_group)
        gathered = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=tp_group)
        return torch.cat(gathered, dim=1), None


def reduce_scatter_to_sp(input, tp_group):
    """Reduce partial TP outputs and scatter along sequence dim."""
    return _ReduceScatterToSequenceParallel.apply(input, tp_group)
