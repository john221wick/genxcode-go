"""TP+SP transformer.

Each transformer block:
  1. Input arrives in SP form (seq sharded across ranks)
  2. All-gather to full sequence for TP attention/FFN
  3. Reduce-scatter back to SP after TP region
  4. LayerNorm/dropout run on SP shard (memory savings!)
"""

import torch
import torch.nn as nn
import math

from .config import MODEL_CONFIG
from .parallel_layers import ColumnParallelLinear, RowParallelLinear
from .sp_ops import gather_from_sp, reduce_scatter_to_sp, scatter_to_sp


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SPAttention(nn.Module):
    """TP attention that accepts SP input."""

    def __init__(self, hidden_dim, num_heads, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = torch.distributed.get_world_size(tp_group)
        self.num_heads = num_heads
        self.heads_per_rank = num_heads // self.tp_size
        self.head_dim = hidden_dim // num_heads

        self.qkv = ColumnParallelLinear(hidden_dim, 3 * hidden_dim, tp_group)
        self.out_proj = RowParallelLinear(hidden_dim, hidden_dim, tp_group)

    def forward(self, x):
        # x is SP-sharded: (B, T/tp_size, C)
        # Gather to full seq for attention
        x_full = gather_from_sp(x, self.tp_group)  # (B, T, C)

        B, T, C = x_full.shape
        qkv = self.qkv(x_full)
        qkv = qkv.reshape(B, T, 3, self.heads_per_rank, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, self.heads_per_rank * self.head_dim)
        out = self.out_proj(out)  # all-reduce happens inside

        # Scatter back to SP
        return scatter_to_sp(out, self.tp_group)  # (B, T/tp_size, C)


class SPFeedForward(nn.Module):
    """TP FFN that accepts SP input."""

    def __init__(self, hidden_dim, tp_group):
        super().__init__()
        self.tp_group = tp_group
        intermediate = 4 * hidden_dim
        self.gate_proj = ColumnParallelLinear(hidden_dim, intermediate, tp_group)
        self.up_proj = ColumnParallelLinear(hidden_dim, intermediate, tp_group)
        self.down_proj = RowParallelLinear(intermediate, hidden_dim, tp_group)

    def forward(self, x):
        # x is SP-sharded
        x_full = gather_from_sp(x, self.tp_group)
        out = self.down_proj(nn.functional.silu(self.gate_proj(x_full)) * self.up_proj(x_full))
        return scatter_to_sp(out, self.tp_group)


class SPTransformerBlock(nn.Module):
    """Block where LayerNorm runs on SP shard, TP runs on full seq."""

    def __init__(self, hidden_dim, num_heads, tp_group):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)  # runs on SP shard
        self.attn = SPAttention(hidden_dim, num_heads, tp_group)
        self.ff_norm = RMSNorm(hidden_dim)  # runs on SP shard
        self.ff = SPFeedForward(hidden_dim, tp_group)

    def forward(self, x):
        # x is SP-sharded throughout
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class SPTransformer(nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.tp_group = tp_group
        hidden_dim = MODEL_CONFIG["hidden_dim"]
        num_layers = MODEL_CONFIG["num_layers"]
        num_heads = MODEL_CONFIG["num_heads"]
        vocab_size = MODEL_CONFIG["vocab_size"]
        seq_len = MODEL_CONFIG["seq_len"]

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.layers = nn.ModuleList(
            [SPTransformerBlock(hidden_dim, num_heads, tp_group) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok + pos

        # Scatter to SP before entering layers
        x = scatter_to_sp(x, self.tp_group)

        for layer in self.layers:
            x = layer(x)

        # Gather back for final norm + head
        x = gather_from_sp(x, self.tp_group)
        x = self.norm(x)
        return self.head(x)
