import torch
import torch.nn as nn
from .registry import register_attn


@register_attn("torch")
class TorchAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim**0.5
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.drop(weights)
        out = weights @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


@register_attn("flash")
class FlashAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        from flash_attn import flash_attn_func

        B, T, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = flash_attn_func(
            q, k, v, causal=True, dropout_p=self.drop.p if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)
