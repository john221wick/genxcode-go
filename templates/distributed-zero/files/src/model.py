import torch
import torch.nn as nn
import math

from .config import MODEL_CONFIG


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        intermediate = 4 * hidden_dim
        self.gate_proj = nn.Linear(hidden_dim, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim, num_heads)
        self.ff_norm = RMSNorm(hidden_dim)
        self.ff = FeedForward(hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = MODEL_CONFIG["hidden_dim"]
        num_layers = MODEL_CONFIG["num_layers"]
        num_heads = MODEL_CONFIG["num_heads"]
        vocab_size = MODEL_CONFIG["vocab_size"]
        seq_len = MODEL_CONFIG["seq_len"]

        self.tok_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok = self.tok_emb(input_ids)
        pos = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok + pos
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)
