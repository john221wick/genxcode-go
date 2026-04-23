import torch
import torch.nn as nn
from src.layers import TransformerBlock
from kernels import get_norm_layer


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_len = cfg.max_len
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.max_len,
                    cfg.dropout,
                    kernel_backend=cfg.kernel_backend,
                    use_flash_attn=cfg.use_flash_attn,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = get_norm_layer(cfg.d_model, cfg.kernel_backend)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.max_len, f"Seq len {T} > max_len {self.max_len}"
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss
