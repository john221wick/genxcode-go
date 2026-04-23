import torch.nn as nn
from kernels import get_norm_layer, get_feedforward, get_attention, KernelBackend
from kernels.registry import ATTN_REGISTRY


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        max_len,
        dropout=0.1,
        kernel_backend: KernelBackend | str = "torch",
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.kernel_backend = kernel_backend

        self.ln1 = get_norm_layer(d_model, kernel_backend)
        self.ln2 = get_norm_layer(d_model, kernel_backend)

        attn_backend = "flash" if use_flash_attn else str(kernel_backend)
        if use_flash_attn and "flash" not in ATTN_REGISTRY:
            import warnings

            warnings.warn(
                "FlashAttention not available. Install with: pip install flash-attn. "
                "Falling back to standard attention."
            )
            attn_backend = str(kernel_backend)
        self.attn = get_attention(d_model, n_heads, max_len, dropout, attn_backend)

        self.ff = get_feedforward(d_model, dropout, kernel_backend)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
