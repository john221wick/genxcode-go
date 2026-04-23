import torch
import torch.nn as nn
from .registry import register_ffn


@register_ffn("torch")
class TorchFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


@register_ffn("triton")
class TritonFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        from kernels.triton.gelu import triton_gelu

        h = self.w1(x)
        h = triton_gelu(h)
        out = self.w2(h)
        return self.drop(out)


@register_ffn("cuda")
class CUDAFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, 4 * d_model)
        self.w2 = nn.Linear(4 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.w1(x)
        try:
            from kernels.cuda.fused_norm_kernel import fused_gelu

            h = fused_gelu(h)
        except ImportError:
            import warnings

            warnings.warn(
                "CUDA fused GELU not available, falling back to torch.nn.GELU. "
                "Run 'make build-cuda' to compile the CUDA kernel."
            )
            h = torch.nn.functional.gelu(h)
        out = self.w2(h)
        return self.drop(out)
