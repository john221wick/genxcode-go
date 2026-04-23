import torch
import torch.nn as nn
from .registry import register_norm


@register_norm("torch")
class TorchNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.norm(x)


@register_norm("triton")
class TritonNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        from kernels.triton.layernorm import triton_layernorm

        return triton_layernorm(x, self.weight, self.bias, self.eps)


@register_norm("cuda")
class CUDAFusedNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        try:
            from kernels.cuda.fused_norm_kernel import fused_layernorm

            return fused_layernorm(x, self.weight, self.bias, self.eps)
        except ImportError:
            import warnings

            warnings.warn(
                "CUDA fused norm not available, falling back to PyTorch LayerNorm. "
                "Run 'make build-cuda' to compile the CUDA kernel."
            )
            return torch.nn.functional.layer_norm(
                x, self.weight.shape, self.weight, self.bias, self.eps
            )
