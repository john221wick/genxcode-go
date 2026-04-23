try:
    from fused_norm_kernel import fused_layernorm as _fused_layernorm_cuda
    from fused_norm_kernel import fused_gelu as _fused_gelu_cuda

    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False


import torch


def fused_layernorm(input, weight, bias, eps=1e-5):
    if not HAS_CUDA_KERNEL:
        raise ImportError(
            "CUDA fused norm kernel not compiled. "
            "Run: cd kernels/cuda && python setup.py install"
        )
    results = _fused_layernorm_cuda(input.cuda(), weight.cuda(), bias.cuda(), eps)
    return results[0]


def fused_gelu(input):
    if not HAS_CUDA_KERNEL:
        raise ImportError(
            "CUDA fused GELU kernel not compiled. "
            "Run: cd kernels/cuda && python setup.py install"
        )
    return _fused_gelu_cuda(input.cuda())


__all__ = ["fused_layernorm", "fused_gelu", "HAS_CUDA_KERNEL"]
