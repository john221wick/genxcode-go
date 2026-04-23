from enum import StrEnum

class KernelBackend(StrEnum):
    TORCH = "torch"
    TRITON = "triton"
    CUDA = "cuda"
