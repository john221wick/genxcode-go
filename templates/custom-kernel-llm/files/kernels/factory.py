from .enum import KernelBackend
from .registry import NORM_REGISTRY, FFN_REGISTRY, ATTN_REGISTRY


def get_norm_layer(d_model: int, backend: KernelBackend | str, eps: float = 1e-5):
    backend = KernelBackend(backend) if isinstance(backend, str) else backend
    if backend not in NORM_REGISTRY:
        raise ValueError(
            f"No norm backend '{backend}' registered. "
            f"Available: {list(NORM_REGISTRY.keys())}"
        )
    return NORM_REGISTRY[backend](d_model, eps=eps)


def get_feedforward(d_model: int, dropout: float, backend: KernelBackend | str):
    backend = KernelBackend(backend) if isinstance(backend, str) else backend
    if backend not in FFN_REGISTRY:
        raise ValueError(
            f"No FFN backend '{backend}' registered. "
            f"Available: {list(FFN_REGISTRY.keys())}"
        )
    return FFN_REGISTRY[backend](d_model, dropout=dropout)


def get_attention(d_model, n_heads, max_len, dropout, backend: KernelBackend | str):
    backend = KernelBackend(backend) if isinstance(backend, str) else backend
    if backend not in ATTN_REGISTRY:
        raise ValueError(
            f"No attention backend '{backend}' registered. "
            f"Available: {list(ATTN_REGISTRY.keys())}"
        )
    return ATTN_REGISTRY[backend](d_model, n_heads, max_len, dropout=dropout)
