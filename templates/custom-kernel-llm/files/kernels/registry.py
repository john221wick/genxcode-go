NORM_REGISTRY: dict[str, type] = {}
FFN_REGISTRY: dict[str, type] = {}
ATTN_REGISTRY: dict[str, type] = {}

def register_norm(backend: str):
    def decorator(cls):
        NORM_REGISTRY[backend] = cls
        return cls
    return decorator

def register_ffn(backend: str):
    def decorator(cls):
        FFN_REGISTRY[backend] = cls
        return cls
    return decorator

def register_attn(backend: str):
    def decorator(cls):
        ATTN_REGISTRY[backend] = cls
        return cls
    return decorator
