import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def _check_triton():
    if not HAS_TRITON:
        raise ImportError(
            "Triton is not installed. Install with: pip install triton\n"
            "Note: Triton requires a CUDA GPU and will not work on CPU/MPS."
        )


if HAS_TRITON:

    @triton.jit
    def _layernorm_kernel(
        X,
        Y,
        W,
        B,
        stride,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(X + row * stride + cols, mask=mask, other=0.0)

        mean = tl.sum(x, axis=0) / N
        diff = x - mean
        var = tl.sum(diff * diff, axis=0) / N
        x_norm = diff / tl.sqrt(var + eps)

        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = x_norm * w + b

        tl.store(Y + row * stride + cols, y, mask=mask)

    class _TritonLayerNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            _check_triton()
            M, N = x.shape
            y = torch.empty_like(x)
            BLOCK_SIZE = triton.next_power_of_2(N)

            _layernorm_kernel[(M,)](
                x,
                y,
                weight,
                bias,
                x.stride(0),
                N,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(x, weight, bias)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, bias = ctx.saved_tensors
            eps = ctx.eps
            M, N = x.shape

            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / torch.sqrt(var + eps)

            grad_weight = (grad_output * x_norm).sum(dim=0)
            grad_bias = grad_output.sum(dim=0)

            grad_x_norm = grad_output * weight
            grad_var = (grad_x_norm * (x - mean) * -0.5 * (var + eps) ** (-1.5)).sum(
                dim=-1, keepdim=True
            )
            grad_mean = (grad_x_norm * -1.0 / torch.sqrt(var + eps)).sum(
                dim=-1, keepdim=True
            ) + grad_var * (-2.0 * (x - mean).mean(dim=-1, keepdim=True))
            grad_x = (
                grad_x_norm / torch.sqrt(var + eps)
                + grad_var * 2.0 * (x - mean) / N
                + grad_mean / N
            )

            return grad_x, grad_weight, grad_bias, None


def triton_layernorm(x, weight, bias, eps=1e-5):
    if not HAS_TRITON:
        raise ImportError(
            "Triton is not available. Install with: pip install triton\n"
            "Triton requires a CUDA GPU."
        )
    return _TritonLayerNormFn.apply(x, weight, bias, eps)
