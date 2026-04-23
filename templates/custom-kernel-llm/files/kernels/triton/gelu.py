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
    def _gelu_kernel(
        INPUT,
        OUTPUT,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(INPUT + offsets, mask=mask, other=0.0)

        const = 0.7978845608028654
        beta = 0.044715
        inner = const * (x + beta * x * x * x)
        tanh_inner = tl.math.tanh(inner)
        result = 0.5 * x * (1.0 + tanh_inner)

        tl.store(OUTPUT + offsets, result, mask=mask)

    class _TritonGELUFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            _check_triton()
            output = torch.empty_like(x)
            n_elements = x.numel()
            BLOCK_SIZE = 1024
            n_programs = triton.cdiv(n_elements, BLOCK_SIZE)

            _gelu_kernel[(n_programs,)](
                x,
                output,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )

            ctx.save_for_backward(x)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
            tanh_inner = torch.tanh(inner)
            sech_squared = 1.0 - tanh_inner * tanh_inner
            grad_inner = 0.7978845608028654 * (1.0 + 0.134145 * x * x)
            grad_x = grad_output * (
                0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_squared * grad_inner
            )
            return grad_x


def triton_gelu(x):
    if not HAS_TRITON:
        raise ImportError(
            "Triton is not available. Install with: pip install triton\n"
            "Triton requires a CUDA GPU."
        )
    return _TritonGELUFn.apply(x)
