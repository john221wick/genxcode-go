#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void layernorm_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    const int N,
    const double eps,
    const int stride) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= stride) return;

    const scalar_t* row_input = input + row * N;
    scalar_t* row_output = output + row * N;

    // Compute mean
    scalar_t mean = 0.0;
    for (int i = 0; i < N; i++) {
        mean += row_input[i];
    }
    mean /= N;

    // Compute variance
    scalar_t var = 0.0;
    for (int i = 0; i < N; i++) {
        scalar_t diff = row_input[i] - mean;
        var += diff * diff;
    }
    var /= N;

    // Normalize + scale + shift
    scalar_t inv_std = 1.0 / sqrt(var + eps);
    for (int i = 0; i < N; i++) {
        row_output[i] = weight[i] * (row_input[i] - mean) * inv_std + bias[i];
    }
}

template <typename scalar_t>
__global__ void gelu_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int n_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const scalar_t x = input[idx];
    const scalar_t const_val = 0.7978845608028654; // sqrt(2/pi)
    const scalar_t beta = 0.044715;
    const scalar_t inner = const_val * (x + beta * x * x * x);
    const scalar_t tanh_inner = tanh(inner);

    output[idx] = 0.5 * x * (1.0 + tanh_inner);
}

std::vector<at::Tensor> fused_layernorm_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {

    auto output = at::empty_like(input);
    const int rows = input.size(0);
    const int N = input.size(1);

    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_layernorm_cuda", ([&] {
        layernorm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, eps, rows);
    }));

    return {output};
}

at::Tensor fused_gelu_cuda(const at::Tensor& input) {
    auto output = at::empty_like(input);
    const int n_elements = input.numel();

    const int threads = 1024;
    const int blocks = (n_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_gelu_cuda", ([&] {
        gelu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            n_elements);
    }));

    return output;
}
