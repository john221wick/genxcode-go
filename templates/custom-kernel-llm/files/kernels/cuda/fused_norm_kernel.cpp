#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA kernels
std::vector<at::Tensor> fused_layernorm_cuda(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

at::Tensor fused_gelu_cuda(const at::Tensor& input);

// C++ interface functions
std::vector<at::Tensor> fused_layernorm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    return fused_layernorm_cuda(input, weight, bias, eps);
}

at::Tensor fused_gelu(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    return fused_gelu_cuda(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_layernorm", &fused_layernorm, "Fused LayerNorm (CUDA)");
    m.def("fused_gelu", &fused_gelu, "Fused GELU (CUDA)");
}
