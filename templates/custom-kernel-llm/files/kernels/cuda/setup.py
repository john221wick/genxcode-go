from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_norm_kernel",
    ext_modules=[
        CUDAExtension(
            name="fused_norm_kernel",
            sources=[
                "fused_norm_kernel.cpp",
                "fused_norm_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
