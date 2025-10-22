"""
Setup script for roiaware_pool3d CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="roiaware_pool3d_cuda",
    ext_modules=[
        CUDAExtension(
            name="roiaware_pool3d_cuda",
            sources=[
                "src/roiaware_pool3d.cpp",
                "src/roiaware_pool3d_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
