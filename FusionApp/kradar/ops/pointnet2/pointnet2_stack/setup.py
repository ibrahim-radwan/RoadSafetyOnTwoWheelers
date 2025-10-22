"""
Setup script for pointnet2_stack CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os

_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="pointnet2_stack_cuda",
    ext_modules=[
        CUDAExtension(
            name="pointnet2_stack_cuda",
            sources=sorted(glob.glob("src/*.cpp") + glob.glob("src/*.cu")),
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
