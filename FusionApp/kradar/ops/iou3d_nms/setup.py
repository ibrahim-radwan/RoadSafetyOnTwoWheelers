"""
Setup script for iou3d_nms CUDA extension
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this setup.py
_src_path = os.path.dirname(os.path.abspath(__file__))

setup(
    name="iou3d_nms_cuda",
    ext_modules=[
        CUDAExtension(
            name="iou3d_nms_cuda",
            sources=[
                "src/iou3d_cpu.cpp",
                "src/iou3d_nms_api.cpp",
                "src/iou3d_nms.cpp",
                "src/iou3d_nms_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
