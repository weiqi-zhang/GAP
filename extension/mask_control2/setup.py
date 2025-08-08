import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='depthid_render_mask_control_v2',
    version='2.0',
    author='mts',
    author_email='',
    description='test',
    ext_modules=[
        CUDAExtension(
            name='depthid_render_mask_control_v2',
            sources=['wake_up.cpp','render2.cu'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--extended-lambda']
            }
        )
    ],
    cmdclass={
        'build_ext':BuildExtension
    }
)