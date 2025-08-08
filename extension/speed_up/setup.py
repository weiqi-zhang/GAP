import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='depthid_render',
    version='1.0',
    author='mts',
    author_email='',
    description='',
    ext_modules=[
        CUDAExtension(
            name='depthid_render',
            sources=['wake_up.cpp','render.cu'])
    ],
    cmdclass={
        'build_ext':BuildExtension
    }
)