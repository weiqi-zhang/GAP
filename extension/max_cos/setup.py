from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='find_max_in_circles',
    ext_modules=[
        CUDAExtension('find_max_in_circles', [
            'find_max_in_circles.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })