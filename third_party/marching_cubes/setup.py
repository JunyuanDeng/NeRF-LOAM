from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_sources = glob.glob("src/*.cpp") + glob.glob("src/*.cu")

setup(
    name='marching_cubes',
    ext_modules=[
        CUDAExtension(
            name='marching_cubes',
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I./include"],
                "nvcc": ["-I./include"]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)