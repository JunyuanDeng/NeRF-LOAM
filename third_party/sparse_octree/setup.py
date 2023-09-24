# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import glob

_ext_sources = glob.glob("src/*.cpp")

setup(
    name='svo',
    ext_modules=[
        CppExtension(
            name='svo',
            sources=_ext_sources,
            include_dirs=["./include"],
            extra_compile_args={
                "cxx": ["-O2", "-I./include"]
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
