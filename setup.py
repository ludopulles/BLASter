from pathlib import Path
from sys import argv, platform
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


# Make sure OpenMP is used in Cython and Eigen.
openmp_arg = '/openmp' if platform.startswith("win") else '-fopenmp'
include_dirs = [np.get_include()]

# Compile with extra arguments
compile_args = [
    '--std=c++17',
    '-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION',
    openmp_arg,
]

# Link with extra arguments
link_args = [openmp_arg]

if '--cython-gdb' in argv:
    # Debug arguments
    debug_args = [
        '-O1',
        '-fsanitize=address,undefined',
        '-g',
        '-fno-omit-frame-pointer',
    ]
    compile_args += debug_args
    link_args += debug_args
else:
    # "Release" arguments
    compile_args += [
        '-O3',
        '-march=native',
    ]

extensions = [Extension(
    name="blaster_core",
    sources=["core/blaster.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args
)]

setup(
    ext_modules=cythonize(extensions, language_level="3", build_dir='build/cpp'),
    options={'build': {'build_lib': 'src/'}},
)
