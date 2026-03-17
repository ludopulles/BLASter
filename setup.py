from pathlib import Path
from sys import argv, platform
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy as np


# Make sure OpenMP is used in Cython and Eigen.
openmp_arg = '/openmp' if platform.startswith("win") else '-fopenmp'
include_dirs = [np.get_include()]

# Look for the Eigen library in `/usr/include` and `~/.local/include`.
for f in [Path('eigen3'), Path('/usr/include/eigen3'), Path.home().joinpath('.local/include/eigen3')]:
    if f.exists() and f.is_dir():
        include_dirs += [str(f)]
        break
else:
    print("ERROR: Eigen3 library is required!")
    print("NOTE : Please run 'make eigen3'")
    exit(1)

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
        '-DEIGEN_NO_DEBUG',
    ]

extensions = [Extension(
    name="blaster._core",
    sources=["core/blaster.pyx"],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args
)]

setup(
    ext_modules=cythonize(extensions, language_level="3", build_dir='build/cpp'),
    options={'build': {'build_lib': 'src/'}},
)
