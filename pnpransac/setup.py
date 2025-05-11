import sys
import os
from distutils.core import setup
from distutils.extension import Extension

import cv2
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

# where to find opencv headers and libraries
cv_include_dir = os.path.join(sys.prefix, 'include')
cv_library_dir = os.path.join(sys.prefix, 'lib')
lib_dir = [numpy.get_include(), cv_include_dir, cv_library_dir, ]

ext_modules = [
    Extension(
        "pnpransac",
        sources=["pnpransacpy.pyx"],
        language="c++",
        libraries=['opencv_core', 'opencv_calib3d'],
        extra_compile_args=['-fopenmp', '-std=c++11'],
    )
]

setup(
    name='pnpransac',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    include_dirs=lib_dir,
)
