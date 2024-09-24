# This file will be used to compile and build the cython code

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Note: make sure enkf.pyx is in the same directory as this file

setup(
    ext_modules = cythonize("enkf.pyx"),  # Cython file to be compiled
    include_dirs=[np.get_include()]  # Ensure numpy headers are included
)

# Usage: python setup.py build_ext --inplace
#      This will create a .so file which can be imported in python
