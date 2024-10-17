# =============================================================================
# @author: Brian Kyanjo
# @date: 2024-09-24
# @description: This file will be used to compile and build the cython code
# =============================================================================

# import libraries
import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Note: make sure enkf.pyx is in the same directory as this file
# Change to the directory with the cython file
os.chdir("src/EnKF/cython_enkf") 

# clean the directory
os.system("rm -rf build* *.so enkf.c out") 

# setup the cython file using current directory
setup(
    ext_modules = cythonize("enkf.pyx"),  # Cython file to be compiled
    include_dirs=[np.get_include()],  # Ensure numpy headers are 
    compiler_directives={'language_level' : "3"}, # Compile for python 3
    script_args=["build", "--build-lib","."] # Build the .so file in the current directory
)

# copy the .so file to the current directory
os.system("cp EnKF/cython_enkf/* .")

# remove the EnKF directory
os.system("rm -rf EnKF")

# Usage: python setup.py build_ext --inplace
#      This will create a .so file which can be imported in python
