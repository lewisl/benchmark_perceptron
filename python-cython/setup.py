# -*- coding: utf-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize('ps1perceptroncython.pyx'), requires=['numba', 'numpy'])
# at cmdline: python setup.py build_ext --inplace