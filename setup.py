# -*- coding: utf-8 -*-

"""
Installation script for the Retro project
"""

from __future__ import absolute_import

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


EXT_MODULES = [
    Extension(
        'retro.tables.sphbin2cartbin',
        ['retro/tables/sphbin2cartbin.pyx'],
    ),
    Extension(
        'retro.tables.shift_and_bin',
        ['retro/tables/shift_and_bin.pyx'],
    )
]

setup(
    name='retro',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(EXT_MODULES)
)
