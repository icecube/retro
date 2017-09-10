from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        'retro.sphbin2cartbin',
        ['retro/sphbin2cartbin.pyx'],
    ),
    Extension(
        'retro.shift_and_bin',
        ['retro/shift_and_bin.pyx'],
    )
]

setup(
    name='retro',
    include_dirs=[np.get_include()],
    ext_modules=cythonize(ext_modules),
)
