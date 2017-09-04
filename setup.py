from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'retro.shift_and_bin',
        ['retro/shift_and_bin.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='shift_and_bin',
    ext_modules=cythonize(ext_modules)
)
