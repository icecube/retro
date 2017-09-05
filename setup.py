from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        'retro.sphbin2cartbin',
        ['retro/sphbin2cartbin.pyx'],
    ),
    Extension(
        'retro.shift_and_bin',
        ['retro/shift_and_bin.pyx'],
        #extra_compile_args=['-fopenmp'],
        #extra_link_args=['-fopenmp']
    )
]

setup(
    name='retro',
    ext_modules=cythonize(ext_modules)
)
