# -*- coding: utf-8 -*-

"""
Installation script for the Retro project
"""

from __future__ import absolute_import

from setuptools import setup, Extension, find_packages

from Cython.Build import cythonize
import numpy as np
import versioneer


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
    name='retro-reco',
    description=(
        'Reverse table reconstruction for of neutrino events in ice/water'
        ' Cherenkov detectors'
    ),
    author='Philipp Eller and Justin Lanfranchi',
    author_email='pde3@psu.edu',
    url='https://github.com/philippeller/retro',
    license='Apache 2.0',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires='>=2.7',
    setup_requires=[
        'pip>=1.8',
        'setuptools>18.5',
        'cython',
        'numpy>=1.11'
    ],
    install_requires=[
        'astropy',
        'enum34',
        'funcsigs',
        'matplotlib>=2.0',
        'numba>=0.37',
        'pymultinest',
        'scipy>=0.17',
        'sobol',
        'sobol_seq',
        'dynesty',
        'spherical_opt @ git+https://github.com/philippeller/spherical_opt.git',
        'zstandard',
    ],
    packages=find_packages(),
    include_dirs=[np.get_include()],
    package_data={
        'retro.tables': ['*.pyx',],
        '' : ['*.csv', '*.pkl'],
    },
    #data_files=[('retro_data', ['retro_data/*.csv',]),],
    ext_modules=cythonize(EXT_MODULES),
    zip_safe=False
)
