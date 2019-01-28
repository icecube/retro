# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Tools for code inspection
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_arg_names']

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2019 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import inspect

import numba


def get_arg_names(func):
    """Extract argument names from a pure-Python or Numba jit-compiled function.

    Parameters
    ----------
    func : callable

    Returns
    -------
    arg_names : tuple of strings

    """
    if isinstance(func, numba.targets.registry.CPUDispatcher):
        py_func = func.py_func
    else:
        py_func = func

    # Get all the function's argument names
    arg_names = inspect.getargspec(py_func).args

    return tuple(arg_names)
