# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'DiscreteHypo'
]

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from collections import Mapping
from copy import deepcopy

import numpy as np
import numba
import inspect

from retro.const import EMPTY_SOURCES
from retro_types import PARAM_NAMES

def get_hypo_args(kernel):
    '''
    get the hypo args from a kernel by inspecting it
    '''
    if isinstance(kernel, numba.targets.registry.CPUDispatcher):
        py_func = kernel.py_func
    else:
        py_func = kernel
    kernel_args = inspect.getargspec(py_func)[0]
    hypo_args = set(kernel_args) & set(PARAM_NAMES)
    return list(hypo_args)



class DiscreteHypo(object):
    """Discretely-sampled event hypothesis.

    This calls any number of kernel functions to enable mixed-physics
    hypotheses without requiring all of the complexity be in a single kernel.

    Parameters
    ----------
    hypo_kernels : callable or iterable thereof
        Each kernel must accept at least the argument `hypo_params`. Any
        further arguments must be keyword arguments stored (in order of the
        kernels) in `kernel_kwargs`, and will be passed via **kwargs to the
        respective kernel function.

    kernel_kwargs : None, dict, or iterable thereof (len == len(`hypo_kernels`)
        Each dict contains keyword arguments to pass on to the respective
        kernel via **kwargs. An item in the iterable can be None for a kernel
        function that takes no additional kwargs.

    pegleg_kernel : callable

    pegleg_kernel_kwargs : None or dict

    """
    def __init__(self, hypo_kernels, kernel_kwargs=None, pegleg_kernel=None, pegleg_kernel_kwargs=None):
        # If a single kernel is passed, make it into a singleton list
        if callable(hypo_kernels):
            hypo_kernels = [hypo_kernels]

        # Make sure items in hypo_kernels are callables (functions or methods)
        for kernel in hypo_kernels:
            assert callable(kernel)

        if pegleg_kernel is not None:
            assert callable(pegleg_kernel)

        # If None or dict is passed, duplicate it to send to each hypo kernel
        if kernel_kwargs is None or isinstance(kernel_kwargs, Mapping):
            kernel_kwargs = [deepcopy(kernel_kwargs) for _ in hypo_kernels]

        # Translate each None into an empty dict
        kernel_kwargs = [{} if kw is None else kw for kw in kernel_kwargs]

        kernel_args = []
        for kernel in hypo_kernels:
            kernel_args.append(get_hypo_args(kernel))


        if pegleg_kernel is None:
            pegleg_kernel_args = []
        else:
            pegleg_kernel_args = get_hypo_args(pegleg_kernel)

        self.hypo_kernels = hypo_kernels
        self.kernel_kwargs = kernel_kwargs
        self.kernel_args = kernel_args
        self.pegleg_kernel = pegleg_kernel
        self.pegleg_kernel_kwargs = pegleg_kernel_kwargs
        self.pegleg_kernel_args = pegleg_kernel_args

    def get_sources(self, hypo):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters and return the sources produced by the hypothesis.

        Parameters
        ----------
        hypo : dict
            This is a module-level constant defined in ``__init__.py``, e.g.
            retro.HypoParams8D. See docstring on the `HYPO_PARAMS_T` defined
            for the specification of `hypo_params` including units.

        Returns
        -------
        sources : shape (N, len(`SRC_T`)) numpy.ndarray
            Each row is a `SRC_T`.

        """
        sources = []
        for kernel, args, kwargs in zip(self.hypo_kernels, self.kernel_args, self.kernel_kwargs):
            total_kwargs = {a:hypo[a] for a in args}
            total_kwargs.update(kwargs)
            sources.append(kernel(**total_kwargs))
        sources = np.concatenate(sources, axis=0)
        sources.sort(order='time')
        return sources

    def get_pegleg_sources(self, hypo):
        """Evaluate a discrete hypothesis supporting pegleg given particular
        parameters and return the array of sources produced by the hypothesis.

        Parameters
        ----------
        hypo : dict 
            This is a module-level constant defined in ``__init__.py``, e.g.
            retro.HypoParams8D. See docstring on the `HYPO_PARAMS_T` defined
            for the specification of `hypo_params` including units.

        Returns
        -------
        sources : shape (N, len(`SRC_T`)) numpy.ndarray
            Each row is a `SRC_T`.

        """
        if self.pegleg_kernel is None:
            return EMPTY_SOURCES
        total_kwargs = {a:hypo[a] for a in self.pegleg_kernel_args}
        total_kwargs.update(self.pegleg_kernel_kwargs)
        return self.pegleg_kernel(**total_kwargs)

    @property
    def params(self):
        '''
        return all used hypo parameter dimensions
        '''
        params = []
        params.extend(self.pegleg_kernel_args)
        for kernel_args in self.kernel_args:
            params.extend(kernel_args)
        params = set(params)
        return list(params)

    @property
    def pegleg_params(self):
        if self.pegleg_kernel is None:
            return []
        else:
            return ['track_energy']
