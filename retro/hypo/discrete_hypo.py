# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_hypo_param_names', 'DiscreteHypo']

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
import inspect
from os.path import abspath, dirname
import sys

import numpy as np
import numba

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import EMPTY_SOURCES
from retro.retro_types import PARAM_NAMES


def get_hypo_param_names(kernel):
    """Get the hypothesis parameter names that a hypo kernel takes based on the
    argument names specified by the function.

    Parameters
    ----------
    kernel

    Returns
    -------
    hypo_param_names : list

    """
    if isinstance(kernel, numba.targets.registry.CPUDispatcher):
        py_func = kernel.py_func
    else:
        py_func = kernel
    # Get all the function's argument names
    kernel_argnames = inspect.getargspec(py_func)[0]

    # Select out the argument names that are "officially recognized" hypo
    # params; this is effectively an "intersection" operation, but where we
    # preserve the ordering of the names from the `PARAM_NAMES` constant
    hypo_param_names = [name for name in PARAM_NAMES if name in kernel_argnames]

    return hypo_param_names


class DiscreteHypo(object):
    """Discretely-sampled event hypothesis.

    This calls any number of kernel functions to enable mixed-physics
    hypotheses without requiring all of the complexity be in a single kernel.

    the pegleg kernel is a track hypothesis that is sampled along a dimension,
    which is used to determine the track energy in an inner loop

    The scaling kernel is a hypothesis that can be adjusted with a scalefactor for the light yield
    allowing to analytically calculate the maximum likelihood for its parameter

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

    scaling_kernel : callable

    scaling_kernel_kwargs : None or dict

    """
    def __init__(self, hypo_kernels, kernel_kwargs=None, pegleg_kernel=None,
                 pegleg_kernel_kwargs=None, scaling_kernel=None,
                 scaling_kernel_kwargs=None):
        # If a single kernel is passed, make it into a singleton list
        if callable(hypo_kernels):
            hypo_kernels = [hypo_kernels]

        # Make sure items in hypo_kernels are callables (functions or methods)
        for kernel in hypo_kernels:
            assert callable(kernel), str(kernel)

        assert pegleg_kernel is None or callable(pegleg_kernel), str(pegleg_kernel)
        assert scaling_kernel is None or callable(scaling_kernel), str(scaling_kernel)

        # If None or dict is passed, duplicate it to send to each hypo kernel
        if kernel_kwargs is None or isinstance(kernel_kwargs, Mapping):
            kernel_kwargs = [deepcopy(kernel_kwargs) for _ in hypo_kernels]

        # Translate each None into an empty dict
        self.kernel_kwargs = [kw or {} for kw in kernel_kwargs]
        self.pegleg_kernel_kwargs = pegleg_kernel_kwargs or {}
        self.scaling_kernel_kwargs = scaling_kernel_kwargs or {}

        # Get hypo param names required by kernels handled by optimization
        # (hypo_kernels), pegleg-ing (pegleg_kernel), and scaling
        # (scaling_kernel)
        self.kernel_param_names = [get_hypo_param_names(k) for k in hypo_kernels]
        self.pegleg_kernel_param_names = (
            get_hypo_param_names(pegleg_kernel) if pegleg_kernel else ()
        )
        self.scaling_kernel_param_names = (
            get_hypo_param_names(scaling_kernel) if scaling_kernel else ()
        )

        self.hypo_kernels = hypo_kernels
        self.pegleg_kernel = pegleg_kernel
        self.scaling_kernel = scaling_kernel

    def get_sources(self, hypo):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters and return the sources produced by the hypothesis.

        Parameters
        ----------
        hypo : dict

        Returns
        -------
        sources : shape (n_sources,) array of dtype SRC_T

        """
        sources = []
        for kernel, param_names, kwargs in zip(self.hypo_kernels, self.kernel_param_names, self.kernel_kwargs):
            total_kwargs = {a:hypo[a] for a in param_names}
            total_kwargs.update(kwargs)
            sources.append(kernel(**total_kwargs))
        if len(sources) == 0:
            return EMPTY_SOURCES
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
        sources : shape (n_sources,) array of dtype SRC_T

        """
        if self.pegleg_kernel is None:
            return EMPTY_SOURCES
        total_kwargs = {a:hypo[a] for a in self.pegleg_kernel_param_names}
        total_kwargs.update(self.pegleg_kernel_kwargs)
        return self.pegleg_kernel(**total_kwargs)

    def get_scaling_sources(self, hypo):
        """Evaluate a discrete hypothesis supporting scaling given particular
        parameters and return the array of sources produced by the hypothesis.

        Parameters
        ----------
        hypo : dict
            This is a module-level constant defined in ``__init__.py``, e.g.
            retro.HypoParams8D. See docstring on the `HYPO_PARAMS_T` defined
            for the specification of `hypo_params` including units.

        Returns
        -------
        sources : shape (n_sources,) array of dtype SRC_T

        """
        if self.scaling_kernel is None:
            return EMPTY_SOURCES
        total_kwargs = {a:hypo[a] for a in self.scaling_kernel_param_names}
        total_kwargs.update(self.scaling_kernel_kwargs)
        return self.scaling_kernel(**total_kwargs)


    @property
    def params(self):
        """Return all used hypo parameter dimensions"""
        params = []
        params.extend(self.pegleg_kernel_param_names)
        params.extend(self.scaling_kernel_param_names)
        for kernel_param_names in self.kernel_param_names:
            params.extend(kernel_param_names)
        params = set(params)
        return list(params)

    @property
    def pegleg_params(self):
        """Return all used hypo parameter dimensions of pegleg kernel"""
        if self.pegleg_kernel is None:
            return []
        return ['track_energy']

    @property
    def scaling_params(self):
        """Return all used hypo parameter dimensions of scaling kernel"""
        if self.scaling_kernel is None:
            return []
        return ['cascade_energy']
