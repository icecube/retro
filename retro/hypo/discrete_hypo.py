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

from collections import OrderedDict
try:
    from collections.abc import Mapping
except Exception:
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
from retro.const import (
    EMPTY_SOURCES, PARAM_NAMES, PEGLEG_PARAM_NAMES, SCALING_PARAM_NAMES
)


def get_hypo_param_names(kernel):
    """Get the hypothesis parameter names that a hypo kernel takes based on the
    argument names specified by the function.

    Parameters
    ----------
    kernel : callable

    Returns
    -------
    hypo_param_names : tuple

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
    return tuple(n for n in PARAM_NAMES if n in kernel_argnames)


class DiscreteHypo(object):
    """Discretely-sampled event hypothesis.

    This calls any number of kernel functions to enable mixed-physics
    hypotheses without requiring all of the complexity be in a single kernel.

    the pegleg kernel is a track hypothesis that is sampled along a dimension,
    which is used to determine the track energy in an inner loop

    The scaling kernel is a hypothesis that can be adjusted with a scalefactor
    for the light yield allowing to analytically calculate the maximum
    likelihood for its parameter

    Parameters
    ----------
    generic_kernels : callable or iterable thereof
        Each kernel must accept at least the argument `hypo_params`. Any
        further arguments must be keyword arguments stored (in order of the
        kernels) in `generic_kernels_kwargs`, and will be passed via **kwargs
        to the respective kernel function.

    generic_kernels_kwargs : None, dict, or iterable thereof with len(`generic_kernels`) elements
        Each dict contains keyword arguments to pass on to the respective
        kernel via **kwargs. An item in the iterable can be None for a kernel
        function that takes no additional kwargs.

    pegleg_kernel : callable

    pegleg_kernel_kwargs : None or dict

    scaling_kernel : callable

    scaling_kernel_kwargs : None or dict

    """
    def __init__(
        self,
        generic_kernels,
        generic_kernels_kwargs=None,
        pegleg_kernel=None,
        pegleg_kernel_kwargs=None,
        scaling_kernel=None,
        scaling_kernel_kwargs=None,
    ):
        # If a single generic kernel is passed, make it into a singleton list
        if callable(generic_kernels):
            generic_kernels = [generic_kernels]

        for kernel in generic_kernels:
            assert callable(kernel), str(kernel)
        assert pegleg_kernel is None or callable(pegleg_kernel), str(pegleg_kernel)
        assert scaling_kernel is None or callable(scaling_kernel), str(scaling_kernel)

        self.generic_kernels = generic_kernels
        self.pegleg_kernel = pegleg_kernel
        self.scaling_kernel = scaling_kernel

        # If None or dict is passed, duplicate it to send to each generic
        # kernel
        if generic_kernels_kwargs is None or isinstance(generic_kernels_kwargs, Mapping):
            generic_kernels_kwargs = [
                deepcopy(generic_kernels_kwargs) for _ in generic_kernels
            ]

        # Translate each None into an empty dict
        self.generic_kernels_kwargs = [kw or {} for kw in generic_kernels_kwargs]
        self.pegleg_kernel_kwargs = pegleg_kernel_kwargs or {}
        self.scaling_kernel_kwargs = scaling_kernel_kwargs or {}

        self.generic_param_names = tuple(
            get_hypo_param_names(kernel) for kernel in generic_kernels
        )
        """One set of param names for each kernel (possible to have duplicated names)"""

        self.pegleg_param_names = ()
        """All param names required by the pegleg kernel (if any)"""

        if pegleg_kernel:
            self.pegleg_param_names = tuple(get_hypo_param_names(pegleg_kernel))

        self.scaling_param_names = ()
        """All param names required by the scaling kernel (if any)"""

        if scaling_kernel:
            self.scaling_param_names = tuple(get_hypo_param_names(scaling_kernel))

        # Place param names first that are handled by external optimizers (e.g.
        # MultiNest), then include name(s) of paramters handled by pegleg and
        # finally scaling param names. It is required to do this for e.g.
        # MultiNest where the first n_opt_params params are optimized over
        # while the remaining dimensions are left alone.
        all_param_names = []
        for param_names in self.generic_param_names:
            for param_name in param_names:
                if param_name not in all_param_names:
                    all_param_names.append(param_name)
        for param_name in self.pegleg_param_names + self.scaling_param_names:
            if param_name not in all_param_names:
                all_param_names.append(param_name)
        self._opt_param_names = tuple(all_param_names)
        self.fixed_params = OrderedDict()

    @property
    def opt_param_names(self):
        """tuple of strings : Hypothesis parameter names to be handled by a
        generic optimizer, i.e., non-fixed and not including pegleg and
        scaling parameters"""
        # remove fixed params
        return tuple(n for n in self._opt_param_names if n not in self.fixed_params)

    @property
    def hypo_param_names(self):
        """list of strings : Fixed and free hypothesis parameter names"""
        return self._opt_param_names

    @property
    def all_param_names(self):
        """tuple of strings : All parameter names: fixed, free, pegleg, and
        scaling parameters"""
        all_param_names = list(self._opt_param_names)
        if self.pegleg_kernel is not None:
            for param_name in PEGLEG_PARAM_NAMES:
                if param_name not in all_param_names:
                    all_param_names.append(param_name)
        if self.scaling_kernel is not None:
            for param_name in SCALING_PARAM_NAMES:
                if param_name not in all_param_names:
                    all_param_names.append(param_name)
        return all_param_names

    @property
    def n_params(self):
        """int : Number of parameters (including all parameters kinds)"""
        return len(self.all_param_names)

    @property
    def n_hypo_params(self):
        """int : Number of hypothesis parameters"""
        return len(self.hypo_param_names)

    @property
    def n_opt_params(self):
        """int: Number of hypothesis parameters to be handled by a generic optimizer"""
        return len(self.opt_param_names)

    def get_generic_sources(self, hypo):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters and return the sources produced by the hypothesis.

        Parameters
        ----------
        hypo : dict

        Returns
        -------
        sources : shape (n_generic_sources,) array of dtype SRC_T

        """
        hypo.update(self.fixed_params)
        sources = []
        for kernel, param_names, kwargs in zip(
            self.generic_kernels,
            self.generic_param_names,
            self.generic_kernels_kwargs
        ):
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
        sources : shape (n_pegleg_sources,) array of dtype SRC_T

        """
        hypo.update(self.fixed_params)
        if self.pegleg_kernel is None:
            return EMPTY_SOURCES
        total_kwargs = {a:hypo[a] for a in self.pegleg_param_names}
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
        sources : shape (n_scaling_sources,) array of dtype SRC_T

        """
        hypo.update(self.fixed_params)
        if self.scaling_kernel is None:
            return EMPTY_SOURCES
        total_kwargs = {a:hypo[a] for a in self.scaling_param_names}
        total_kwargs.update(self.scaling_kernel_kwargs)
        return self.scaling_kernel(**total_kwargs)
