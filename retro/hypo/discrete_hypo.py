# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    SRC_DTYPE
    SRC_OMNI
    SRC_CKV_BETA1
    DiscreteHypo
'''.split()

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


SRC_DTYPE = np.dtype(
    [
        ('kind', np.uint32),
        ('t', np.float32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('photons', np.float32),
        ('dir_x', np.float32),
        ('dir_y', np.float32),
        ('dir_z', np.float32)
    ],
    align=True
)
"""Each source point is described by (up to) these 9 fields"""

SRC_OMNI = np.uint32(0)
"""Source kind designator for a point emitting omnidirectional light"""

SRC_CKV_BETA1 = np.uint32(1)
"""Source kind designator for a point emitting Cherenkov light with beta ~ 1"""


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

    """
    def __init__(self, hypo_kernels, kernel_kwargs=None):
        # If a single kernel is passed, make it into a singleton list
        if callable(hypo_kernels):
            hypo_kernels = [hypo_kernels]

        # Make sure items in hypo_kernels are callables (functions or methods)
        for kernel in hypo_kernels:
            assert callable(kernel)

        # If None or dict is passed, duplicate it to send to each hypo kernel
        if kernel_kwargs is None or isinstance(kernel_kwargs, Mapping):
            kernel_kwargs = [deepcopy(kernel_kwargs) for _ in hypo_kernels]

        # Translate each None into an empty dict
        kernel_kwargs = [{} if kw is None else kw for kw in kernel_kwargs]

        self.hypo_kernels = hypo_kernels
        self.kernel_kwargs = kernel_kwargs

    def get_sources(self, hypo_params):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters and return the sources produced by the hypothesis.

        Parameters
        ----------
        hypo_params : HYPO_PARAMS_T
            This is a module-level constant defined in ``__init__.py``, e.g.
            retro.HypoParams8D. See docstring on the `HYPO_PARAMS_T` defined
            for the specification of `hypo_params` including units.

        Returns
        -------
        sources : shape (N, len(`SRC_DTYPE`)) numpy.ndarray
            Each row is a `SRC_DTYPE`.

        """
        sources = []
        for kernel, kwargs in zip(self.hypo_kernels, self.kernel_kwargs):
            print(type(hypo_params))
            print(hypo_params)
            print(type(kwargs))
            print(kwargs)
            sources.append(kernel(hypo_params, **kwargs))
        return np.concatenate(sources, axis=0)
