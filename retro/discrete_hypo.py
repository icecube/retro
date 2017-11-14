"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""


from __future__ import absolute_import, division, print_function

from collections import Mapping
from copy import deepcopy

import numpy as np


__all__ = ['DiscreteHypo']


class DiscreteHypo(object):
    """Discretely-sampled event hypothesis.

    This calls any number of kernel functions to enable mixed-physics
    hypotheses without requiring all of the complexity be in a single kernel.

    Parameters
    ----------
    hypo_kernels : callable or iterable thereof
        Each kernel must accept at least arguments `hypo_params` and `limits`.
        Any further arguments must be keyword arguments stored (in order of the
        kernels) in `kernel_kwargs`, and will be passed via **kwargs to the
        respective kernel function.

    kernel_kwargs : None, dict, or iterable thereof the same length as `hypo_kernels`
        Each dict contains keyword arguments to pass on to the respective
        kernel via **kwargs. An item in the iterable can be None for a kernel
        function that takes no additional kwargs.

    limits : None or TimeCart3DCoord of 2-tuples
        Rectangular limits in time and space outside of which each hypothesis
        should produce no samples. This can speed up likelihood calculations by
        avoiding samples of light sources so far outside the detector that
        there is negligible probability that the produced light will be
        detected.

    """
    def __init__(self, hypo_kernels, kernel_kwargs=None, limits=None):
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

        # If provided, put `limits` into each kwarg dict to pass on to kernels
        if limits is not None:
            for kw in kernel_kwargs:
                if 'limits' not in kw:
                    kw['limits'] = limits

        self.hypo_kernels = hypo_kernels
        self.kernel_kwargs = kernel_kwargs
        self.limits = limits
        self.kernels_and_kwargs = zip(self.hypo_kernels, self.kernel_kwargs)

    def get_pinfo_gen(self, hypo_params):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters and return the hypothesis's expected generated photons.

        Parameters
        ----------
        hypo_params : HypoParams8D or HypoParams10D

        Returns
        -------
        pinfo_gen : shape (N, 8) numpy.ndarray, dtype float64
            Each row contains (t, x, y, z, p_count, p_x, p_y, p_z)

        """
        pinfo_gen_arrays = []
        for kernel, kwargs in self.kernels_and_kwargs:
            pinfo_gen_arrays.append(kernel(hypo_params, **kwargs))
        pinfo_gen = np.concatenate(pinfo_gen_arrays, axis=0)
        return pinfo_gen
