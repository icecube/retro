"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""


from __future__ import absolute_import, division, print_function

from collections import Iterable

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

    kernel_kwargs : None or iterable the same length as `hypo_kernels` of dicts
    limits : TimeCart3DCoord of 2-tuples

    """
    def __init__(self, hypo_kernels, kernel_kwargs=None, limits=None):
        if callable(hypo_kernels):
            hypo_kernels = [hypo_kernels]
        assert isinstance(hypo_kernels, Iterable)
        for kernel in hypo_kernels:
            assert callable(kernel)
        self.hypo_kernels = hypo_kernels
        if kernel_kwargs is None:
            kernel_kwargs = [{} for _ in hypo_kernels]
        assert isinstance(kernel_kwargs, Iterable)
        if limits is not None:
            for kwargs in kernel_kwargs:
                if 'limits' not in kwargs:
                    kwargs['limits'] = limits
        self.kernel_kwargs = kernel_kwargs
        self.limits = limits

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
        for kernel, kwargs in zip(self.hypo_kernels, self.kernel_kwargs):
            pinfo_gen_arrays.append(kernel(hypo_params, **kwargs))
        pinfo_gen = np.concatenate(pinfo_gen_arrays, axis=0)
        return pinfo_gen
