# pylint: disable=wrong-import-position

"""
Simple class DiscreteHypo for evaluating discrete hypotheses.
"""


from __future__ import absolute_import, division, print_function

from collections import Iterable
import math
import os
from os.path import abspath, dirname

import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (SPEED_OF_LIGHT_M_PER_NS, CASCADE_PHOTONS_PER_GEV,
                   TRACK_M_PER_GEV, TRACK_PHOTONS_PER_M)


__all__ = ['ALL_REALS', 'const_energy_loss_muon', 'point_cascade',
           'DiscreteHypo']


ALL_REALS = (-np.inf, np.inf)


# TODO: use / check limits...?
def const_energy_loss_muon(hypo_params, limits=None, dt=1):
    """Simple discrete-time track hypothesis.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams*
        Must have vertex (`.t`, `.x`, `.y`, and `.z), `.track_energy`,
        `.track_azimuth`, and `.track_zenith` attributes.

    limits
        NOT IMPLEMENTED

    dt : float
        Time step in nanoseconds

    Returns
    -------
    pinfo_gen : shape (N, 8) numpy.ndarray, dtype float32

    """
    #if limits is None:
	#    limits = TimeCart3DCoord(t=ALL_REALS, x=ALL_REALS, y=ALL_REALS,
    #                             z=ALL_REALS)

    length = hypo_params.track_energy * TRACK_M_PER_GEV
    duration = length / SPEED_OF_LIGHT_M_PER_NS
    first_sample_t = hypo_params.t + dt/2
    final_sample_t = hypo_params.t + duration - dt/2
    n_samples = int((final_sample_t - first_sample_t) / dt)
    segment_length = length / n_samples
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    sin_zen = math.sin(hypo_params.track_zenith)
    dir_x = sin_zen * math.cos(hypo_params.track_azimuth)
    dir_y = sin_zen * math.sin(hypo_params.track_azimuth)
    dir_z = math.cos(hypo_params.track_zenith)

    pinfo_gen = np.empty((n_samples, 8), dtype=np.float32)
    t = pinfo_gen[:, 0] = np.linspace(first_sample_t, final_sample_t,
                                        n_samples)
    pinfo_gen[:, 1] = hypo_params.x + t * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 2] = hypo_params.y + t * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 3] = hypo_params.z + t * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 4] = photons_per_segment
    pinfo_gen[:, 5] = dir_x * 0.562
    pinfo_gen[:, 6] = dir_y * 0.562
    pinfo_gen[:, 7] = dir_z * 0.562

    return pinfo_gen


def point_cascade(hypo_params, limits=None):
    """Point-like cascade.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams8D or HypoParams10D
    limits

    Returns
    -------
    pinfo_gen

    """
    pinfo_gen = np.empty((1, 8), dtype=np.float32)
    pinfo_gen[0, 0] = hypo_params.t
    pinfo_gen[0, 1] = hypo_params.x
    pinfo_gen[0, 2] = hypo_params.y
    pinfo_gen[0, 3] = hypo_params.z
    pinfo_gen[0, 4] = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy
    pinfo_gen[0, 5] = 0
    pinfo_gen[0, 6] = 0
    pinfo_gen[0, 7] = 0
    return pinfo_gen


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
            kernel_kwargs = [{} for _ in range(hypo_kernels)]
        assert isinstance(kernel_kwargs, Iterable)
        if limits is not None:
            for kwargs in kernel_kwargs:
                if 'limits' not in kwargs:
                    kwargs['limits'] = limits
        self.kernel_kwargs = kernel_kwargs
        self.limits = limits

    def get_photon_gen_info(self, hypo_params):
        """Evaluate the discrete hypothesis (all hypo kernels) given particular
        parameters to yield the hypothesis's expected generated photons.

        Parameters
        ----------
        hypo_params : HypoParams8D or HypoParams10D

        Returns
        -------
        pinfo_gen : shape (N, 8) numpy.ndarray, dtype float32
            Each row contains (t, x, y, z, p_count, p_x, p_y, p_z)

        """
        pinfo_gen_arrays = []
        for kernel, kwargs in zip(self.hypo_kernels, self.kernel_kwargs):
            pinfo_gen_arrays.append(kernel(hypo_params, **kwargs))
        pinfo_gen = np.concatenate(pinfo_gen_arrays, axis=0)
        return pinfo_gen
