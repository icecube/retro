# pylint: disable=wrong-import-position

"""
Discrete-time models for muons generating photons, to be used as hypo_kernels
in discrete_hypo/DiscreteHypo class.
"""


from __future__ import absolute_import, division, print_function

import math
import os
from os.path import abspath, dirname

import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV,
                   TRACK_PHOTONS_PER_M)


__all__ = ['ALL_REALS', 'const_energy_loss_muon']


ALL_REALS = (-np.inf, np.inf)


# TODO: use / check limits...?
def const_energy_loss_muon(hypo_params, limits=None, dt=1.0):
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

    track_energy = hypo_params.track_energy

    if track_energy == 0:
        pinfo_gen = np.array(
            [[hypo_params.t,
              hypo_params.x,
              hypo_params.y,
              hypo_params.z,
              0,
              0,
              0,
              0]],
            dtype=np.float64
        )
        return pinfo_gen

    length = track_energy * TRACK_M_PER_GEV
    duration = length / SPEED_OF_LIGHT_M_PER_NS
    n_samples = int(np.floor(duration / dt))
    first_sample_t = dt/2
    final_sample_t = first_sample_t + (n_samples - 1) * dt
    segment_length = length / n_samples
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    sin_zen = math.sin(hypo_params.track_zenith)
    dir_x = -sin_zen * math.cos(hypo_params.track_azimuth)
    dir_y = -sin_zen * math.sin(hypo_params.track_azimuth)
    dir_z = -math.cos(hypo_params.track_zenith)

    pinfo_gen = np.empty((n_samples, 8), dtype=np.float64)
    sampled_dt = np.linspace(dt*0.5, dt * (n_samples - 0.5), n_samples)
    pinfo_gen[:, 0] = hypo_params.t + dt
    pinfo_gen[:, 1] = hypo_params.x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 2] = hypo_params.y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 3] = hypo_params.z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 4] = photons_per_segment
    pinfo_gen[:, 5] = dir_x * 0.562
    pinfo_gen[:, 6] = dir_y * 0.562
    pinfo_gen[:, 7] = dir_z * 0.562

    return pinfo_gen
