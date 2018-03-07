# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Discrete-time kernels for muons generating photons, to be used as hypo_kernels
in discrete_hypo/DiscreteHypo class.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    ALL_REALS
    const_energy_loss_muon
    table_energy_loss_muon
'''.split()

__author__ = 'P. Eller, J.L. Lanfranchi, K. Crust'
__license__ = '''Copyright 2017 Philipp Eller, Justin L. Lanfranchi, Kevin Crust

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import csv
import math
from os.path import abspath, dirname, join
import sys

import numpy as np
from scipy import interpolate

RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS
from retro.const import (
    COS_CKV, SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV, TRACK_PHOTONS_PER_M
)

print(RETRO_DIR)

ALL_REALS = (-np.inf, np.inf)


# Create spline (for table_energy_loss_muon)
with open(join(RETRO_DIR, 'data', 'dedx_total_e.csv'), 'rb') as csvfile:
    # pylint: disable=invalid-name
    reader = csv.reader(csvfile)
    rows = []
    for row in reader:
        rows.append(row)
    energies = rows[0]
    stopping_power = rows[1]
    energies.pop(0)
    stopping_power.pop(0)
    idx = 0
    while idx < len(energies):
        energies[idx] = float(energies[idx])
        stopping_power[idx] = float(stopping_power[idx])
        idx += 1

SPLINE = interpolate.splrep(energies, stopping_power, s=0)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def const_energy_loss_muon(hypo_params, dt=1.0):
    """Simple discrete-time track hypothesis.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams*
        Must have vertex (`.t`, `.x`, `.y`, and `.z), `.track_energy`,
        `.track_azimuth`, and `.track_zenith` attributes.

    dt : float
        Time step in nanoseconds

    Returns
    -------
    pinfo_gen : shape (N, 8) numpy.ndarray, dtype float32

    """
    track_energy = hypo_params.track_energy

    if track_energy == 0:
        pinfo_gen = np.array([
            hypo_params.t,
            hypo_params.x,
            hypo_params.y,
            hypo_params.z,
            0.0,
            0.0,
            0.0,
            0.0
        ], dtype=np.float32).reshape((1, 8))
        return pinfo_gen

    length = track_energy * TRACK_M_PER_GEV
    duration = length / SPEED_OF_LIGHT_M_PER_NS
    n_samples = int(np.floor(duration / dt))
    segment_length = 0.0
    if n_samples > 0:
        segment_length = length / n_samples
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    sin_zen = math.sin(hypo_params.track_zenith)
    dir_x = -sin_zen * math.cos(hypo_params.track_azimuth)
    dir_y = -sin_zen * math.sin(hypo_params.track_azimuth)
    dir_z = -math.cos(hypo_params.track_zenith)

    pinfo_gen = np.empty((n_samples, 8), dtype=np.float32)
    sampled_dt = np.linspace(dt*0.5, dt * (n_samples - 0.5), n_samples)
    pinfo_gen[:, 0] = hypo_params.t + sampled_dt
    pinfo_gen[:, 1] = hypo_params.x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 2] = hypo_params.y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 3] = hypo_params.z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    pinfo_gen[:, 4] = photons_per_segment
    pinfo_gen[:, 5] = dir_x * COS_CKV
    pinfo_gen[:, 6] = dir_y * COS_CKV
    pinfo_gen[:, 7] = dir_z * COS_CKV

    return pinfo_gen


def table_energy_loss_muon(hypo_params, dt=1.0):
    """Discrete-time track hypothesis that calculates dE/dx as the muon travels
    using splined tabulated data.

    Use as a hypo_kernel with DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams*
        Must have vertex (`.t`, `.x`, `.y`, and `.z), `.track_energy`,
        `.track_azimuth`, and `.track_zenith` attributes.

    dt : float
        Time step in nanoseconds

    Returns
    -------
    pinfo_gen : shape (N,8) numpy.ndarray, dtype float 32
    """
    track_energy = hypo_params.track_energy

    # Check for only cascade
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
            dtype=np.float32
        )
        return pinfo_gen

    # Declare constants
    segment_length = dt * SPEED_OF_LIGHT_M_PER_NS
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    # Assign vertex
    t = hypo_params.t
    x = hypo_params.x
    y = hypo_params.y
    z = hypo_params.z

    # Assign dir_x, dir_y, dir_z for the track
    #NOTE: ask justin about the negative sign on zenith
    sin_zen = math.sin(hypo_params.track_zenith)
    dir_x = -sin_zen * math.cos(hypo_params.track_azimuth)
    dir_y = -sin_zen * math.sin(hypo_params.track_azimuth)
    dir_z = -math.cos(hypo_params.track_zenith)

    # Create array at vertex
    photon_array = [
        (t, x, y, z, photons_per_segment, dir_x * COS_CKV, dir_y * COS_CKV, dir_z * COS_CKV)
    ]

    dx = dt * dir_x * SPEED_OF_LIGHT_M_PER_NS
    dy = dt * dir_y * SPEED_OF_LIGHT_M_PER_NS
    dz = dt * dir_z * SPEED_OF_LIGHT_M_PER_NS

    # Loop uses rest mass of 105.658 MeV/c^2 for muon
    rest_mass = 0.105658

    # Loop through track, appending new photon dump on axis 0
    while True:
        # Move along track
        t += dt
        x += dx
        y += dy
        z += dz

        # Change energy of muon
        dedx = interpolate.splev(track_energy, SPLINE, der=0)
        track_energy -= dedx * segment_length

        # Append new row if energy still above rest mass, else break out of
        # loop
        if track_energy > rest_mass:
            photon_array.append(
                (t, x, y, z, photons_per_segment, dir_x * COS_CKV, dir_y * COS_CKV, dir_z * COS_CKV)
            )
        else:
            break

    return np.array(photon_array, dtype=np.float32)
