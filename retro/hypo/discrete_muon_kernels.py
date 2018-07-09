# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, invalid-name

"""
Discrete-time kernels for muons generating photons, to be used as hypo_kernels
in discrete_hypo/DiscreteHypo class.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'ALL_REALS',
    'MULEN_INTERP',
    'TABLE_LOWER_BOUND',
    'TABLE_UPPER_BOUND',
    'const_energy_loss_muon',
    'table_energy_loss_muon'
]

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
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import (
    COS_CKV, SIN_CKV, THETA_CKV, SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV,
    TRACK_PHOTONS_PER_M, SRC_CKV_BETA1, EMPTY_SOURCES
)
from retro.retro_types import SRC_T


ALL_REALS = (-np.inf, np.inf)

def pegleg_muon(time, x, y, z, track_azimuth, track_zenith, dt, n_segments=3000):
    """Simple discrete-time track hypothesis.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    time, x, y, z, track_azimuth, track_zenith

    dt : float
        Time step in nanoseconds

    n_segments : int
        Number of segments to supply for pegleg

    Returns
    -------
    sources : shape (n_segments,) numpy.ndarray, dtype SRC_T

    """
    sampled_dt = np.arange(dt*0.5, (n_segments + 0.5)*dt, dt)

    segment_length = dt * SPEED_OF_LIGHT_M_PER_NS
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    # NOTE: add pi to make dir vector go in "math-standard" vector notation
    # (vector components point in direction of motion), as opposed to "IceCube"
    # vector notation (vector components point opposite to direction of
    # motion).
    opposite_zenith = np.pi - track_zenith
    opposite_azimuth = np.pi + track_azimuth

    dir_costheta = math.cos(opposite_zenith)
    dir_sintheta = math.sin(opposite_zenith)

    dir_cosphi = np.cos(opposite_azimuth)
    dir_sinphi = np.sin(opposite_azimuth)

    dir_x = dir_sintheta * dir_cosphi
    dir_y = dir_sintheta * dir_sinphi
    dir_z = dir_costheta

    sources = np.empty(shape=sampled_dt.shape, dtype=SRC_T)

    sources['kind'] = SRC_CKV_BETA1
    sources['time'] = time + sampled_dt
    sources['x'] = x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    sources['y'] = y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    sources['z'] = z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    sources['photons'] = photons_per_segment

    sources['dir_costheta'] = dir_costheta
    sources['dir_sintheta'] = dir_sintheta

    sources['dir_cosphi'] = dir_cosphi
    sources['dir_sinphi'] = dir_sinphi

    sources['ckv_theta'] = THETA_CKV
    sources['ckv_costheta'] = COS_CKV
    sources['ckv_sintheta'] = SIN_CKV

    return sources


def const_energy_loss_muon(time, x, y, z, track_energy, track_azimuth, track_zenith, dt):
    """Simple discrete-time track hypothesis.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    time, x, y, z, track_energy, track_azimuth, track_zenith

    dt : float
        Time step in nanoseconds

    Returns
    -------
    sources : shape (N,) numpy.ndarray, dtype SRC_T

    """
    if track_energy == 0:
        return EMPTY_SOURCES

    length = track_energy * TRACK_M_PER_GEV

    sampled_dt = np.arange(dt*0.5, length/SPEED_OF_LIGHT_M_PER_NS, dt)
    # At least one segment
    if len(sampled_dt) == 0:
        sampled_dt = np.array([length/2./SPEED_OF_LIGHT_M_PER_NS])

    segment_length = dt * SPEED_OF_LIGHT_M_PER_NS
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    # NOTE: add pi to make dir vector go in "math-standard" vector notation
    # (vector components point in direction of motion), as opposed to "IceCube"
    # vector notation (vector components point opposite to direction of
    # motion).
    opposite_zenith = np.pi - track_zenith
    opposite_azimuth = np.pi + track_azimuth

    dir_costheta = math.cos(opposite_zenith)
    dir_sintheta = math.sin(opposite_zenith)

    dir_cosphi = np.cos(opposite_azimuth)
    dir_sinphi = np.sin(opposite_azimuth)

    dir_x = dir_sintheta * dir_cosphi
    dir_y = dir_sintheta * dir_sinphi
    dir_z = dir_costheta

    sources = np.empty(shape=sampled_dt.shape, dtype=SRC_T)

    sources['kind'] = SRC_CKV_BETA1
    sources['time'] = time + sampled_dt
    sources['x'] = x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    sources['y'] = y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    sources['z'] = z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    sources['photons'] = photons_per_segment

    sources['dir_costheta'] = dir_costheta
    sources['dir_sintheta'] = dir_sintheta

    sources['dir_cosphi'] = dir_cosphi
    sources['dir_sinphi'] = dir_sinphi

    sources['ckv_theta'] = THETA_CKV
    sources['ckv_costheta'] = COS_CKV
    sources['ckv_sintheta'] = SIN_CKV

    return sources


# Create spline (for table_energy_loss_muon)
with open(join(RETRO_DIR, 'data', 'dedx_total_e.csv'), 'r') as csvfile:
    # pylint: disable=invalid-name
    reader = csv.reader(csvfile)
    rows = []
    for row in reader:
        rows.append(row)

energies = np.array([float(x) for x in rows[0][1:]])

TABLE_UPPER_BOUND = np.max(energies)
TABLE_LOWER_BOUND = np.min(energies)

stopping_power = np.array([float(x) for x in rows[1][1:]])
dxde = interpolate.UnivariateSpline(x=energies, y=1/stopping_power, s=0, k=3)
esamps = np.logspace(np.log10(TABLE_LOWER_BOUND), np.log10(TABLE_UPPER_BOUND), int(1e4))
dxde_samps = np.clip(dxde(esamps), a_min=0, a_max=np.inf)

lengths = [0]
for idx, egy in enumerate(esamps[1:]):
    lengths.append(np.trapz(y=dxde_samps[:idx+1], x=esamps[:idx+1]))
lengths = np.clip(np.array(lengths), a_min=0, a_max=np.inf)

MULEN_INTERP = interpolate.UnivariateSpline(x=esamps, y=lengths, k=1, s=0)
# does that work? :P
MUEN_INTERP = interpolate.UnivariateSpline(y=esamps[1:], x=lengths[1:], k=1, s=0)


def table_energy_loss_muon(
        time, x, y, z, track_energy, track_azimuth, track_zenith, dt
):
    """Discrete-time track hypothesis that calculates dE/dx as the muon travels
    using splined tabulated data.

    Use as a hypo_kernel with DiscreteHypo class.

    Parameters
    ----------
    time, x, y, z, track_energy, track_azimuth, track_zenith

    dt : float
        Time step in nanoseconds

    Returns
    -------
    sources : shape (N,) numpy.ndarray, dtype SRC_T

    """
    # Check for no-track condition
    if track_energy == 0:
        return EMPTY_SOURCES

    if track_energy > TABLE_UPPER_BOUND:
        raise ValueError('Make sure to set energy bounds such that track_energy'
                         ' cannot exceed table upper limit of {:.3f}'
                         ' GeV'.format(TABLE_UPPER_BOUND))

    # Total expected length of muon from table
    length = MULEN_INTERP(track_energy)

    # Since table cuts off, this can be 0 even for track_energy != 0
    if length == 0:
        return EMPTY_SOURCES

    sampled_dt = np.arange(dt*0.5, length/SPEED_OF_LIGHT_M_PER_NS, dt)
    # At least one segment
    if len(sampled_dt) == 0:
        sampled_dt = np.array([length/2./SPEED_OF_LIGHT_M_PER_NS])

    segment_length = dt * SPEED_OF_LIGHT_M_PER_NS
    photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

    # Assign dir_x, dir_y, dir_z for the track
    opposite_zenith = np.pi - track_zenith
    opposite_azimuth = np.pi + track_azimuth

    dir_costheta = math.cos(opposite_zenith)
    dir_sintheta = math.sin(opposite_zenith)

    dir_cosphi = np.cos(opposite_azimuth)
    dir_sinphi = np.sin(opposite_azimuth)

    dir_x = dir_sintheta * dir_cosphi
    dir_y = dir_sintheta * dir_sinphi
    dir_z = dir_costheta

    sources = np.empty(shape=sampled_dt.shape, dtype=SRC_T)

    sources['kind'] = SRC_CKV_BETA1
    sources['time'] = time + sampled_dt
    sources['x'] = x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
    sources['y'] = y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
    sources['z'] = z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
    sources['photons'] = photons_per_segment

    sources['dir_costheta'] = dir_costheta
    sources['dir_sintheta'] = dir_sintheta

    sources['dir_cosphi'] = dir_cosphi
    sources['dir_sinphi'] = dir_sinphi

    sources['ckv_theta'] = THETA_CKV
    sources['ckv_costheta'] = COS_CKV
    sources['ckv_sintheta'] = SIN_CKV

    return sources


def pegleg_eval(pegleg_idx, dt, const_e_loss):
    """Convert a pegleg index into track energy in GeV.

    Parameters
    ----------
    pegleg_idx : int
    dt : float
    const_e_loss : bool

    Returns
    -------
    muon_energy : float

    """
    length = pegleg_idx / dt * SPEED_OF_LIGHT_M_PER_NS
    if const_e_loss:
        return length * TRACK_M_PER_GEV
    return MUEN_INTERP(length)
