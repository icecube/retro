# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
namedtuples for interface simplicity and consistency
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'HypoParams8D',
    'HypoParams10D',
    'TrackParams',
    'Hit',
    'Photon',
    'Event',
    'RetroPhotonInfo',
    'HypoPhotonInfo',
    'Cart2DCoord',
    'Cart3DCoord',
    'PolCoord',
    'SphCoord',
    'TimeCart3DCoord',
    'TimePolCoord',
    'TimeSphCoord',
    'DOM_INFO_T',
    'LLHP8D_T',
    'LLHP10D_T',
    'HIT_T',
    'SD_INDEXER_T',
    'HITS_SUMMARY_T',
    'TypeID',
    'SourceID',
    'SubtypeID',
    'TRIGGER_T',
    'SRC_T'
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

from collections import namedtuple
import enum

import numpy as np


HypoParams8D = namedtuple( # pylint: disable=invalid-name
    typename='HypoParams8D',
    field_names=('time', 'x', 'y', 'z', 'track_zenith', 'track_azimuth',
                 'track_energy', 'cascade_energy')
)
"""Hypothesis in 8 dimensions (parameters). Units are: t/ns, {x,y,z}/m,
{track_zenith,track_azimuth}/rad, {track_energy,cascade_energy}/GeV"""

HypoParams10D = namedtuple( # pylint: disable=invalid-name
    typename='HypoParams10D',
    field_names=(HypoParams8D._fields + ('cascade_zenith', 'cascade_azimuth'))
)
"""Hypothesis in 10 dimensions (parameters). Units are: t/ns, {x,y,z}/m,
{track_zenith,track_azimuth}/rad, {track_energy,cascade_energy}/GeV,
{cascade_zenith,cascade_azimuth}/rad"""

TrackParams = namedtuple( # pylint: disable=invalid-name
    typename='TrackParams',
    field_names=('time', 'x', 'y', 'z', 'track_zenith', 'track_azimuth',
                 'track_energy')
)
"""Hypothesis for just the track (7 dimensions / parameters). Units are: t/ns,
{x,y,z}/m, {track_zenith,track_azimuth}/rad, track_energy/GeV"""

Event = namedtuple( # pylint: disable=invalid-name
    typename='Event',
    field_names=('filename', 'event', 'uid', 'pulses', 'interaction',
                 'neutrino', 'track', 'cascade', 'ml_reco', 'spe_reco')
)

Hit = namedtuple(
    typename='Hit',
    field_names=('time', 'charge', 'width')
)

Photon = namedtuple(
    typename='Photon',
    field_names=('x', 'y', 'z', 'time', 'wavelength', 'coszen', 'azimuth')
)

RetroPhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='RetroPhotonInfo',
    field_names=('survival_prob', 'time_indep_survival_prob', 'theta', 'deltaphi', 'length')
)
"""Info contained in (original) retro tables: Photon survival probability
(survival_prob) and average photon direction and length (theta, deltaphi,
length). `deltaphi` is the direction in the azimuthal direction relative to
the bin center's azimuth (phi) direction. Note that directions are expected to
follow "standard" spherical coordinates where direction of vector is the
direciton in which it points, NOT the direction from which it comes (as is the
astro / IceCube convention). Intended to contain dictionaries with DOM depth
index as keys and arrays as values."""

HypoPhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='HypoPhotonInfo',
    field_names=('count', 'theta', 'phi', 'length')
)
"""Info contained in (original) retro tables: Photon survival probability
(survival_prob) and average photon direction and length (theta, phi, length).
Note that directions are expected to follow "standard" spherical coordinates
where direction of vector is the direciton in which it points, NOT the
direction from which it comes (as is the astro / IceCube convention). Intended
to contain dictionaries with DOM depth index as keys and arrays as values."""

Cart2DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Cart2DCoord',
    field_names=('x', 'y'))
"""Cartesian 2D coordinate: x, y."""

Cart3DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Cart3DCoord',
    field_names=('x', 'y', 'z'))
"""Cartesian 3D coordinate: x, y, z."""

PolCoord = namedtuple( # pylint: disable=invalid-name
    typename='PolCoord',
    field_names=('r', 'theta')
)
"""2D polar coordinate: r, theta."""

SphCoord = namedtuple( # pylint: disable=invalid-name
    typename='SphCoord',
    field_names=('r', 'theta', 'phi')
)
"""3D spherical coordinate: r, theta, and phi."""

TimeCart3DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Time3DCartCoord',
    field_names=('time',) + Cart3DCoord._fields
)
"""Time and Cartesian 3D coordinate: t, x, y, z."""

TimePolCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimePolCoord',
    field_names=('time',) + PolCoord._fields
)
"""Time and polar coordinate: t, r, theta."""

TimeSphCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeSphCoord',
    field_names=('time',) + SphCoord._fields
)
"""Time and spherical coordinate: t, r, theta, phi."""

DOM_INFO_T = np.dtype(
    [
        ('operational', np.bool),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('quantum_efficiency', np.float32),
        ('noise_rate_per_ns', np.float32)
    ]
)

LLHP8D_T = np.dtype(
    [('llh', np.float32)]
    + [(field, np.float32) for field in HypoParams8D._fields]
)

LLHP10D_T = np.dtype(
    [('llh', np.float32)]
    + [(field, np.float32) for field in HypoParams10D._fields]
)

PULSE_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32),
    ('width', np.float32),
])

PHOTON_T = np.dtype([
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('coszen', np.float32),
    ('azimuth', np.float32),
    ('wavelength', np.float32),
])

HIT_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32)
])

#EVENT_INDEXER_T = np.dtype([
#    ('first_idx', np.uint32),
#    ('num', np.uint32)
#])

SD_INDEXER_T = np.dtype([
    ('sd_idx', np.uint32),
    ('offset', np.uint64),
    ('num', np.uint32)
])

HITS_SUMMARY_T = np.dtype([
    ('earliest_hit_time', np.float32),
    ('latest_hit_time', np.float32),
    ('average_hit_time', np.float32),
    ('total_charge', np.float32),
    ('total_num_hits', np.uint32),
    ('total_num_doms_hit', np.uint32),
    ('time_window_start', np.float32),
    ('time_window_stop', np.float32)
])


class ConfigID(enum.IntEnum):
    """Trigger common names mapped into ConfigID (or config_id) in i3 files.

    Note that this seems to be a really unique ID for a trigger, subsuming
    TypeID and SourceID.

    See docs at
      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    """
    SMT8_IN_ICE = 1006
    SMT3_DeepCore = 1011
    SMT6_ICE_TOP = 102
    SLOP = 24002
    Cluster = 1007
    Cylinder = 21001

    # Added for 2016-2017
    Cylinder_ICE_TOP = 21002

    # Unique to 2011
    SLOP_2011 = 22005

    # Unique to IC79
    SMT3_DeepCore_IC79 = 1010


class TypeID(enum.IntEnum):
    SIMPLE_MULTIPLICITY = 0
    CALIBRATION = 10
    MIN_BIAS = 20
    THROUGHPUT = 30
    TWO_COINCIDENCE = 40
    THREE_COINCIDENCE = 50
    MERGED = 70
    SLOW_PARTICLE = 80
    FRAGMENT_MULTIPLICITY = 105
    STRING = 120
    VOLUME = 125
    SPHERE = 127
    UNBIASED = 129
    SPASE_2 = 170
    UNKNOWN_TYPE = 180


class SourceID(enum.IntEnum):
    IN_ICE = 0
    ICE_TOP = 10
    AMANDA_TWR_DAQ = 20
    EXTERNAL = 30
    GLOBAL = 40
    AMANDA_MUON_DAQ = 50
    SPASE = 70
    UNKNOWN_SOURCE = 80


class SubtypeID(enum.IntEnum):
    NO_SUBTYPE = 0
    M18 = 50
    M24 = 100
    T0 = 150
    LASER = 200
    UNKNOWN_SUBTYPE = 250


TRIGGER_T = np.dtype([
    ('type', np.uint8),
    ('subtype', np.uint8),
    ('source', np.uint8),
    ('config_id', np.int32),
    ('fired', np.bool),
    ('time', np.float32),
    ('length', np.float32)
])


SRC_T = np.dtype(
    [
        ('kind', np.uint32),
        ('time', np.float32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('photons', np.float32),
        ('dir_costheta', np.float32),
        ('dir_sintheta', np.float32),
        ('dir_cosphi', np.float32),
        ('dir_sinphi', np.float32),
        ('ckv_theta', np.float32),
        ('ckv_costheta', np.float32),
        ('ckv_sintheta', np.float32),
    ],
    align=True
)
"""Each source point is described by (up to) these 9 fields"""
