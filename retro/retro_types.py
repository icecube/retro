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
    'DOM_INFO',
    'LLHP8D',
    'LLHP10D'
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

import numpy as np


HypoParams8D = namedtuple( # pylint: disable=invalid-name
    typename='HypoParams8D',
    field_names=('t', 'x', 'y', 'z', 'track_zenith', 'track_azimuth',
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
    field_names=('t', 'x', 'y', 'z', 'track_zenith', 'track_azimuth',
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
    field_names=('x', 'y', 'z', 't', 'wavelength', 'coszen', 'azimuth')
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
    field_names=('t',) + Cart3DCoord._fields
)
"""Time and Cartesian 3D coordinate: t, x, y, z."""

TimePolCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimePolCoord',
    field_names=('t',) + PolCoord._fields
)
"""Time and polar coordinate: t, r, theta."""

TimeSphCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeSphCoord',
    field_names=('t',) + SphCoord._fields
)
"""Time and spherical coordinate: t, r, theta, phi."""

DOM_INFO = np.dtype(
    [
        ('operational', np.bool),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('quantum_efficiency', np.float32),
        ('noise_rate_per_ns', np.float32)
    ],
    align=True
)

LLHP8D = np.dtype(
    [('llh', np.float16)]
    + [(field, np.float16) for field in HypoParams8D._fields]
)

LLHP10D = np.dtype(
    [('llh', np.float16)]
    + [(field, np.float16) for field in HypoParams10D._fields]
)
