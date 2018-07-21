# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, range-builtin-not-iterating

"""
Physical constants and constant-for-us values
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'omkeys_to_sd_indices', 'get_sd_idx', 'get_string_dom_pair',

    # Constants
    'PI', 'TWO_PI', 'PI_BY_TWO', 'SPEED_OF_LIGHT_M_PER_NS',

    # Pre-calculated values
    'COS_CKV', 'THETA_CKV', 'SIN_CKV',
    'TRACK_M_PER_GEV', 'TRACK_PHOTONS_PER_M', 'CASCADE_PHOTONS_PER_GEV',
    'IC_DOM_JITTER_NS', 'DC_DOM_JITTER_NS', 'POL_TABLE_DCOSTHETA',
    'POL_TABLE_DRPWR', 'POL_TABLE_DT', 'POL_TABLE_RPWR', 'POL_TABLE_RMAX',
    'POL_TABLE_NTBINS', 'POL_TABLE_NRBINS', 'POL_TABLE_NTHETABINS',
    'IC_DOM_QUANT_EFF', 'DC_DOM_QUANT_EFF',

    # Particle naming conventions
    'ABS_FLAV_STR', 'ABS_FLAV_TEX', 'BAR_NOBAR_STR', 'BAR_NOBAR_TEX',
    'INT_TYPE_STR', 'INT_TYPE_TEX', 'PDG_STR', 'PDG_TEX', 'PDG_INTER_STR',
    'PDG_INTER_TEX', 'STR_TO_PDG_INTER',

    # "Enum"-like things
    'STR_ALL', 'STR_IC', 'STR_DC', 'AGG_STR_NONE', 'AGG_STR_ALL',
    'AGG_STR_SUBDET', 'DOM_ALL',

    'NUM_STRINGS', 'NUM_DOMS_PER_STRING', 'NUM_DOMS_TOT',

    'IC_STRS', 'DC_STRS', 'DC_IC_STRS', 'DC_ALL_STRS', 'DC_SUBDUST_DOMS',
    'IC_SUBDUST_DOMS', 'DC_SUBDUST_STRS_DOMS', 'DC_IC_SUBDUST_STRS_DOMS',
    'DC_ALL_SUBDUST_STRS_DOMS', 'ALL_STRS', 'ALL_DOMS', 'ALL_STRS_DOMS',
    'ALL_STRS_DOMS_SET', 'DC_ALL_STRS_DOMS',

    'EMPTY_HITS', 'EMPTY_SOURCES',
    'SRC_OMNI', 'SRC_CKV_BETA1',

    'PARAM_NAMES', 'PEGLEG_PARAM_NAMES', 'SCALING_PARAM_NAMES',
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

from itertools import product
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import FTYPE
from retro import retro_types


def omkeys_to_sd_indices(omkeys):
    """Get a single integer index from OMKeys.

    Parameters
    ----------
    omkeys : array of dtype OMKEY_T
        The dtype `OMKEY_T` must contain "string" and "dom" and can optionally
        include "pmt".

    Returns
    -------
    sd_idx : array of np.uint32

    """
    if 'pmt' in omkeys.dtype.names:
        raise NotImplementedError("OMKey field 'pmt' not implemented")
    return get_sd_idx(string=omkeys['string'], dom=omkeys['dom'])


def get_sd_idx(string, dom, pmt=0):
    """Get a single integer index from an IceCube string number (from 1 to 86)
    and DOM number (from 1 to 60).

    Parameters
    ----------
    string : int in [1, 60]
        String number
    dom : int in [1, 60]
        DOM number
    pmt : int
        PMT number in the DOM; if == 0, then this is ignored.

    Returns
    -------
    sd_idx : int

    """
    if pmt > 0:
        raise NotImplementedError('PMT != 0 is not implemented')
    return (dom - 1) * NUM_STRINGS + (string - 1)


def get_string_dom_pair(sd_idx):
    """Get an IceCube string number (1 to 86) and a DOM number (1 to 60) from
    the single-integer index (sd_idx).

    Parameters
    ----------
    sd_idx : int in [0, 5159]

    Returns
    -------
    string : int in [1, 86]
    dom : int in [1, 60]

    """
    dom_idx, string_idx = divmod(sd_idx, NUM_STRINGS)
    string = string_idx + 1
    dom = dom_idx + 1
    return string, dom


# -- Physical / mathematical constants -- #

PI = FTYPE(np.pi)
"""pi"""

TWO_PI = FTYPE(2*np.pi)
"""2 * pi"""

PI_BY_TWO = FTYPE(np.pi / 2)
"""pi / 2"""

SPEED_OF_LIGHT_M_PER_NS = FTYPE(299792458 / 1e9)
"""Speed of light in units of m/ns"""


# -- Pre-calculated values -- #

COS_CKV = 0.764540803152
"""Cosine of the Cherenkov angle for beta ~1 and IceCube phase index as used"""

THETA_CKV = np.arccos(0.764540803152)
"""Cosine of the Cherenkov angle for beta ~1 and IceCube phase index as used"""

SIN_CKV = np.sin(THETA_CKV)
"""Cosine of the Cherenkov angle for beta ~1 and IceCube phase index as used"""

TRACK_M_PER_GEV = FTYPE(15 / 3.3)
"""Track length per energy, in units of m/GeV"""

TRACK_PHOTONS_PER_M = FTYPE(2451.4544553)
"""Track photons per length, in units of 1/m (see ``nphotons.py``)"""

CASCADE_PHOTONS_PER_GEV = FTYPE(12805.3383311)
"""Cascade photons per energy, in units of 1/GeV (see ``nphotons.py``)"""

# TODO: Is jitter same (or close enough to the same) for all DOMs? Is it
#       different for DeepCore vs. non-DeepCore DOMs? Didn't see as much in
#       section 3.3. of arXiv:1612.05093v2 so assuming same for now.

# See arXiv:1612.05093v2, section 3.3
IC_DOM_JITTER_NS = 1.7
"""Timing jitter (stddev) for string 0-79 DOMs, in units of ns"""

# See arXiv:1612.05093v2, section 3.3
DC_DOM_JITTER_NS = 1.7
"""Timing jitter (stddev) for DeepCore (strings 80-86) DOMs, in units of ns"""

# TODO: figure these out from the tables rather than defining as constants
POL_TABLE_RMAX = 400 # m
POL_TABLE_DT = 10 # ns
POL_TABLE_RPWR = 2
POL_TABLE_DRPWR = 0.1
POL_TABLE_DCOSTHETA = -0.05
POL_TABLE_NTBINS = 300
POL_TABLE_NRBINS = 200
POL_TABLE_NTHETABINS = 40

#IC_DOM_QUANT_EFF = 0.25
IC_DOM_QUANT_EFF = 1.
"""scalar in [0, 1] : (Very rough approximation!) IceCube (i.e. non-DeepCore)
DOM quantum efficiency. Multiplies the tabulated detection probabilities to
yield the actual probabilitiy that a photon is detected."""
#DC_DOM_QUANT_EFF = 0.35
DC_DOM_QUANT_EFF = 1.
"""scalar in [0, 1] : (Very rough approximation!) DeepCore DOM quantum
efficiency. Multiplies the tabulated detection probabilities to yield the
actual probabilitiy that a photon is detected."""


# -- Particle / interaction type naming conventions -- #

ABS_FLAV_STR = {12: 'nue', 13: 'numu', 14: 'nutau'}
ABS_FLAV_TEX = {12: r'\nu_e', 13: r'\nu_\mu', 14: r'\nu_\tau'}

BAR_NOBAR_STR = {-1: 'bar', 1: ''}
BAR_NOBAR_TEX = {-1: r'\bar', 1: ''}

INT_TYPE_STR = {1: 'cc', 2: 'nc'}
INT_TYPE_TEX = {1: r'\, {\rm CC}', 2: r'\, {\rm NC}'}

PDG_STR = {}
PDG_TEX = {}
for _bnb, _abs_code in product(BAR_NOBAR_STR.keys(), ABS_FLAV_STR.keys()):
    PDG_STR[_abs_code*_bnb] = ABS_FLAV_STR[_abs_code] + BAR_NOBAR_STR[_bnb]
    PDG_TEX[_abs_code*_bnb] = BAR_NOBAR_TEX[_bnb] + ABS_FLAV_TEX[_abs_code]

PDG_INTER_STR = {}
PDG_INTER_TEX = {}
for _pdg, _it in product(PDG_STR.keys(), INT_TYPE_STR.keys()):
    PDG_INTER_STR[(_pdg, _it)] = '%s_%s' % (PDG_STR[_pdg], INT_TYPE_STR[_it])
    PDG_INTER_TEX[(_pdg, _it)] = '%s %s' % (PDG_TEX[_pdg], INT_TYPE_TEX[_it])

STR_TO_PDG_INTER = {v: k for k, v in PDG_INTER_STR.items()}


# -- "enums" -- #
STR_ALL, STR_IC, STR_DC = -1, -2, -3
AGG_STR_NONE, AGG_STR_ALL, AGG_STR_SUBDET = 0, 1, 2
DOM_ALL = -1

# -- geom constants --- #

NUM_STRINGS = 86
NUM_DOMS_PER_STRING = 60
NUM_DOMS_TOT = NUM_STRINGS * NUM_DOMS_PER_STRING


IC_STRS = np.array(range(1, 78+1), dtype=np.uint8)
DC_STRS = np.array(range(79, 86+1), dtype=np.uint8)
DC_IC_STRS = np.array([26, 27, 35, 36, 37, 45, 46], dtype=np.uint8)
DC_ALL_STRS = np.concatenate([DC_STRS, DC_IC_STRS], axis=0)

DC_SUBDUST_DOMS = np.array(range(11, 60+1), dtype=np.uint8)
IC_SUBDUST_DOMS = np.array(range(25, 60+1), dtype=np.uint8)

DC_SUBDUST_STRS_DOMS = np.array(
    [get_sd_idx(s, d) for s, d in product(DC_STRS, DC_SUBDUST_DOMS)]
)
DC_IC_SUBDUST_STRS_DOMS = np.array(
    [get_sd_idx(s, d) for s, d in product(DC_IC_STRS, IC_SUBDUST_DOMS)]
)

DC_ALL_SUBDUST_STRS_DOMS = np.concatenate(
    (DC_SUBDUST_STRS_DOMS, DC_IC_SUBDUST_STRS_DOMS)
)

ALL_STRS = list(range(1, 86+1))
ALL_DOMS = list(range(1, 60+1))
ALL_STRS_DOMS = np.array([get_sd_idx(s, d) for s, d in product(ALL_STRS, ALL_DOMS)])
ALL_STRS_DOMS_SET = set(ALL_STRS_DOMS)
DC_ALL_STRS_DOMS = np.array([get_sd_idx(s, d) for s, d in product(DC_STRS, ALL_DOMS)])


EMPTY_HITS = np.empty(shape=0, dtype=retro_types.HIT_T)

EMPTY_SOURCES = np.empty(shape=0, dtype=retro_types.SRC_T)

SRC_OMNI = np.uint32(0)
"""Source kind designator for a point emitting omnidirectional light"""

SRC_CKV_BETA1 = np.uint32(1)
"""Source kind designator for a point emitting Cherenkov light with beta ~ 1"""


PARAM_NAMES = [
    'time', 'x', 'y', 'z', 'track_azimuth', 'track_zenith', 'cascade_azimuth',
    'cascade_zenith', 'track_energy', 'cascade_energy', 'cascade_d_zenith',
    'cascade_d_azimuth'
]
"""All possible hypothesis param names"""

PEGLEG_PARAM_NAMES = ['track_energy']
"""Hypothesis param names handled by pegleg, if it's used"""

SCALING_PARAM_NAMES = ['cascade_energy']
"""Hypothesis param names handled by scaling, if it's used"""
