# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Physical constants and constant-for-us values
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    # Constants
    'SPEED_OF_LIGHT_M_PER_NS', 'PI', 'TWO_PI', 'PI_BY_TWO',

    # Pre-calculated values
    'TRACK_M_PER_GEV', 'TRACK_PHOTONS_PER_M', 'CASCADE_PHOTONS_PER_GEV',
    'IC_DOM_JITTER_NS', 'DC_DOM_JITTER_NS', 'POL_TABLE_DCOSTHETA',
    'POL_TABLE_DRPWR', 'POL_TABLE_DT', 'POL_TABLE_RPWR', 'POL_TABLE_RMAX',
    'POL_TABLE_NTBINS', 'POL_TABLE_NRBINS', 'POL_TABLE_NTHETABINS',
    'IC_DOM_QUANT_EFF', 'DC_DOM_QUANT_EFF',

    # Particle naming conventions
    'ABS_FLAV_STR', 'ABS_FLAV_TEX', 'BAR_NOBAR_STR', 'BAR_NOBAR_TEX',
    'INT_TYPE_STR', 'INT_TYPE_TEX', 'PDG_STR', 'PDG_TEX', 'PDG_INTER_STR',
    'PDG_INTER_TEX', 'STR_TO_PDG_INTER',
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
