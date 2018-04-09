# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

from __future__ import absolute_import, division, print_function

__all__ = '''
    RETRO_DIR
    DATA_DIR
    NUMBA_AVAIL
    FTYPE
    UITYPE
    HYPO_PARAMS_T
    LLHP_T
    DEBUG
    DFLT_NUMBA_JIT_KWARGS
    DFLT_PULSE_SERIES
    DFLT_ML_RECO_NAME
    DFLT_SPE_RECO_NAME
    DETECTOR_GEOM_FILE
    DETECTOR_GCD_DICT_FILE
'''.split()

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

from collections import namedtuple, OrderedDict, Iterable, Mapping, Sequence
import cPickle as pickle
from itertools import product
import math
from os import environ
from os.path import abspath, basename, dirname, expanduser, expandvars, join
import re
import sys
from time import time

import numpy as np

NUMBA_AVAIL = False
def dummy_func(x):
    """Decorate to to see if Numba actually works"""
    x += 1
try:
    from numba import jit as numba_jit
    from numba import vectorize as numba_vectorize
    numba_jit(dummy_func)
except Exception:
    #logging.debug('Failed to import or use numba', exc_info=True)
    def numba_jit(*args, **kwargs): # pylint: disable=unused-argument
        """Dummy decorator to replace `numba.jit` when Numba is not present"""
        def decorator(func):
            """Decorator that smply returns the function being decorated"""
            return func
        return decorator
    numba_vectorize = numba_jit # pylint: disable=invalid-name
else:
    NUMBA_AVAIL = True

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import HypoParams8D, HypoParams10D, LLHP8D_T, LLHP10D_T

if 'RETRO_DATA_DIR' in environ:
    DATA_DIR = environ['RETRO_DATA_DIR']
else:
    DATA_DIR = join(RETRO_DIR, 'data')


# -- Datatype choices for consistency throughout code -- #

FTYPE = np.float64
"""Datatype to use for explicitly-typed floating point numbers"""

UITYPE = np.int64
"""Datatype to use for explicitly-typed unsigned integers"""

HYPO_PARAMS_T = HypoParams10D
"""Global selection of which hypothesis to use throughout the code."""

LLHP_T = LLHP8D_T if HYPO_PARAMS_T is HypoParams8D else LLHP10D_T
"""Global selection of LLH/params dtype."""


# -- Default choices we've made -- #

DEBUG = 0
"""Level of debug messages to display"""

DFLT_NUMBA_JIT_KWARGS = dict(nopython=True, nogil=True, fastmath=True, cache=True)
"""kwargs to pass to numba.jit"""

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

DETECTOR_GEOM_FILE = join(RETRO_DIR, 'data', 'geo_array.npy')
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""

DETECTOR_GCD_DICT_FILE = join(dirname(dirname(abspath(__file__))), 'data', 'gcd_dict.pkl')
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""
