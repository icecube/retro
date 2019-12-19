# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name


"""
Retro Reco: global defs
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    '__version__',
    'MissingOrInvalidPrefitError',
    'NUMBA_AVAIL',
    'numba_jit',
    'RETRO_DIR',
    'DATA_DIR',
    'FTYPE',
    'UITYPE',
    'DEBUG',
    'DFLT_NUMBA_JIT_KWARGS',
    'PL_NUMBA_JIT_KWARGS',
    'DFLT_PULSE_SERIES',
    'DFLT_ML_RECO_NAME',
    'DFLT_SPE_RECO_NAME',
    'DETECTOR_GEOM_FILE',
    'DETECTOR_GCD_DICT_FILE',
    'PY2',
    'PY3',
    'load_pickle'
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

from collections import namedtuple, OrderedDict
try:
    from collections import Iterable, Mapping, Sequence
except ImportError:
    from collections.abc import Iterable, Mapping, Sequence
from itertools import product
import math
from os import environ
from os.path import abspath, basename, dirname, expanduser, expandvars, join
import re
import sys
from time import time

from six import PY2, PY3
from six.moves import cPickle as pickle
import numpy as np

from ._version import get_versions


__version__ = get_versions()['version']
del get_versions


class MissingOrInvalidPrefitError(ValueError):
    """Prefit / seed fit missing or invalid"""
    pass


NUMBA_AVAIL = False
def _dummy_func(x):
    """Decorate to to see if Numba actually works"""
    x += 1
try:
    from numba import jit as numba_jit
    from numba import vectorize as numba_vectorize
    numba_jit(_dummy_func)
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

if 'RETRO_DATA_DIR' in environ:
    DATA_DIR = environ['RETRO_DATA_DIR']
else:
    DATA_DIR = join(RETRO_DIR, 'retro_data')


# -- Datatype choices for consistency throughout code -- #

FTYPE = np.float32
"""Datatype to use for explicitly-typed floating point numbers"""

UITYPE = np.int64
"""Datatype to use for explicitly-typed unsigned integers"""

# -- Default choices we've made -- #

DEBUG = 0
"""Level of debug messages to display"""

DFLT_NUMBA_JIT_KWARGS = dict(
    nopython=True,
    nogil=True,
    fastmath=False,
    cache=True,
    error_model='numpy',
)
"""kwargs to pass to numba.jit"""
PL_NUMBA_JIT_KWARGS = dict(
    nopython=True,
    nogil=True,
    fastmath=False,
    cache=True,
    error_model='numpy',
    parallel=True,
)
"""kwargs to pass to numba.jit for parallel computation"""

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

DETECTOR_GEOM_FILE = join(RETRO_DIR, 'retro_data', 'geo_array.npy')
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""

DETECTOR_GCD_DICT_FILE = join(
    dirname(dirname(abspath(__file__))), 'retro_data', 'gcd_dict.pkl'
)
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""


def load_pickle(path):
    """Load a pickle file, independent of Python2 or Python3.

    Parameters
    ----------
    path : string
        Filepath (will be expanded).

    Returns
    -------
    obj

    """
    expanded_path = expanduser(expandvars(path))
    try:
        with open(expanded_path, 'rb') as fobj:
            if PY2:
                return pickle.load(fobj)
            return pickle.load(fobj, encoding='latin1')
    except:
        sys.stderr.write('Failed to load pickle at path "{}"\n'.format(expanded_path))
        raise
