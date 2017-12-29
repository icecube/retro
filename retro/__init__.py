"""
Basic module-wide definitions and simple types (namedtuples).
"""


from __future__ import absolute_import, division, print_function


__all__ = [
    # Defaults
    'DFLT_NUMBA_JIT_KWARGS', 'DFLT_PULSE_SERIES', 'DFLT_ML_RECO_NAME',
    'DFLT_SPE_RECO_NAME',
    'CLSIM_TABLE_FNAME_PROTO', 'CLSIM_TABLE_FNAME_RE',
    'CLSIM_TABLE_METANAME_PROTO', 'CLSIM_TABLE_METANAME_RE',
    'RETRO_DOM_TABLE_FNAME_PROTO', 'RETRO_DOM_TABLE_FNAME_RE',
    'GEOM_FILE_PROTO',
    'GEOM_META_PROTO', 'DETECTOR_GEOM_FILE', 'TDI_TABLE_FNAME_PROTO',
    'TDI_TABLE_FNAME_RE', 'NUMBA_AVAIL',

    # Type/namedtuple definitions
    'HypoParams8D', 'HypoParams10D', 'TrackParams', 'Event', 'Pulses',
    'RetroPhotonInfo', 'HypoPhotonInfo', 'Cart2DCoord', 'Cart3DCoord',
    'PolCoord', 'SphCoord', 'TimeCart3DCoord', 'TimeSphCoord',

    # Type selections
    'FTYPE', 'UITYPE', 'HYPO_PARAMS_T',

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

    # Functions...

    # Generic utils
    'expand', 'mkdir', 'force_little_endian', 'hash_obj', 'test_hash_obj',
    'get_file_md5', 'convert_to_namedtuple',

    # Retro-specific functions
    'event_to_hypo_params', 'hypo_to_track_params', 'generate_anisotropy_str',
    'generate_geom_meta', 'generate_unique_ids', 'get_primary_interaction_str',
    'get_primary_interaction_tex', 'interpret_clsim_table_fname',

    # Binning / geometry
    'linbin', 'test_linbin', 'powerbin', 'test_powerbin',
    'powerspace', 'inv_power_2nd_diff', 'infer_power', 'test_infer_power',
    'bin_edges_to_binspec', 'linear_bin_centers', 'spherical_volume',
    'sph2cart', 'pol2cart', 'cart2pol', 'spacetime_separation',

    # Other math
    'poisson_llh', 'partial_poisson_llh', 'weighted_average',

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


import base64
from collections import namedtuple, OrderedDict, Iterable, Mapping, Sequence
import cPickle as pickle
import errno
import hashlib
from itertools import product
import math
from numbers import Number
from os import makedirs
from os.path import abspath, basename, dirname, expanduser, expandvars, join
import re
import struct
from time import time

import numpy as np
import pyfits
from scipy.optimize import brentq
from scipy.special import gammaln

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

# -- Default choices we've made -- #

DFLT_NUMBA_JIT_KWARGS = dict(nopython=True, nogil=True, cache=True)
"""kwargs to pass to numba.jit"""

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

CLSIM_TABLE_FNAME_PROTO = (
    'clsim_table'
    '_set_{hash_val:s}'
    '_string_{string}'
    '_depth_{depth_idx:d}'
    '_seed_{seed}'
    '.fits'
)
"""String template for CLSim ("raw") retro tables. Note that `string` can
either be a specific string number OR either "ic" or "dc" indicating a generic
DOM of one of these two types located at the center of the detector, where z
location is averaged over all DOMs. `seed` can either be an integer or a
human-readable range (e.g. "0-9" for a table that combines toegether seeds, 0,
1, ..., 9)"""

CLSIM_TABLE_FNAME_RE = re.compile(
    r'''
    clsim_table
    _set_(?P<hash_val>[0-9a-f]+)
    _string_(?P<string>[0-9a-z]+)
    _depth_(?P<depth_idx>[0-9]+)
    _seed_(?P<seed>[0-9]+)
    \.fits
    ''', re.IGNORECASE | re.VERBOSE
)

CLSIM_TABLE_METANAME_PROTO = 'clsim_table_set_{hash_val:s}_meta.json'

CLSIM_TABLE_METANAME_RE = re.compile(
    r'''
    clsim_table
    _set_(?P<hash_val>[0-9a-f]+)
    _meta
    \.json
    ''', re.IGNORECASE | re.VERBOSE
)

#IC_RAW_TABLE_FNAME_PROTO = 'retro_nevts1000_IC_DOM{depth_idx:d}.fits'
#"""String template for IceCube single-DOM raw retro tables"""

#DC_RAW_TABLE_FNAME_PROTO = 'retro_nevts1000_DC_DOM{depth_idx:d}.fits'
#"""String template for DeepCore single-DOM raw retro tables"""

RETRO_DOM_TABLE_FNAME_PROTO = (
    'retro_dom_table'
    '_set_{hash_val:s}'
    '_string_{string}'
    '_depth_{depth_idx:d}'
    '_seed_{seed}'
    '.fits'
)
"""String template for single-DOM "final-level" retro tables"""

RETRO_DOM_TABLE_FNAME_RE = re.compile(
    r'''
    retro_dom_table
    _set_(?P<hash_val>[0-9a-f]+)
    _string_(?P<string>[0-9a-z]+)
    _depth_(?P<depth_idx>[0-9]+)
    _seed_(?P<seed>[0-9]+)
    \.fits
    ''', re.IGNORECASE | re.VERBOSE
)
"""Regex for single-DOM retro tables"""

#IC_TABLE_FNAME_PROTO = 'retro_nevts1000_IC_DOM{depth_idx:d}_r_cz_t_angles.fits'
#"""String template for IceCube single-DOM final-level retro tables"""
#
#DC_TABLE_FNAME_PROTO = 'retro_nevts1000_DC_DOM{depth_idx:d}_r_cz_t_angles.fits'
#"""String template for DeepCore single-DOM final-level retro tables"""

GEOM_FILE_PROTO = 'geom_{hash:s}.npy'
"""File containing detector geometry as a Numpy 5D array with coordinates
(string, om, x, y, z)"""

GEOM_META_PROTO = 'geom_{hash:s}_meta.json'
"""File containing metadata about source of detector geometry"""

DETECTOR_GEOM_FILE = join(dirname(abspath(__file__)), 'data', 'geo_array.npy')
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""

TDI_TABLE_FNAME_PROTO = (
    'retro_tdi_table'
    '_{tdi_hash:s}'
    '_binmap_{binmap_hash:s}'
    '_geom_{geom_hash:s}'
    '_domtbl_{dom_tables_hash:s}'
    '_times_{times_str:s}'
    '_x{x_min:.3f}_{x_max:.3f}'
    '_y{y_min:.3f}_{y_max:.3f}'
    '_z{z_min:.3f}_{z_max:.3f}'
    '_bw{binwidth:.9f}'
    '_anisot_{anisotropy_str:s}'
    '_icqe{ic_dom_quant_eff:.5f}'
    '_dcqe{dc_dom_quant_eff:.5f}'
    '_icexp{ic_exponent:.5f}'
    '_dcexp{dc_exponent:.5f}'
    '_{table_name:s}'
    '.fits'
)
"""Time- and DOM-independent (TDI) table file names follow this template"""

TDI_TABLE_FNAME_RE = re.compile(
    r'^retro_tdi_table'
    r'_(?P<tdi_hash>[^_]+)'
    r'_binmap_(?P<binmap_hash>[^_]+)'
    r'_geom_(?P<geom_hash>[^_]+)'
    r'_domtbl_(?P<dom_tables_hash>[^_]+)'
    r'_times_(?P<times_str>[^_]+)'
    r'_x(?P<x_min>[^_]+)_(?P<x_max>[^_]+)'
    r'_y(?P<y_min>[^_]+)_(?P<y_max>[^_]+)'
    r'_z(?P<z_min>[^_]+)_(?P<z_max>[^_]+)'
    r'_bw(?P<binwidth>[^_]+)'
    r'_anisot_(?P<anisotropy>.+?)'
    r'_icqe(?P<ic_dom_quant_eff>.+?)'
    r'_dcqe(?P<dc_dom_quant_eff>.+?)'
    r'_icexp(?P<ic_exponent>.+?)'
    r'_dcexp(?P<dc_exponent>.+?)'
    r'_(?P<table_name>(avg_photon_x|avg_photon_y|avg_photon_z|survival_prob))'
    r'\.fits$'
    , re.IGNORECASE
)
"""Time- and DOM-independent (TDI) table file names can be found / interpreted
using this regex"""


# -- namedtuples for interface simplicity and consistency -- #

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

Pulses = namedtuple( # pylint: disable=invalid-name
    typename='Pulses',
    field_names=('strings', 'oms', 'times', 'charges')
)

RetroPhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='RetroPhotonInfo',
    field_names=('survival_prob', 'theta', 'deltaphi', 'length')
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


# -- Datatype choices for consistency throughout code -- #

FTYPE = np.float64
"""Datatype to use for explicitly-typed floating point numbers"""

UITYPE = np.int64
"""Datatype to use for explicitly-typed unsigned integers"""

HYPO_PARAMS_T = HypoParams8D
"""Global selection of which hypothesis to use throughout the code."""


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

IC_DOM_QUANT_EFF = 0.25
"""scalar in [0, 1] : (Very rough approximation!) IceCube (i.e. non-DeepCore)
DOM quantum efficiency. Multiplies the tabulated detection probabilities to
yield the actual probabilitiy that a photon is detected."""
DC_DOM_QUANT_EFF = 0.35
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


# -- Functions -- #


# -- Files, dirs, I/O functions -- #

def expand(p):
    """Fully expand a path.

    Parameters
    ----------
    p : string
        Path to expand

    Returns
    -------
    e : string
        Expanded path

    """
    return abspath(expanduser(expandvars(p)))


def mkdir(d, mode=0o0750):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    """
    try:
        makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


def force_little_endian(x):
    """Convert a numpy ndarray to little endian if it isn't already.

    E.g., use when loading from FITS files since that spec is big endian, while
    most CPUs (e.g. Intel) are little endian.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    x : numpy.ndarray
        Same as input `x` with byte order swapped if necessary.

    """
    if np.isscalar(x):
        return x
    if x.dtype.byteorder == '>':
        x = x.byteswap().newbyteorder()
    return x


def hash_obj(obj, prec=None, fmt='hex'):
    """Hash an object (recursively), sorting dict keys first and (optionally)
    rounding floating point values to ensure consistency between invocations on
    effectively-equivalent objects.

    Parameters
    ----------
    obj
        Object to hash

    prec : None, np.float32, or np.float64
        Precision to enforce on numeric values

    fmt : string, one of {'int', 'hex', 'base64'}
        Format to use for hash

    Returns
    -------
    hash_val : int or string
        Hash value, where type is determined by `fmt`

    """
    if isinstance(obj, Mapping):
        hashable = []
        for key in sorted(obj.keys()):
            hashable.append((key, hash_obj(obj[key], prec=prec)))
        hashable = pickle.dumps(hashable)
    elif isinstance(obj, np.ndarray):
        if prec is not None:
            obj = obj.astype(prec)
        hashable = obj.tobytes()
    elif isinstance(obj, Number):
        if prec is not None:
            obj = prec(obj)
        hashable = obj
    elif isinstance(obj, basestring):
        hashable = obj
    elif isinstance(obj, Iterable):
        hashable = tuple(hash_obj(x, prec=prec) for x in obj)
    else:
        raise ValueError('`obj`="{}" is unhandled type {}'
                         .format(obj, type(obj)))

    if not isinstance(hashable, basestring):
        hashable = pickle.dumps(hashable, pickle.HIGHEST_PROTOCOL)

    md5hash = hashlib.md5(hashable)
    if fmt == 'hex':
        hash_val = md5hash.hexdigest()#[:16]
    elif fmt == 'int':
        hash_val, = struct.unpack('<q', md5hash.digest()[:8])
    elif fmt == 'base64':
        hash_val = base64.b64encode(md5hash.digest()[:8], '+-')
    else:
        raise ValueError('Unrecognized `fmt`: "%s"' % fmt)

    return hash_val


def test_hash_obj():
    """Unit tests for `hash_obj` function"""
    obj = {'x': {'one': {1:[1, 2, {1:1, 2:2}], 2:2}, 'two': 2}, 'y': {1:1, 2:2}}
    print('base64:', hash_obj(obj, fmt='base64'))
    print('int   :', hash_obj(obj, fmt='int'))
    print('hex   :', hash_obj(obj, fmt='hex'))
    obj = {
        'r_binning_kw': {'n_bins': 200, 'max': 400, 'power': 2, 'min': 0},
        't_binning_kw': {'n_bins': 300, 'max': 3000, 'min': 0},
        'costhetadir_binning_kw': {'n_bins': 40, 'max': 1, 'min': -1},
        'costheta_binning_kw': {'n_bins': 40, 'max': 1, 'min': -1},
        'deltaphidir_binning_kw': {'n_bins': 80, 'max': 3.141592653589793,
                                   'min': -3.141592653589793},
        'tray_kw_to_hash': {'DisableTilt': True, 'PhotonPrescale': 1,
                            'IceModel': 'spice_mie',
                            'Zenith': 3.141592653589793, 'NEvents': 1,
                            'Azimuth': 0.0, 'Sensor': 'none',
                            'PhotonSource': 'retro'}
    }
    print('hex   :', hash_obj(obj, fmt='hex'))
    obj['r_binning_kw']['n_bins'] = 201
    print('hex   :', hash_obj(obj, fmt='hex'))


def get_file_md5(fpath, blocksize=2**20):
    """Get a file's MD5 checksum.

    Code from stackoverflow.com/a/1131255

    Parameters
    ----------
    fpath : string
        Path to file

    blocksize : int
        Read file in chunks of this many bytes

    Returns
    -------
    md5sum : string
        32-characters representing hex MD5 checksum

    """
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            md5.update(buf)
    return md5.hexdigest()


def convert_to_namedtuple(val, nt_type):
    """Convert ``val`` to a namedtuple of type ``nt_type``.

    If ``val`` is:
    * ``nt_type``: return without conversion
    * Mapping: instantiate an ``nt_type`` via ``**val``
    * Iterable or Sequence: instantiate an ``nt_type`` via ``*val``

    Parameters
    ----------
    val : nt_type, Mapping, Iterable, or Sequence
        Value to be converted

    nt_type : namedtuple type
        Namedtuple type (class)

    Returns
    -------
    nt_val : nt_type
        ``val`` converted to ``nt_type``

    Raises
    ------
    TypeError
        If ``val`` is not one of the above-specified types.

    """
    if isinstance(val, nt_type):
        return val

    if isinstance(val, Mapping):
        return nt_type(**val)

    if isinstance(val, (Iterable, Sequence)):
        return nt_type(*val)

    raise TypeError('Cannot convert %s to %s' % (type(val), nt_type))


# -- Retro-specific functions -- #


def event_to_hypo_params(event):
    """Convert an event to hypothesis params, for purposes of defining "truth"
    hypothesis.

    For now, only works with HypoParams8D.

    Parameters
    ----------
    event : likelihood.Event namedtuple

    Returns
    -------
    params : HYPO_PARAMS_T namedtuple

    """
    assert HYPO_PARAMS_T is HypoParams8D

    track_energy = event.track.energy
    cascade_energy = event.cascade.energy
    #if event.interaction == 1: # charged current
    #    track_energy = event.neutrino.energy
    #    cascade_energy = 0
    #else: # neutral current (2)
    #    track_energy = 0
    #    cascade_energy = event.neutrino.energy

    hypo_params = HYPO_PARAMS_T(
        t=event.neutrino.t,
        x=event.neutrino.x,
        y=event.neutrino.y,
        z=event.neutrino.z,
        track_azimuth=event.neutrino.azimuth,
        track_zenith=event.neutrino.zenith,
        track_energy=track_energy,
        cascade_energy=cascade_energy
    )

    return hypo_params


def hypo_to_track_params(hypo_params):
    """Extract track params from hypo params.

    Parameters
    ----------
    hypo_params : HYPO_PARAMS_T namedtuple

    Returns
    -------
    track_params : TrackParams namedtuple

    """
    track_params = TrackParams(
        t=hypo_params.t,
        x=hypo_params.x,
        y=hypo_params.y,
        z=hypo_params.z,
        zenith=hypo_params.track_zenith,
        azimuth=hypo_params.track_azimuth,
        energy=hypo_params.track_energy
    )
    return track_params


def generate_anisotropy_str(anisotropy):
    """Generate a string from anisotropy specification parameters.

    Parameters
    ----------
    anisotropy : None or tuple of values

    Returns
    -------
    anisotropy_str : string

    """
    if anisotropy is None:
        anisotropy_str = 'none'
    else:
        anisotropy_str = '_'.join(str(param) for param in anisotropy)
    return anisotropy_str


def generate_geom_meta(geom):
    """Generate geometry metadata dict. Currently, this sinmply hashes on the
    geometry coordinates. Note that the values are rounded to the nearest
    centimeter for hashing purposes. (Also, the values are converted to
    integers at this precision to eliminate any possible float32 / float64
    issues that could cause discrepancies in hash values for what we consider
    to be equal geometries.)

    Parameters
    ----------
    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float{32,64}

    Returns
    -------
    metadata : OrderedDict
        Contains the item:
            'hash' : length-8 str
                Hex characters convert to a string of length 8

    """
    assert len(geom.shape) == 3
    assert geom.shape[2] == 3
    rounded_ints = np.round(geom * 100).astype(np.int)
    geom_hash = hash_obj(rounded_ints, fmt='hex')[:8]
    return OrderedDict([('hash', geom_hash)])


def generate_unique_ids(events):
    """Generate unique IDs from `event` fields because people are lazy
    inconsiderate assholes.

    Parameters
    ----------
    events : array of int

    Returns
    -------
    uids : array of int

    """
    uids = (
        events
        + 1e7 * np.cumsum(np.concatenate(([0], np.diff(events) < 0)))
    ).astype(int)
    return uids


def get_primary_interaction_str(event):
    """Produce simple string representation of event's primary neutrino and
    interaction type (if present).

    Parameters
    ----------
    event : Event namedtuple

    Returns
    -------
    flavintstr : string

    """
    pdg = int(event.neutrino.pdg)
    inter = int(event.interaction)
    return PDG_INTER_STR[(pdg, inter)]


def get_primary_interaction_tex(event):
    """Produce latex representation of event's primary neutrino and interaction
    type (if present).

    Parameters
    ----------
    event : Event namedtuple

    Returns
    -------
    flavinttex : string

    """
    pdg = int(event.neutrino.pdg)
    inter = int(event.interaction)
    return PDG_INTER_TEX[(pdg, inter)]


def interpret_clsim_table_fname(fname):
    """Get fields from fname and interpret these (e.g. by covnerting into
    appropriate Python types).

    The fields are parsed into the following types / values:
        - hash_val : str
        - string : str (one of {'ic', 'dc'}) or int
        - depth_idx : int
        - seed : str (exactly '*'), int, or list of ints

    Parameters
    ----------
    fname : string

    Returns
    -------
    info : dict

    Raises
    ------
    ValueError
        If ``basename(fname)`` does not match the regex
        ``CLSIM_TABLE_FNAME_RE``

    """
    from pisa.utils.format import hrlist2list

    fname = basename(fname)
    match = CLSIM_TABLE_FNAME_RE.match(fname)
    if not match:
        raise ValueError('File basename "{}" does not match regex {}'
                         .format(fname, CLSIM_TABLE_FNAME_RE.pattern))
    info = match.groupdict()

    try:
        info['string'] = int(info['string'])
    except ValueError:
        assert info['string'] in ['ic', 'dc']

    try:
        info['seed'] = int(info['seed'])
    except ValueError:
        if info['seed'] != '*':
            info['seed'] = hrlist2list(info['seed'])

    info['depth_idx'] = int(info['depth_idx'])

    return info


# -- Binning / geometry functions -- #


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def _linbin_numba(val, start, stop, num):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    out_type

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    width = (stop - start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.uint32)
    bin_num_flat = bin_num.flat
    for i, v in enumerate(val.flat):
        bin_num_flat[i] = (v - start) // width
    return bin_num


def _linbin_numpy(val, start, stop, num):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    width = (stop - start) / num_bins
    bin_num = (val - start) // width
    #if np.isscalar(bin_num):
    #    bin_num = int(bin_num)
    #else:
    #    bin_num =
    return bin_num


linbin = _linbin_numba # pylint: disable=invalid-name


def test_linbin():
    """Unit tests for function `linbin`."""
    kw = dict(start=0, stop=100, num=200)
    bin_edges = np.linspace(**kw)

    rand = np.random.RandomState(seed=0)
    x = rand.uniform(0, 100, int(1e6))

    test_args = (np.array([0.0]), np.float64(kw['start']),
                 np.float64(kw['stop']), np.uint32(kw['num']))
    _linbin_numba(*test_args)

    test_args = (x, np.float64(kw['start']), np.float64(kw['stop']),
                 np.uint32(kw['num']))
    t0 = time()
    bins_ref = np.digitize(x, bin_edges) - 1
    t1 = time()
    bins_test_numba = _linbin_numba(*test_args)
    t2 = time()
    bins_test_numpy = _linbin_numpy(*test_args)
    t3 = time()

    print('np.digitize:   {} s'.format(t1 - t0))
    print('_linbin_numba: {} s'.format(t2 - t1))
    print('_linbin_numpy: {} s'.format(t3 - t2))

    assert np.all(bins_test_numba == bins_ref)
    assert np.all(bins_test_numpy == bins_ref)
    #assert isinstance(_linbin_numpy(1, **kw), int)

    print('<< PASS : test_linbin >>')


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def _powerbin_numba(val, start, stop, num, power): #, out_type=np.uint64):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float
    out_type

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    inv_power = 1.0 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.uint32)
    bin_num_flat = bin_num.flat
    for i, v in enumerate(val.flat):
        bin_num_flat[i] = (v**inv_power - inv_power_start) // inv_power_width
    return bin_num


def _powerbin_numpy(val, start, stop, num, power):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : scalar or array
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float

    Returns
    -------
    bin_num : int or np.ndarray of dtype int
        If `val` is scalar, returns int; if `val` is a sequence or array-like,
        returns `np.darray` of dtype int.

    """
    num_bins = num - 1
    inv_power = 1 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = (val**inv_power - inv_power_start) // inv_power_width
    if np.isscalar(bin_num):
        bin_num = int(bin_num)
    else:
        bin_num = bin_num.astype(int)
    return bin_num


powerbin = _powerbin_numpy # pylint: disable=invalid-name


#def powerbin(val, start, stop, num, power):
#    """Determine the bin number(s) in a powerspace binning of value(s).
#
#    Parameters
#    ----------
#    val : scalar or array
#    start : float
#    stop : float
#    num : int
#        Number of bin _edges_ (number of bins is one less than `num`)
#    power : float
#
#    Returns
#    -------
#    bin_num : int or np.ndarray of dtype int
#        If `val` is scalar, returns int; if `val` is a sequence or array-like,
#        returns `np.darray` of dtype int.
#
#    """
#    if np.isscalar(val):
#        val = np.array(val)
#    else:
#        val = np.asarray(val)
#
#    if num < 1000:
#        pass


def test_powerbin():
    """Unit tests for function `powerbin`."""
    kw = dict(start=0, stop=100, num=100, power=2)
    bin_edges = powerspace(**kw)

    rand = np.random.RandomState(seed=0)
    x = rand.uniform(0, 100, int(1e6))

    ftype = np.float32
    utype = np.uint32
    test_args = (ftype(kw['start']), ftype(kw['stop']),
                 utype(kw['num']), utype(kw['power']))

    # Run once to force compilation
    _powerbin_numba(np.array([0.0], dtype=ftype), *test_args)

    # Run actual tests / take timings
    t0 = time()
    bins_ref = np.digitize(x, bin_edges) - 1
    t1 = time()
    bins_test_numba = _powerbin_numba(x, *test_args)
    t2 = time()
    bins_test_numpy = _powerbin_numpy(x, *test_args)
    t3 = time()

    print('np.digitize:     {:e} s'.format(t1 - t0))
    print('_powerbin_numba: {:e} s'.format(t2 - t1))
    print('_powerbin_numpy: {:e} s'.format(t3 - t2))

    assert np.all(bins_test_numba == bins_ref), str(bins_test_numba) + '\n' + str(bins_ref)
    assert np.all(bins_test_numpy == bins_ref), str(bins_test_numpy) + '\n' + str(bins_ref)
    #assert isinstance(_powerbin_numpy(1, **kw), int)

    print('<< PASS : test_powerbin >>')


# TODO: add `endpoint`, `retstep`, and `dtype` kwargs
def powerspace(start, stop, num, power):
    """Create bin edges evenly spaced w.r.t. ``x**power``.

    Reverse engineered from Jakob van Santen's power axis, with arguments
    defined with analogy to :function:`numpy.linspace` (adding `power` but
    removing the `endpoint`, `retstep`, and `dtype` arguments).

    Parameters
    ----------
    start : float
        Lower-most bin edge

    stop : float
        Upper-most bin edge

    num : int
        Number of edges (this defines ``num - 1`` bins)

    power : float
        Power-law to use for even spacing

    Returns
    -------
    edges : numpy.ndarray of shape (1, num)
        Edges

    """
    inv_power = 1 / power
    liner_edges = np.linspace(start=np.power(start, inv_power),
                              stop=np.power(stop, inv_power),
                              num=num)
    bin_edges = np.power(liner_edges, power)
    return bin_edges


def inv_power_2nd_diff(power, edges):
    """Second finite difference of edges**(1/power)"""
    return np.diff(edges**(1/power), n=2)


def infer_power(edges):
    """Infer the power used for bin edges evenly spaced w.r.t. ``x**power``."""
    first_three_edges = edges[:3]
    atol = 1e-15
    rtol = 4*np.finfo(np.float).eps
    power = None
    try:
        power = brentq(
            f=inv_power_2nd_diff,
            a=1, b=100,
            maxiter=1000, xtol=atol, rtol=rtol,
            args=(first_three_edges,)
        )
    except RuntimeError:
        raise ValueError('Edges do not appear to be power-spaced'
                         ' (optimizer did not converge)')
    diff = inv_power_2nd_diff(power, edges)
    if not np.allclose(diff, diff[0], atol=1000*atol, rtol=10*rtol):
        raise ValueError('Edges do not appear to be power-spaced'
                         ' (power found does not hold for all edges)\n%s'
                         % str(diff))
    return power


def test_infer_power():
    """Unit test for function `infer_power`"""
    ref_powers = np.arange(1, 10, 0.001)
    total_time = 0.0
    for ref_power in ref_powers:
        edges = powerspace(start=0, stop=400, num=201, power=ref_power)
        try:
            t0 = time()
            inferred_power = infer_power(edges)
            t1 = time()
        except ValueError:
            print(ref_power, edges)
            raise
        assert np.isclose(inferred_power, ref_power,
                          atol=1e-14, rtol=4*np.finfo(np.float).eps), ref_power
        total_time += t1 - t0
    print('Average time to infer power: {} s'
          .format(total_time/len(ref_powers)))
    print('<< PASS : test_infer_power >>')


def bin_edges_to_binspec(edges):
    """Convert bin edges to a binning specification (start, stop, and num_bins).

    Note:
    * t-bins are assumed to be linearly spaced in ``t``
    * r-bins are assumed to be evenly spaced w.r.t. ``r**2``
    * theta-bins are assumed to be evenly spaced w.r.t. ``cos(theta)``
    * phi bins are assumed to be linearly spaced in ``phi``

    Parameters
    ----------
    edges

    Returns
    -------
    start : TimeSphCoord containing floats
    stop : TimeSphCoord containing floats
    num_bins : TimeSphCoord containing ints

    """
    dims = TimeSphCoord._fields
    start = TimeSphCoord(*(np.min(getattr(edges, d)) for d in dims))
    stop = TimeSphCoord(*(np.max(getattr(edges, d)) for d in dims))
    num_bins = TimeSphCoord(*(len(getattr(edges, d)) - 1 for d in dims))

    return start, stop, num_bins


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def linear_bin_centers(bin_edges):
    """Return bin centers for bins defined in a linear space.

    Parameters
    ----------
    bin_edges : sequence of numeric
        Note that all numbers contained must be of the same dtype (this is a
        limitation due to numba, at least as of version 0.35).

    Returns
    -------
    bin_centers : numpy.ndarray
        Length is one less than that of `bin_edges`, and datatype is inferred
        from the first element of `bin_edges`.

    """
    num_edges = len(bin_edges)
    bin_centers = np.empty(num_edges - 1, bin_edges.dtype)
    edge0 = bin_edges[0]
    for n in range(num_edges - 1):
        edge1 = bin_edges[n + 1]
        bin_centers[n] = 0.5 * (edge0 + edge1)
        edge0 = edge1
    return bin_centers


def spacetime_separation(dt, dx, dy, dz):
    """Compute the separation between two events in spacetime. Negative values
    are non-causal.

    Parameters
    ----------
    dt, dx, dy, dz : numeric
        Separation between events in nanoseconds (dt) and meters (dx, dy, and
        dz).

    """
    return SPEED_OF_LIGHT_M_PER_NS*dt - np.sqrt(dx**2 + dy**2 + dz**2)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def spherical_volume(rmin, rmax, dcostheta, dphi):
    """Find volume of a finite element defined in spherical coordinates.

    Parameters
    ----------
    rmin, rmax : float (in arbitrary distance units)
        Difference between initial and final radii.

    dcostheta : float
        Difference between initial and final zenith angles' cosines (where
        zenith angle is defined as out & down from +Z axis).

    dphi : float (in units of radians)
        Difference between initial and final azimuth angle (defined as positive
        from +X-axis towards +Y-axis looking "down" on the XY-plane (i.e.,
        looking in -Z direction).

    Returns
    -------
    vol : float
        Volume in units of the cube of the units that ``dr`` is provided in.
        E.g. if those are provided in meters, ``vol`` will be in units of `m^3`.

    """
    return -dcostheta * (rmax**3 - rmin**3) * dphi / 3


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def sph2cart(r, theta, phi, x, y, z):
    """Convert spherical coordinates to Cartesian.

    Parameters
    ----------
    r, theta, phi : numeric of same shape

    x, y, z : numpy.ndarrays of same shape as `r`, `theta`, `phi`
        Result is stored in these arrays.

    """
    shape = r.shape
    num_elements = int(np.prod(np.array(shape)))
    r_flat = r.flat
    theta_flat = theta.flat
    phi_flat = phi.flat
    x_flat = x.flat
    y_flat = y.flat
    z_flat = z.flat
    for idx in range(num_elements):
        rf = r_flat[idx]
        thetaf = theta_flat[idx]
        phif = phi_flat[idx]
        rho = rf * math.sin(thetaf)
        x_flat[idx] = rho * math.cos(phif)
        y_flat[idx] = rho * math.sin(phif)
        z_flat[idx] = rf * math.cos(thetaf)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pol2cart(r, theta, x, y):
    """Convert plane polar (r, theta) to Cartesian (x, y) Coordinates.

    Parameters
    ----------
    r, theta : numeric of same shape

    x, y : numpy.ndarrays of same shape as `r`, `theta`
        Result is stored in these arrays.

    """
    shape = r.shape
    num_elements = int(np.prod(np.array(shape)))
    r_flat = r.flat
    theta_flat = theta.flat
    x_flat = x.flat
    y_flat = y.flat
    for idx in range(num_elements):
        rf = r_flat[idx]
        tf = theta_flat[idx]
        x_flat[idx] = rf * math.cos(tf)
        y_flat[idx] = rf * math.sin(tf)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def cart2pol(x, y, r, theta):
    """Convert plane Cartesian (x, y) to plane polar (r, theta) Coordinates.

    Parameters
    ----------
    x, y : numeric of same shape

    r, theta : numpy.ndarrays of same shape as `x`, `y`
        Result is stored in these arrays.

    """
    shape = x.shape
    num_elements = int(np.prod(np.array(shape)))
    x_flat = x.flat
    y_flat = y.flat
    r_flat = r.flat
    theta_flat = theta.flat
    for idx in range(num_elements):
        xfi = x_flat[idx]
        yfi = y_flat[idx]
        r_flat[idx] = math.sqrt(xfi*xfi + yfi*yfi)
        theta_flat[idx] = math.atan2(yfi, xfi)


#@numba_jit(nopython=True, nogil=True, cache=True, parallel=True)
#def sph2cart(r, theta, phi):
#    """Convert spherical coordinates to Cartesian.
#
#    Parameters
#    ----------
#    r, theta, phi
#        Spherical coordinates: radius, theta (zenith angle, defined as "out"
#        from the +z-axis), and phi (azimuth angle, defined as positive from
#        +x-axis to +y-axix)
#
#    Returns
#    -------
#    x, y, z
#        Cartesian coordinates
#
#    """
#    z = r * np.cos(theta)
#    rsintheta = r * np.sin(theta)
#    x = rsintheta * np.cos(phi)
#    y = rsintheta * np.sin(phi)
#    return x, y, z


# -- Other math functions -- #


def poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - {\rm expected} \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    # TODO: why is there a +1 here? Avoid zero observations? How does this
    # affect the result, besides avoiding inf? Removed for now until we work
    # this out...

    #llh = observed * np.log(expected) - expected - gammaln(observed + 1)
    llh = observed * np.log(expected) - expected - gammaln(observed)
    return llh


def partial_poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood _excluding_ subtracting off
    expected. This part, which constitutes an expected-but-not-observed
    penalty, is intended to be taken care of outside this function.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    # TODO: why is there a +1 here? Avoid zero observations? How does this
    # affect the result, besides avoiding inf? Removed for now until we work
    # this out...
    llh = observed * np.log(expected) - expected - gammaln(observed)
    return llh


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def weighted_average(x, w):
    """Average of elements in `x` weighted by `w`.

    Parameters
    ----------
    x : numpy.ndarray
        Values to average

    w : numpy.ndarray
        Weights, same shape as `x`

    Returns
    -------
    avg : numpy.ndarray
        Weighted average, same shape as `x`

    """
    sum_xw = 0.0
    sum_w = 0.0
    for x_i, w_i in zip(x, w):
        sum_xw += x_i * w_i
        sum_w += w_i
    return sum_xw / sum_w


if __name__ == '__main__':
    test_infer_power()
    test_hash_obj()
    test_linbin()
    test_powerbin()
