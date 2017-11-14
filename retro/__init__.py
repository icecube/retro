"""
Basic module-wide definitions and simple types (namedtuples).
"""


from __future__ import absolute_import, division, print_function

from collections import namedtuple, OrderedDict, Iterable, Mapping, Sequence
import math
from os.path import abspath, dirname, expanduser, expandvars, join
import re

import numba
import numpy as np
import pyfits
from scipy.special import gammaln

from pisa.utils.hash import hash_obj


__all__ = [
    # Defaults
    'DFLT_NUMBA_JIT_KWARGS', 'DFLT_PULSE_SERIES', 'DFLT_ML_RECO_NAME',
    'DFLT_SPE_RECO_NAME', 'IC_RAW_TABLE_FNAME_PROTO',
    'DC_RAW_TABLE_FNAME_PROTO', 'IC_TABLE_FNAME_PROTO', 'DC_TABLE_FNAME_PROTO',
    'DETECTOR_GEOM_FILE', 'TDI_TABLE_FNAME_PROTO', 'TDI_TABLE_FNAME_RE',

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

    # Functions
    'convert_to_namedtuple', 'expand', 'event_to_hypo_params',
    'hypo_to_track_params', 'powerspace', 'bin_edges_to_binspec',
    'linear_bin_centers', 'poisson_llh', 'spacetime_separation',
    'generate_anisotropy_str', 'generate_geom_meta',
    'generate_unique_ids', 'spherical_volume', 'sph2cart',
    'get_primary_interaction_str', 'get_primary_interaction_tex',
    'force_little_endian',
]


# -- Default choices we've made -- #

DFLT_NUMBA_JIT_KWARGS = dict(nopython=True, nogil=True, cache=True)
"""kwargs to pass to numba.jit"""

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

IC_RAW_TABLE_FNAME_PROTO = 'retro_nevts1000_IC_DOM{depth_idx:d}.fits'
"""String template for IceCube single-DOM raw retro tables"""

DC_RAW_TABLE_FNAME_PROTO = 'retro_nevts1000_DC_DOM{depth_idx:d}.fits'
"""String template for DeepCore single-DOM raw retro tables"""

IC_TABLE_FNAME_PROTO = 'retro_nevts1000_IC_DOM{depth_idx:d}_r_cz_t_angles.fits'
"""String template for IceCube single-DOM final-level retro tables"""

DC_TABLE_FNAME_PROTO = 'retro_nevts1000_DC_DOM{depth_idx:d}_r_cz_t_angles.fits'
"""String template for DeepCore single-DOM final-level retro tables"""

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
"""Hypothesis in 8 dimensions (parameters)"""

HypoParams10D = namedtuple( # pylint: disable=invalid-name
    typename='HypoParams10D',
    field_names=(HypoParams8D._fields + ('cascade_zenith', 'cascade_azimuth'))
)
"""Hypothesis in 10 dimensions (parameters)"""

TrackParams = namedtuple( # pylint: disable=invalid-name
    typename='TrackParams',
    field_names=('t', 'x', 'y', 'z', 'zenith', 'azimuth', 'energy')
)
"""Hypothesis for just the track (7 dimensions / parameters)"""

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


# -- Functions -- #

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

# TODO: add `endpoint`, `retstep`, and `dtype` kwargs
def powerspace(start, stop, num, power):
    """Create bin edges evenly spaced w.r.t. ``x**power``.

    Reverse engineered from JVS's power axis, with arguments defined with
    analogy to :function:`numpy.linspace`.

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


@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
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
    geom_hash = hash_obj(rounded_ints, hash_to='hex', full_hash=True)
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


@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
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


@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
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


@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
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


@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
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


#@numba.jit(nopython=True, nogil=True, cache=True, parallel=True)
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
    barnobar = {-1: r'bar', 1: ''}[np.sign(pdg)]
    flav = {12: r'nue', 14: r'numu', 16: r'nutau'}[abs(pdg)]
    int_type = {None: '', 1: r'_cc', 2: r'_nc'}[int(event.interaction)]
    return flav + barnobar + int_type


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
    if isinstance(event, (tuple, Event)):
        prim_int_str = get_primary_interaction_str(event)
    elif isinstance(event, basestring):
        prim_int_str = event
    else:
        raise TypeError('Unhandled type %s for argument `event`' % type(event))

    if prim_int_str.startswith('nue'):
        tex = r'\nu_e'
    elif prim_int_str.startswith('numu'):
        tex = r'\nu_\mu'
    elif prim_int_str.startswith('nutau'):
        tex = r'\nu_\tau'

    if 'bar' in prim_int_str:
        tex = r'\bar' + tex

    if prim_int_str.endswith('_cc'):
        tex += r'\,CC'
    elif prim_int_str.endswith('_nc'):
        tex += r'\,NC'

    return tex


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
