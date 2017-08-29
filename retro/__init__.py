"""
Basic module-wide definitions and simple types (namedtuples).
"""


from __future__ import absolute_import, division, print_function

from collections import namedtuple, Iterable, Mapping, Sequence
import math
from os.path import abspath, dirname, expanduser, expandvars, join

import numba
import numpy as np
import pyfits
from scipy.special import gammaln


__all__ = [
    # Defaults
    'DFLT_PULSE_SERIES', 'DFLT_ML_RECO_NAME', 'DFLT_SPE_RECO_NAME',
    'IC_TABLE_FPATH_PROTO', 'DC_TABLE_FPATH_PROTO', 'DETECTOR_GEOM_FILE',

    # Type definitions
    'HypoParams8D', 'HypoParams10D', 'TrackParams', 'Event', 'Pulses',
    'RetroPhotonInfo', 'HypoPhotonInfo', 'TimeSphCoord', 'TimeCartCoord',

    # Type selections
    'FTYPE', 'UITYPE', 'HYPO_PARAMS_T',

    # Constants
    'SPEED_OF_LIGHT_M_PER_NS', 'PI', 'TWO_PI', 'PI_BY_TWO',

    # Pre-calculated values
    'TRACK_M_PER_GEV', 'TRACK_PHOTONS_PER_M', 'CASCADE_PHOTONS_PER_GEV',
    'IC_DOM_JITTER_NS', 'DC_DOM_JITTER_NS',

    # Functions
    'convert_to_namedtuple', 'expand', 'event_to_hypo_params',
    'hypo_to_track_params', 'powerspace', 'binspec_to_edges',
    'bin_edges_to_binspec', 'bin_edges_to_centers', 'poisson_llh',
    'spacetime_separation', 'generate_unique_ids', 'extract_photon_info',
    'spherical_volume', 'sph2cart', 'get_primary_interaction_str',
    'get_primary_interaction_tex'
]


# -- Default choices we've made -- #

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

IC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_IC_DOM{dom:d}_r_cz_t_angles.fits'
)
"""String template for IceCube single-DOM final-level retro tables"""

DC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_DC_DOM{dom:d}_r_cz_t_angles.fits'
)
"""String template for DeepCore single-DOM final-level retro tables"""

DETECTOR_GEOM_FILE = join(dirname(abspath(__file__)), 'data', 'geo_array.npy')
"""Numpy .npy file containing detector geometry (DOM x, y, z coordinates)"""

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

TimeSphCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeSphCoord',
    field_names=('t', 'r', 'theta', 'phi')
)
"""Binning coordinates."""

TimeCartCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeCartCoord',
    field_names=('t', 'x', 'y', 'z'))
"""Time and space coordinates: t, x, y, z."""


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

SPEED_OF_LIGHT_M_PER_NS = FTYPE(0.299792458)
"""Speed of light in units of m/ns"""


# -- Pre-calculated values -- #

TRACK_M_PER_GEV = FTYPE(15 / 3.3)
"""Track length per energy, in units of m/GeV"""

TRACK_PHOTONS_PER_M = FTYPE(2451.4544553 * 10)
"""Track photons per length, in units of 1/m (see ``nphotons.py``)"""

CASCADE_PHOTONS_PER_GEV = FTYPE(12805.3383311 * 10 * 0.6)
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
    liner_edges = np.linspace(np.power(start, inv_power),
                              np.power(stop, inv_power),
                              num)
    bin_edges = np.power(liner_edges, power)
    return bin_edges


def binspec_to_edges(start, stop, num_bins):
    """Convert binning specification (start, stop, and num_bins) to bin edges.

    Note:
    * t-bins are linearly spaced in ``t``
    * r-bins are evenly spaced w.r.t. ``r**2``
    * theta-bins are evenly spaced w.r.t. ``cos(theta)``
    * phi bins are linearly spaced in ``phi``

    Parameters
    ----------
    start : TimeSphCoord containing floats
    stop : TimeSphCoord containing floats
    num_bins : TimeSphCoord containing ints

    Returns
    -------
    edges : TimeSphCoord containing arrays of floats

    """
    if not isinstance(start, TimeSphCoord):
        start = TimeSphCoord(*start)
    if not isinstance(stop, TimeSphCoord):
        stop = TimeSphCoord(*stop)
    if not isinstance(num_bins, TimeSphCoord):
        num_bins = TimeSphCoord(*num_bins)

    edges = TimeSphCoord(
        t=np.linspace(start.t, stop.t, num_bins.t + 1),
        r=powerspace(start=start.r, stop=stop.r, num=num_bins.r + 1, power=2),
        theta=np.arccos(np.linspace(np.cos(start.theta),
                                    np.cos(stop.theta),
                                    num_bins.theta + 1)),
        phi=np.linspace(start.phi, stop.phi, num_bins.phi + 1)
    )

    return edges


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


def bin_edges_to_centers(bin_edges):
    """Return bin centers, where center is defined in whatever space the
    dimension is "regular."

    E.g., r is binned regularly in r**2-space, so centers are computed as
    averages in r**2-space but returned in r-space.

    Parameters
    ----------
    bin_edges : TimeSphCoord namedtuple or convertible thereto

    Returns
    -------
    bin_centers : TimeSphCoord

    """
    t = bin_edges.t
    rsqaured = np.square(bin_edges.r)
    costheta = np.cos(bin_edges.theta)
    phi = bin_edges.phi
    bin_centers = TimeSphCoord(
        t=0.5 * (t[:-1] + t[1:]),
        r=np.sqrt(0.5 * (rsqaured[:-1] + rsqaured[1:])),
        theta=np.arccos(0.5 * (costheta[:-1] + costheta[1:])),
        phi=0.5 * (phi[:-1] + phi[1:]),
    )
    return bin_centers


def poisson_llh(expected, observed):
    """Compute Poisson log-likelihood and center around zero.

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
    #llh = observed * np.log(expected) - expected - gammaln(observed + 1)
    llh = observed * np.log(expected) - gammaln(observed)
    return llh


def spacetime_separation(dt, dx, dy, dz):
    """Compute the separation between two events in spacetime. Negative values
    are non-causal.

    Parameters
    ----------
    dt, dx, dy, dz : numeric
        Separation between events in ns (dt) and meters (dx, dy, and dz).

    """
    return SPEED_OF_LIGHT_M_PER_NS*dt - np.sqrt(dx**2 + dy**2 + dz**2)


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


def extract_photon_info(fpath, dom_depth_index, scale=1, photon_info=None):
    """Extract photon info from a FITS file containing a (t, r, theta)-binned
    table.

    Parameters
    ----------
    fpath : string
        Path to FITS file corresponding to the passed ``dom_depth_index``.

    dom_depth_index : int
        Depth index (e.g. from 0 to 59)

    scale : float
        Scaling factor to apply to the photon survival probability from the
        table, e.g. for DOM efficiency.

    photon_info : None or RetroPhotonInfo namedtuple of dicts
        If None, creates a new RetroPhotonInfo namedtuple with empty dicts to
        fill. If one is provided, the existing component dictionaries are
        updated.

    Returns
    -------
    photon_info : RetroPhotonInfo namedtuple of dicts
        Tuple fields are 'survival_prob', 'theta', 'phi', and 'length'. Each
        dict is keyed by `dom_depth_index` and values are the arrays loaded
        from the FITS file.

    bin_edges : TimeSphCoord namedtuple
        Each element of the tuple is an array of bin edges.

    """
    # pylint: disable=no-member
    assert 0 < scale <= 1

    if photon_info is None:
        empty_dicts = []
        for _ in RetroPhotonInfo._fields:
            empty_dicts.append({})
        photon_info = RetroPhotonInfo(*empty_dicts)

    with pyfits.open(expand(fpath)) as table:
        if scale == 1:
            photon_info.survival_prob[dom_depth_index] = table[0].data
        else:
            photon_info.survival_prob[dom_depth_index] = table[0].data * scale

        photon_info.theta[dom_depth_index] = table[1].data
        photon_info.deltaphi[dom_depth_index] = table[2].data
        photon_info.length[dom_depth_index] = table[3].data

        # Note that we invert (reverse and multiply by -1) time edges; also,
        # no phi edges are defined in these tables.
        bin_edges = TimeSphCoord(
            t=-table[4].data[::-1],
            r=table[5].data,
            theta=table[6].data,
            phi=np.array([])
        )

    return photon_info, bin_edges


@numba.jit(nopython=True, nogil=True, cache=True)
def spherical_volume(dr, dcostheta, dphi):
    """Find volume of a finite element defined in spherical coordinates.

    Parameters
    ----------
    dr : float (in arbitrary distance units)
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
    return dcostheta * dr**3 * dphi / 3


@numba.jit(nopython=True, nogil=True, cache=True)
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
        tf = theta_flat[idx]
        pf = phi_flat[idx]
        rsintheta = rf * math.sin(tf)
        x_flat[idx] = rsintheta * math.cos(pf)
        y_flat[idx] = rsintheta * math.sin(pf)
        z_flat[idx] = rf * math.cos(tf)


@numba.jit(nopython=True, nogil=True, cache=True)
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


@numba.jit(nopython=True, nogil=True, cache=True)
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
