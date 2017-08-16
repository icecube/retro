"""
Basic module-wide definitions and simple types (namedtuples).
"""


from __future__ import absolute_import, division, print_function

from collections import namedtuple, Iterable, Mapping, Sequence
from os.path import abspath, expanduser, expandvars

import numpy as np


__all__ = [
    # Defaults
    'DFLT_PULSE_SERIES', 'DFLT_ML_RECO_NAME', 'DFLT_SPE_RECO_NAME',

    # Type definitions
    'HypoParams8D', 'HypoParams10D', 'TrackParams', 'Event', 'Pulses',
    'PhotonInfo', 'BinningCoords', 'TimeSpaceCoord',

    # Type selections
    'FTYPE', 'UITYPE', 'HYPO_PARAMS_T',

    # Constants
    'SPEED_OF_LIGHT_M_PER_NS', 'TWO_PI', 'PI_BY_TWO',

    # Pre-calculated values
    'TRACK_M_PER_GEV', 'TRACK_PHOTONS_PER_M', 'CASCADE_PHOTONS_PER_GEV',

    # Functions
    'convert_to_namedtuple', 'expand', 'event_to_hypo_params',
    'hypo_to_track_params', 'power_axis', 'binspec_to_edges',
    'bin_edges_to_centers',
]


# -- Default choices we've made -- #

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""


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
    field_names=('event', 'pulses', 'interaction', 'neutrino', 'track',
                 'cascade', 'ml_reco', 'spe_reco')
)

Pulses = namedtuple( # pylint: disable=invalid-name
    typename='Pulses',
    field_names=('strings', 'oms', 'times', 'charges')
)

PhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='PhotonInfo',
    field_names=('count', 'theta', 'phi', 'length')
)
"""Intended to contain dictionaries with DOM depth number as keys"""

BinningCoords = namedtuple( # pylint: disable=invalid-name
    typename='BinningCoords',
    field_names=('t', 'r', 'theta', 'phi')
)
"""Binning coordinates."""

TimeSpaceCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeSpaceCoord',
    field_names=('t', 'x', 'y', 'z'))
"""Time and space coordinates: t, x, y, z."""


# -- Datatype choices for consistency throughout code -- #

FTYPE = np.float64
"""Datatype to use for explicitly-typed floating point numbers"""
print(FTYPE)

UITYPE = np.int64
"""Datatype to use for explicitly-typed unsigned integers"""

HYPO_PARAMS_T = HypoParams8D
"""Global selection of which hypothesis to use throughout the code."""


# -- Physical / mathematical constants -- #

TWO_PI = FTYPE(2*np.pi)
"""2 * pi"""

PI_BY_TWO = FTYPE(np.pi / 2)
"""pi / 2"""

SPEED_OF_LIGHT_M_PER_NS = FTYPE(0.299792458)
"""Speed of light in units of m/ns"""


# -- Precalculated (using ``nphotons.py``) to avoid icetray -- #

TRACK_M_PER_GEV = FTYPE(15 / 3.3)
"""Track length per energy, in units of m/GeV"""

TRACK_PHOTONS_PER_M = FTYPE(2451.4544553)
"""Track photons per length, in units of 1/m"""

CASCADE_PHOTONS_PER_GEV = FTYPE(12805.3383311)
"""Cascade photons per energy, in units of 1/GeV"""


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


def power_axis(start, stop, num_bins, power):
    """Create bin edges evenly spaced w.r.t. ``x**power``.

    Reverse engineered from JVS's power axis.

    Parameters
    ----------
    start : float
        Lower-most bin edge

    stop : float
        Upper-most bin edge

    num_bins : int
        Number of bins (there are num_bins + 1 edges)

    power : float
        Power-law to use for even spacing

    Returns
    -------
    edges : numpy.ndarray of shape (1, num_bins)
        Bin edges

    """
    inv_power = 1 / power
    liner_edges = np.linspace(np.power(start, inv_power),
                              np.power(stop, inv_power),
                              num_bins + 1)
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
    start : BinningCoords containing floats
    stop : BinningCoords containing floats
    num_bins : BinningCoords containing ints

    Returns
    -------
    edges : BinningCoords containing arrays of floats

    """
    if not isinstance(start, BinningCoords):
        start = BinningCoords(*start)
    if not isinstance(stop, BinningCoords):
        stop = BinningCoords(*stop)
    if not isinstance(num_bins, BinningCoords):
        num_bins = BinningCoords(*num_bins)

    edges = BinningCoords(
        t=np.linspace(start.t, stop.t, num_bins.t + 1),
        r=power_axis(start=start.r, stop=stop.r, num_bins=num_bins.r, power=2),
        theta=np.arccos(np.linspace(np.cos(start.theta),
                                    np.cos(stop.theta),
                                    num_bins.theta + 1)),
        phi=np.linspace(start.phi, stop.phi, num_bins.phi + 1)
    )

    return edges


def bin_edges_to_centers(bin_edges):
    """Return bin centers, where center is defined in whatever space the
    dimension is "regular."

    E.g., r is binned regularly in r**2-space, so centers are computed as
    averages in r**2-space but returned in r-space.

    Parameters
    ----------
    bin_edges : BinningCoords namedtuple or convertible thereto

    Returns
    -------
    bin_centers : BinningCoords

    """
    t = bin_edges.t
    rsqaured = np.square(bin_edges.r)
    costheta = np.cos(bin_edges.theta)
    phi = bin_edges.phi
    bin_centers = BinningCoords(
        t=0.5 * (t[:-1] + t[1:]),
        r=np.sqrt(0.5 * (rsqaured[:-1] + rsqaured[1:])),
        theta=np.arccos(0.5 * (costheta[:-1] + costheta[1:])),
        phi=0.5 * (phi[:-1] + phi[1:]),
    )
    return bin_centers
