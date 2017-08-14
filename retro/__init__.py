"""
Basic module-wide definitions and simple types (namedtuples).
"""


from __future__ import absolute_import, division, print_function

from collections import namedtuple
from os.path import abspath, expanduser, expandvars

import h5py
import numpy as np

from particles import ParticleArray


__all__ = ['FTYPE', 'UITYPE', 'DFLT_PULSE_SERIES', 'DFLT_ML_RECO_NAME',
           'DFLT_SPE_RECO_NAME', 'SPEED_OF_LIGHT_M_PER_NS', 'TWO_PI',
           'PI_BY_TWO', 'HypoParams8D', 'HypoParams10D', 'HYPO_PARAMS_T',
           'TrackParams', 'Event', 'Pulses', 'PhotonInfo', 'BinningCoords',
           'TimeSpaceCoord', 'expand', 'event_to_hypo_params',
           'hypo_to_track_params', 'power_axis', 'Events']


# -- Datatypes to use -- #

FTYPE = np.float32
"""Datatype to use for explicitly-typed floating point numbers"""

UITYPE = np.uint16
"""Datatype to use for explicitly-typed unsigned integers"""

# -- Default choices we've made -- #

DFLT_PULSE_SERIES = 'SRTInIcePulses'
"""Default pulse series to extract from events"""

DFLT_ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
"""Default maximum-likelihood reco to extract for an event"""

DFLT_SPE_RECO_NAME = 'SPEFit2'
"""Default single photoelectron (SPE) reco to extract for an event"""

# -- Physical / mathematical constants -- #

TWO_PI = FTYPE(2*np.pi)
"""2 * pi"""

PI_BY_TWO = FTYPE(np.pi / 2)
"""pi / 2"""

SPEED_OF_LIGHT_M_PER_NS = FTYPE(0.299792458)
"""Speed of light in units of m/ns"""

# -- Precalculated (using ``nphotons.py``) to avoid icetray -- #

TRACK_M_PER_GEV = 15 / 3.3
"""Track length per energy, in units of m/GeV"""

TRACK_PHOTONS_PER_M = 2451.4544553
"""Track photons per length, in units of 1/m"""

CASCADE_PHOTONS_PER_GEV = 12805.3383311
"""Cascade photons per energy, in units of 1/GeV"""

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

HYPO_PARAMS_T = HypoParams8D
"""Global selection of which hypothesis to use throughout the code."""

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


class Events(object):
    """Container for events extracted from an HDF5 file.

    Parameters
    ----------
    events_fpath : string
        Path to HDF5 file

    """
    def __init__(self, events_fpath):
        self.events = []
        self._num_events = 0
        self.pulses = None
        self.pulse_event_boundaries = None
        self.int_type = None
        self.load(events_fpath)

    def load(self, events_fpath, pulse_series=DFLT_PULSE_SERIES,
             spe_reco_name=DFLT_SPE_RECO_NAME, ml_reco_name=DFLT_ML_RECO_NAME):
        """Load events from file, populating `self`.

        Parameters
        ----------
        events_fpath : string
            Path to HDF5 file

        """
        print('loading events from path "%s"' % events_fpath)
        with h5py.File(expand(events_fpath)) as h5:
            pulses = h5[pulse_series]
            pulse_events = pulses['Event']
            strings = pulses['string']
            print('string range: [%d, %d]' % (np.min(strings), np.max(strings)))
            oms = pulses['om']
            print('OM range: [%d, %d]' % (np.min(oms), np.max(oms)))
            # NOTE: string and om indices are one less than their numbers
            self.pulses = Pulses(
                strings=strings - 1,
                oms=oms - 1,
                times=pulses['time'],
                charges=pulses['charge'],
            )

            # Calculate the first index into the pulses array for each unique
            # event
            self.pulse_event_boundaries = [0]
            self.pulse_event_boundaries.extend(
                np.where(np.diff(pulse_events))[0]
            )

            self.interactions = h5['I3MCWeightDict']['InteractionType']

            nu = h5['trueNeutrino']
            self.neutrinos = ParticleArray(
                evt=nu['Event'],
                t=nu['time'],
                x=nu['x'],
                y=nu['y'],
                z=nu['z'],
                zenith=nu['zenith'],
                azimuth=nu['azimuth'],
                energy=nu['energy'],
                length=None,
                pdg=nu['type'],
                color='r',
                linestyle=':',
                label='Neutrino'
            )
            mu = h5['trueMuon']
            self.tracks = ParticleArray(
                evt=mu['Event'],
                t=mu['time'],
                x=mu['x'],
                y=mu['y'],
                z=mu['z'],
                zenith=mu['zenith'],
                azimuth=mu['azimuth'],
                energy=mu['energy'],
                length=mu['length'],
                forward=True,
                color='b',
                linestyle='-',
                label='track'
            )
            cascade = h5['trueCascade']
            self.cascades = ParticleArray(
                evt=cascade['Event'],
                t=cascade['time'],
                x=cascade['x'],
                y=cascade['y'],
                z=cascade['z'],
                zenith=cascade['zenith'],
                azimuth=cascade['azimuth'],
                energy=cascade['energy'],
                length=None,
                color='y',
                label='cascade'
            )
            reco = h5[ml_reco_name]
            self.ml_recos = ParticleArray(
                evt=reco['Event'],
                t=reco['time'],
                x=reco['x'],
                y=reco['y'],
                z=reco['z'],
                zenith=reco['zenith'],
                azimuth=reco['azimuth'],
                energy=reco['energy'],
                length=None,
                color='g',
                label='Multinest'
            )
            reco = h5[spe_reco_name]
            self.spe_recos = ParticleArray(
                evt=reco['Event'],
                t=reco['time'],
                x=reco['x'],
                y=reco['y'],
                z=reco['z'],
                zenith=reco['zenith'],
                azimuth=reco['azimuth'],
                color='m',
                label='SPE'
            )
        self.events = self.neutrinos.evt
        self._num_events = len(self.events)

    def __len__(self):
        return self._num_events

    def __iter__(self):
        for idx in xrange(self._num_events):
            yield self[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Convert slice into (start, stop, step) tuple
            range_args = idx.indices(len(self))
            return [self[i] for i in xrange(*range_args)]

        neutrino = self.neutrinos[idx]
        event = neutrino.evt
        pulse_start_idx = self.pulse_event_boundaries[idx]
        if idx < self._num_events - 1:
            pulse_stop_idx = self.pulse_event_boundaries[idx + 1]
        else:
            pulse_stop_idx = None
        slc = slice(pulse_start_idx, pulse_stop_idx)
        event = Event(
            event=event,
            pulses=Pulses(
                strings=self.pulses.strings[slc],
                oms=self.pulses.oms[slc],
                times=self.pulses.times[slc],
                charges=self.pulses.charges[slc]
            ),
            interaction=self.interactions[idx],
            neutrino=self.neutrinos[idx],
            track=self.tracks[idx],
            cascade=self.cascades[idx],
            ml_reco=self.ml_recos[idx],
            spe_reco=self.spe_recos[idx]
        )
        return event
