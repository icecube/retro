"""
Basic module-wide definitions and simple types (namedtuples).
"""


from collections import namedtuple

import numpy as np


__all__ = ['FTYPE', 'SPEED_OF_LIGHT_M_PER_NS', 'TWO_PI', 'PI_BY_TWO',
           'HypoParams8D', 'HypoParams10D', 'HYPO_PARAMS_T', 'TrackParams',
           'Event', 'Pulses', 'PhotonInfo', 'BinningCoords',
           'event_to_hypo_params', 'hypo_to_track_params', 'power_axis',
           'Events']


FTYPE = np.float32

SPEED_OF_LIGHT_M_PER_NS = FTYPE(0.299792458)
"""Speed of light in units of m/ns"""

TWO_PI = FTYPE(2*np.pi)
PI_BY_TWO = FTYPE(np.pi / 2)

HypoParams8D = namedtuple(typename='HypoParams8D', # pylint: disable=invalid-name
                          field_names=('t', 'x', 'y', 'z', 'track_zenith',
                                       'track_azimuth', 'track_energy',
                                       'cascade_energy'))

HypoParams10D = namedtuple(typename='HypoParams10D', # pylint: disable=invalid-name
                           field_names=('t', 'x', 'y', 'z', 'track_azimuth',
                                        'track_zenith', 'track_energy',
                                        'cascade_azimuth', 'cascade_zenith',
                                        'cascade_energy'))

HYPO_PARAMS_T = HypoParams8D

TrackParams = namedtuple(typename='TrackParams', # pylint: disable=invalid-name
                         field_names=('t', 'x', 'y', 'z', 'azimuth', 'zenith',
                                      'length'))


Event = namedtuple(typename='Event', # pylint: disable=invalid-name
                   field_names=('event', 'pulses', 'interaction', 'neutrino',
                                'track', 'cascade', 'ml_reco', 'spe_reco'))

Pulses = namedtuple(typename='Pulses', # pylint: disable=invalid-name
                    field_names=('strings', 'oms', 'times', 'charges'))

PhotonInfo = namedtuple(typename='PhotonInfo', # pylint: disable=invalid-name
                        field_names=('count', 'theta', 'phi', 'length'))
"""Intended to contain dictionaries with DOM depth number as keys"""

BinningCoords = namedtuple(typename='BinningCoords', # pylint: disable=invalid-name
                           field_names=('t', 'r', 'theta', 'phi'))


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
        length=FTYPE(15 / 3.3 * hypo_params.track_energy)
    )
    return track_params


def power_axis(minval, maxval, n_bins, power):
    """JVS's power axis, reverse engeneered"""
    inv_power = 1/power
    liner_edges = np.linspace(np.power(minval, inv_power),
                              np.power(maxval, inv_power),
                              n_bins + 1)
    bin_edges = np.power(liner_edges, power)
    return bin_edges


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

    def load(self, events_fpath):
        """Load events from file, populating `self`.

        Parameters
        ----------
        events_fpath : string
            Path to HDF5 file

        """
        print 'loading events from path "%s"' % events_fpath
        with h5py.File(expand(events_fpath)) as h5:
            print h5
            pulses = h5[PULSE_SERIES]
            pulse_events = pulses['Event']
            self.pulses = Pulses(
                strings=pulses['string'],
                oms=pulses['om'],
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
            reco = h5[ML_RECO_NAME]
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
            reco = h5[SPE_RECO_NAME]
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


