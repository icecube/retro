# pylint: disable=wrong-import-position

"""
Events class used as container for loading events from an icetray-produced HDF5
file and accessing them in a consistent manner.
"""


from __future__ import absolute_import, division, print_function

import os
from os.path import abspath, dirname

import h5py
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DFLT_PULSE_SERIES, DFLT_ML_RECO_NAME, DFLT_SPE_RECO_NAME
from retro import Event, Pulses
from retro import expand
from retro.particles import ParticleArray


__all__ = ['Events']


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
            # Note that string and om indices are one less than their numbers
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
        for idx in range(self._num_events):
            yield self[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Convert slice into (start, stop, step) tuple
            range_args = idx.indices(len(self))
            return [self[i] for i in range(*range_args)]

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
