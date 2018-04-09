#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Extract information on events from an i3 file needed for running Retro Reco.
"""

from __future__ import absolute_import, division, print_function

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

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, basename, dirname, join
import cPickle as pickle
import re
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import PHOTON_T, PULSE_T, TRIGGER_T
from retro.utils.misc import expand, mkdir


FILENAME_INFO_RE = re.compile(
    r'''
    Level(?P<proc_level>.+) # processing level e.g. 5p or 5pt (???)
    _(?P<detector>[^.]+)    # detector, e.g. IC86
    \.(?P<year>\d+)         # year
    _(?P<generator>.+)      # generator, e.g. genie
    _(?P<flavor>.+)         # flavor, e.g. nue
    \.(?P<run>\d+)          # run, e.g. 012600
    \.(?P<filenum>\d+)      # file number, e.g. 000000
    ''', (re.VERBOSE | re.IGNORECASE)
)


def extract_metadata_from_filename(fname):
    """Get info contained in an i3 filename or filepath.

    Parameters
    ----------
    fname : string
        Path to file or filename

    Returns
    -------
    file_info : OrderedDict

    """
    fname = basename(fname)
    finfo_match = next(FILENAME_INFO_RE.finditer(fname))
    if not finfo_match:
        raise ValueError('Could not interpret file path "%s"' % fname)

    finfo_dict = finfo_match.groupdict()

    file_info = OrderedDict([
        ('detector', finfo_dict['detector'].lower()),
        ('year', int(finfo_dict['year'] or -1)),
        ('generator', finfo_dict['generator'].lower()),
        ('run', int(finfo_dict['run'] or -1)),
        ('filenum', int(finfo_dict['filenum'] or -1)),
        ('proc_level', finfo_dict['proc_level'].lower()),
    ])

    return file_info


def extract_reco(frame, reco):
    """Extract a reconstruction from a frame."""
    reco_dict = OrderedDict()

    if reco.startswith('Pegleg_Fit'):
        casc_name = reco + 'HDCasc'
        track_name = reco + 'Track'
        if casc_name in frame and track_name in frame:
            neutrino = frame[reco]
            casc = frame[casc_name]
            track = frame[track_name]

            reco_dict['x'] = neutrino.pos.x
            reco_dict['y'] = neutrino.pos.y
            reco_dict['z'] = neutrino.pos.z
            reco_dict['time'] = neutrino.time
            reco_dict['energy'] = neutrino.energy
            reco_dict['zenith'] = neutrino.dir.zenith
            reco_dict['azimuth'] = neutrino.dir.azimuth
            reco_dict['track_energy'] = track.energy
            reco_dict['track_zenith'] = track.dir.zenith
            reco_dict['track_azimuth'] = track.dir.azimuth
            reco_dict['cascade_energy'] = casc.energy
        else:
            reco_dict['x'] = np.nan
            reco_dict['y'] = np.nan
            reco_dict['z'] = np.nan
            reco_dict['time'] = np.nan
            reco_dict['energy'] = np.nan
            reco_dict['zenith'] = np.nan
            reco_dict['azimuth'] = np.nan
            reco_dict['track_energy'] = np.nan
            reco_dict['track_zenith'] = np.nan
            reco_dict['track_azimuth'] = np.nan
            reco_dict['cascade_energy'] = np.nan

    # -- HybridReco, as seen in DRAGON 1{2,4,6}60 Monte Carlo -- #

    # MultiNest7D is a cascade-only fit to the event
    elif reco.endswith('MultiNest7D'):
        neutrino = frame[reco + '_Neutrino']
        casc = frame[reco + '_Cascade']

        reco_dict['x'] = neutrino.pos.x
        reco_dict['y'] = neutrino.pos.y
        reco_dict['z'] = neutrino.pos.z
        reco_dict['time'] = neutrino.time
        reco_dict['energy'] = neutrino.energy
        reco_dict['zenith'] = neutrino.dir.zenith
        reco_dict['azimuth'] = neutrino.dir.azimuth
        reco_dict['cascade_energy'] = casc.energy

    # MultiNest8D fits a cascade & track, with casscade in the track direction
    elif reco.endswith('MultiNest8D'):
        neutrino = frame[reco + '_Neutrino']
        casc = frame[reco + '_Cascade']
        track = frame[reco + '_Track']

        reco_dict['x'] = neutrino.pos.x
        reco_dict['y'] = neutrino.pos.y
        reco_dict['z'] = neutrino.pos.z
        reco_dict['time'] = neutrino.time
        reco_dict['energy'] = neutrino.energy
        reco_dict['zenith'] = neutrino.dir.zenith
        reco_dict['azimuth'] = neutrino.dir.azimuth
        reco_dict['track_energy'] = track.energy
        reco_dict['track_zenith'] = track.dir.zenith
        reco_dict['track_azimuth'] = track.dir.azimuth
        reco_dict['cascade_energy'] = casc.energy

    # MultiNest8D fits a cascade & track with their directions independnent
    elif reco.endswith('MultiNest10D'):
        neutrino = frame[reco + '_Neutrino']
        casc = frame[reco + '_Cascade']
        track = frame[reco + '_Track']

        reco_dict['x'] = neutrino.pos.x
        reco_dict['y'] = neutrino.pos.y
        reco_dict['z'] = neutrino.pos.z
        reco_dict['time'] = neutrino.time
        reco_dict['energy'] = neutrino.energy
        reco_dict['zenith'] = neutrino.dir.zenith
        reco_dict['azimuth'] = neutrino.dir.azimuth
        reco_dict['track_energy'] = track.energy
        reco_dict['track_zenith'] = track.dir.zenith
        reco_dict['track_azimuth'] = track.dir.azimuth
        reco_dict['cascade_energy'] = casc.energy
        reco_dict['cascade_zenith'] = track.dir.zenith
        reco_dict['cascade_azimuth'] = track.dir.azimuth

    # -- Anything else -- #

    else:
        i3particle = frame[reco]
        reco_dict['x'] = i3particle.pos.x
        reco_dict['y'] = i3particle.pos.y
        reco_dict['z'] = i3particle.pos.z
        reco_dict['time'] = i3particle.time
        reco_dict['energy'] = i3particle.energy
        reco_dict['zenith'] = i3particle.dir.zenith
        reco_dict['azimuth'] = i3particle.dir.azimuth

    reco = np.array(
        tuple(reco_dict.values()),
        dtype=np.dtype(list(zip(reco_dict.keys(), [np.float32]*len(reco_dict))))
    )

    return reco


def extract_trigger_hierarchy(frame, path):
    from icecube import dataclasses, recclasses, simclasses # pylint: disable=unused-variable
    trigger_hierarchy = frame[path]
    triggers = []
    for _, trigger in trigger_hierarchy.iteritems():
        config_id = trigger.key.config_id or 0
        triggers.append((
            int(trigger.key.type),
            int(trigger.key.subtype),
            int(trigger.key.source),
            config_id,
            trigger.fired,
            trigger.time,
            trigger.length
        ))
    try:
        triggers = np.array(triggers, dtype=TRIGGER_T)
    except TypeError:
        print('triggers:', triggers)
    return triggers


def extract_pulses(frame, pulse_series_name):
    """Extract a pulse series from an I3 frame.

    Parameters
    ----------
    frame : icetray.I3Frame
        Frame object from which to extract the pulses.

    pulse_series_name : str
        Name of the pulse series to retrieve. If it represents a mask, the mask
        will be applied and appropriate pulses retrieved.

    Returns
    -------
    pulses_list : list of ((string, dom, pmt), pulses) tuples
        `sd_idx` is from get_sd_idx and each `pulses` object is a 1-D array of
        dtype retro_types.PULSE_T with length the number of pulses recorded in
        that DOM.

    time_range : tuple of two floats

    """
    from icecube import dataclasses, recclasses, simclasses # pylint: disable=unused-variable
    pulse_series = frame[pulse_series_name]
    time_range_name = pulse_series_name + 'TimeRange'
    if time_range_name in frame:
        i3_time_range = frame[time_range_name]
        time_range = i3_time_range.start, i3_time_range.stop
    else:
        time_range = np.nan, np.nan

    if isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapMask): # pylint: disable=no-member
        pulse_series = pulse_series.apply(frame)

    if not isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMap): # pylint: disable=no-member
        raise TypeError(type(pulse_series))

    pulses_list = []

    for (string, dom, pmt), pinfo in pulse_series:
        pls = []
        for pulse in pinfo:
            pls.append((pulse.time, pulse.charge, pulse.width))
        pls = np.array(pls, dtype=PULSE_T)

        pulses_list.append(((string, dom, pmt), pls))

    return pulses_list, time_range


def extract_photons(frame, photon_key):
    """Extract a photon series from an I3 frame.

    Parameters
    ----------
    frame : icetray.I3Frame
        Frame object from which to extract the pulses.

    photon_key : str
        Name of the photon series to retrieve.

    Returns
    -------
    photons : list of ((string, dom, pmt), phot) tuples
        string and dom are one-indexed, while pmt is zero-indexed. $ach `phot`
        object is a 1D array of dtype retro_types.PHOTON_T with length the
        number of photons recorded in that DOM.

    """
    from icecube import dataclasses, simclasses # pylint: disable=unused-variable

    photon_series = frame[photon_key]
    photons = []
    for omkey, pinfos in photon_series:
        if len(omkey) == 2:
            string, dom = omkey
            pmt = 0
        else:
            string, dom, pmt = omkey

        phot = [] #Photon(*([] for _ in range(len(Photon._fields))))
        for pinfo in pinfos:
            phot.append((
                pinfo.time,
                pinfo.pos.x,
                pinfo.pos.y,
                pinfo.pos.z,
                np.cos(pinfo.dir.zenith),
                pinfo.dir.azimuth,
                pinfo.wavelength
            ))
        phot = np.array(phot, dtype=PHOTON_T)

        photons.append(((string, dom, pmt), phot))

    return photons


def extract_truth(frame, run_id, event_id):
    """Get event truth information from a frame.

    Parameters
    ----------
    frame : I3Frame

    Returns
    -------
    event_truth : OrderedDict

    """
    from icecube import dataclasses, icetray # pylint: disable=unused-variable
    try:
        from icecube import multinest_icetray
    except ImportError:
        multinest_icetray = None

    event_truth = et = OrderedDict()

    # Extract info from I3MCTree: ...
    mctree = frame['I3MCTree']

    # ... primary particle
    primary = mctree.primaries[0]
    pdg = primary.pdg_encoding
    abs_pdg = np.abs(pdg)

    is_nu = False
    if abs_pdg in [11, 13, 15]:
        is_charged_lepton = True
    elif abs_pdg in [12, 14, 16]:
        is_nu = True

    et['pdg'] = pdg
    et['x'] = primary.pos.x
    et['y'] = primary.pos.y
    et['z'] = primary.pos.z
    et['time'] = primary.time
    et['energy'] = primary.energy
    et['coszen'] = np.cos(primary.dir.zenith)
    et['azimuth'] = primary.dir.azimuth

    # Get event number and generate a unique ID
    unique_id = (
        int(1e13) * abs_pdg + int(1e7) * run_id + event_id
    )
    et['unique_id'] = unique_id

    # If neutrino, get charged lepton daughter particles
    if is_nu:
        daughters = mctree.get_daughters(primary)
        highest_e_daughter = None
        longest_daughter = None
        highest_energy = 0
        longest_length = 0
        for daughter in daughters:
            d_pdg = daughter.pdg_encoding
            d_energy = daughter.energy
            d_length = daughter.length

            # Look only at charged leptons
            if np.abs(d_pdg) in [11, 13, 15]:
                if d_energy > highest_energy:
                    highest_e_daughter = daughter
                    highest_energy = d_energy
                if d_length > longest_length:
                    longest_daughter = daughter
                    longest_length = d_length

        daughter_info_defaults = OrderedDict([
            ('pdg', 0),
            ('energy', np.nan),
            ('length', np.nan),
            ('coszen', np.nan),
            ('azimuth', np.nan),
        ])

        highest_e_info = OrderedDict()
        he_name = 'highest_energy_daughter'
        if highest_e_daughter:
            hei = highest_e_info
            hei['%s_pdg' % he_name] = (
                highest_e_daughter.pdg_encoding
            )
            hei['%s_energy' % he_name] = highest_e_daughter.energy
            hei['%s_length' % he_name] = highest_e_daughter.length
            hei['%s_coszen' % he_name] = (
                np.cos(highest_e_daughter.dir.zenith)
            )
            hei['%s_azimuth' % he_name] = highest_e_daughter.dir.azimuth
        else:
            for key, default_value in daughter_info_defaults.items():
                highest_e_info['%s_%s' % (he_name, key)] = default_value
        et.update(highest_e_info)

        longest_info = OrderedDict()
        l_name = 'longest_daughter'
        if longest_daughter is None:
            if highest_e_daughter:
                for subfield in daughter_info_defaults.keys():
                    l_key = '%s_%s' % (l_name, subfield)
                    he_key = '%s_%s' % (he_name, subfield)
                    longest_info[l_key] = deepcopy(highest_e_info[he_key])
            else:
                for key, default_value in daughter_info_defaults.items():
                    longest_info['%s_%s' % (l_name, key)] = default_value
        else:
            li = longest_info
            li['%s_pdg' % l_name] = longest_daughter.pdg_encoding
            li['%s_energy' % l_name] = longest_daughter.energy
            li['%s_length' % l_name] = longest_daughter.length
            li['%s_coszen' % l_name] = np.cos(longest_daughter.dir.zenith)
            li['%s_azimuth' % l_name] = longest_daughter.dir.azimuth
        et.update(longest_info)

    # Extract info from {MC,true}Cascade
    has_mc_true_cascade = True
    if 'trueCascade' in frame:
        true_cascade = frame['trueCascade']
    elif 'MCCascade' in frame:
        true_cascade = frame['MCCascade']
    else:
        has_mc_true_cascade = False

    if has_mc_true_cascade:
        et['cascade_pdg'] = true_cascade.pdg_encoding
        et['cascade_energy'] = true_cascade.energy
        et['cascade_coszen'] = np.cos(true_cascade.dir.zenith)
        et['cascade_azimuth'] = true_cascade.dir.azimuth

    # Extract per-event info from I3MCWeightDict
    mcwd_copy_keys = '''
        Crosssection
        EnergyLost
        GENIEWeight
        GlobalProbabilityScale
        InteractionProbabilityWeight
        InteractionType
        LengthInVolume
        OneWeight
        TargetPDGCode
        TotalInteractionProbabilityWeight
    '''.split()
    mcwd = frame['I3MCWeightDict']
    for key in mcwd_copy_keys:
        try:
            et[key] = mcwd[key]
        except KeyError:
            print('Could not get "{}" from I3MCWeightDict'.format(key))

    dt_spec = []
    for key in et.keys():
        if 'pdg' in key:
            dt_spec.append((key, np.int64))
        else:
            dt_spec.append((key, np.float64))

    event_truth = np.array(tuple(et.values()), dtype=np.dtype(dt_spec))
    return event_truth


def extract_metadata_from_frame(frame):
    event_meta = OrderedDict()
    event_header = frame['I3EventHeader']
    event_meta['run'] = event_header.run_id
    event_meta['event_id'] = event_header.event_id
    return event_meta


def extract_events(
        fpath,
        outdir=None,
        photons=tuple(),
        pulses=tuple(),
        recos=tuple(),
        triggers=tuple(),
        truth=False
    ):
    """Extract information from an i3 file.

    Parameters
    ----------
    fpath : str

    outdir : str, optional

    photons : None, str, or iterable of str
        Names of photons series' to extract from each event

    pulses : None, str, or iterable of str
        Names of pulse series' to extract from each event

    recos : None, str, or iterable of str
        Names of reconstructions to extract from each event

    triggers : None, str, or iterable of str
        Names of trigger hierarchies to extract from each event

    truth : bool
        Whether or not Monte Carlo truth for the event should be extracted for
        each event

    Returns
    -------
    list of OrderedDict, one per event
        Each dict contains key "meta" with sub-dict containing file metadata.
        Depending on which arguments are provided, OrderedDicts for each named
        key passed will appear as a key within the OrderedDicts named "pulses",
        "recos", "photons", and "truth". E.g.:

        .. python ::

            {
                'i3_metadata': {...},
                'events': [{...}, {...}, ...],
                'truth': [{'pdg': ..., 'daughters': [{...}, ...]}],
                'photons': {
                    'photons': [[...], [...], ...],
                    'other_photons': [[...], [...], ...]
                },
                'pulse_series': {
                    'SRTOfflinePulses': [[...], [...], ...],
                    'WaveDeformPulses': [[...], [...], ...]
                },
                'recos': {
                    'PegLeg8D': [{...}, ...],
                    'Retro8D': [{...}, ...]
                },
                'triggers': {
                    'I3TriggerHierarchy': [{...}, ...],
                }
            }

    """
    from icecube import dataclasses, dataio, icetray # pylint: disable=unused-variable

    fpath = expand(fpath)
    file_info = None
    try:
        file_info = extract_metadata_from_filename(fpath)
    except (StopIteration, KeyError, ValueError):
        pass

    i3file = dataio.I3File(fpath, 'r') # pylint: disable=no-member

    events = []
    truths = []

    photons_d = OrderedDict()
    for name in photons:
        photons_d[name] = []

    pulses_d = OrderedDict()
    for name in pulses:
        pulses_d[name] = []
        pulses_d[name + 'TimeRange'] = []

    recos_d = OrderedDict()
    for name in recos:
        recos_d[name] = []

    trigger_hierarchies = OrderedDict()
    for name in triggers:
        trigger_hierarchies[name] = []

    if outdir is None:
        outdir = fpath.rstrip('.bz2').rstrip('.gz').rstrip('.i3')
    mkdir(outdir)

    frame_buffer = []
    finished = False
    while not finished:
        if i3file.more():
            try:
                next_frame = i3file.pop_frame()
            except:
                print('Failed to pop frame, source file "{}"'.format(fpath))
                raise
        else:
            finished = True

        if len(frame_buffer) == 0 or next_frame.Stop != icetray.I3Frame.DAQ:
            if next_frame.Stop in [icetray.I3Frame.DAQ,
                                   icetray.I3Frame.Physics]:
                frame_buffer.append(next_frame)
            if not finished:
                continue

        if not frame_buffer:
            raise ValueError(
                'Empty frame buffer; possibly no Q fraomes in file "{}"'
                .format(fpath)
            )

        frame = frame_buffer[-1]

        i3header = frame['I3EventHeader']

        event = OrderedDict()
        event['run_id'] = run_id = i3header.run_id
        event['event_id'] = event_id = i3header.event_id
        event['start_time'] = i3header.start_time.utc_daq_time
        event['end_time'] = i3header.end_time.utc_daq_time

        if 'TimeShift' in frame:
            event['TimeShift'] = frame['TimeShift'].value

        if truth:
            try:
                event_truth = extract_truth(
                    frame, run_id=run_id, event_id=event_id
                )
            except:
                print('Failed to get truth form "{}" event_id {}'
                      .format(fpath, event_id))
                raise
            truths.append(event_truth)
            event['unique_id'] = unique_id = truths[-1]['unique_id']

        events.append(event)

        for photon_name in photons:
            photons_d[photon_name].append(extract_photons(frame, photon_name))

        for pulse_series_name in pulses:
            pulses_list, time_range = extract_pulses(frame, pulse_series_name)
            pulses_d[pulse_series_name].append(pulses_list)
            pulses_d[pulse_series_name + 'TimeRange'].append(time_range)

        for reco_name in recos:
            recos_d[reco_name].append(extract_reco(frame, reco_name))

        for trigger_hierarchy_name in triggers:
            trigger_hierarchies[trigger_hierarchy_name].append(
                extract_trigger_hierarchy(frame, trigger_hierarchy_name)
            )

        # Clear frame buffer and start a new "chain" with the next frame
        frame_buffer = [next_frame]

    photon_series_dir = join(outdir, 'photons')
    pulse_series_dir = join(outdir, 'pulses')
    recos_dir = join(outdir, 'recos')
    trigger_hierarchy_dir = join(outdir, 'triggers')

    if photons:
        mkdir(photon_series_dir)
    if pulses:
        mkdir(pulse_series_dir)
    if recos:
        mkdir(recos_dir)
    if triggers:
        mkdir(trigger_hierarchy_dir)

    dt_spec = []
    for key in event.keys():
        if '_time' in key:
            dt_spec.append((key, np.uint64))
        elif '_id' in key:
            dt_spec.append((key, np.int64))
        else:
            dt_spec.append((key, np.float64))

    events = np.array([tuple(ev.values()) for ev in events], dtype=dt_spec)
    np.save(join(outdir, 'events.npy'), events)

    if truth:
        np.save(join(outdir, 'truth.npy'), np.array(truths))

    for name in photons:
        pickle.dump(photons_d[name],
                    open(join(photon_series_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in pulses:
        pickle.dump(pulses_d[name],
                    open(join(pulse_series_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        np.save(join(pulse_series_dir, name + 'TimeRange' + '.npy'),
                np.array(pulses_d[name + 'TimeRange'], dtype=np.float32))

    for name in recos:
        np.save(join(recos_dir, name + '.npy'), np.array(recos_d[name]))

    for name in triggers:
        pickle.dump(trigger_hierarchies[name],
                    open(join(trigger_hierarchy_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)


def parse_args(description=__doc__):
    """Parse command line args"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--fpath', required=True,
        help='''Path to i3 file'''
    )
    parser.add_argument(
        '--outdir'
    )
    parser.add_argument(
        '--photons', nargs='+', default=[],
        help='''Photon series names to extract from each event'''
    )
    parser.add_argument(
        '--pulses', nargs='+', default=[],
        help='''Pulse series names to extract from each event'''
    )
    parser.add_argument(
        '--recos', nargs='+', default=[],
        help='''Pulse series names to extract from each event'''
    )
    parser.add_argument(
        '--triggers', nargs='+', default=[],
        help='''Trigger hierarchy names to extract from each event'''
    )
    parser.add_argument(
        '--truth', action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    extract_events(**vars(parse_args()))
