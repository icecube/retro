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
from retro import const
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
    i3particle = frame[reco]
    reco_dict = OrderedDict()

    reco_dict['x'] = i3particle.pos.x
    reco_dict['y'] = i3particle.pos.y
    reco_dict['z'] = i3particle.pos.z
    reco_dict['time'] = i3particle.time
    reco_dict['energy'] = i3particle.energy
    reco_dict['zenith'] = i3particle.dir.zenith
    reco_dict['azimuth'] = i3particle.dir.azimuth

    return reco_dict

    #raise NotImplementedError('extract_reco not implemented')
    ## Get millipede LLH at "truth" from a run of HybridReco (if present)
    ## (Note that this requires ``from icecube import multinest_icetray``)
    #et['llh_mc_truth'] = np.nan
    #keys = frame.keys()
    #for key in keys:
    #    hr_re_dict = key_to_hybridreco_re_dict(key)
    #    if hr_re_dict:
    #        bname = HYBRIDRECO_NAME.format(**hr_re_dict)
    #        fit_params = frame[bname + '_FitParams']
    #        et['llh_mc_truth'] = fit_params.LLH.MC_truth
    #        break


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
    pulses : list of (sd_idx, pulses) tuples
        `sd_idx` is from get_sd_idx and each `pulses` object is a 1-D array of
        dtype retro_types.PULSE_T with length the number of pulses recorded in
        that DOM.

    """
    from icecube import dataclasses, recclasses, simclasses # pylint: disable=unused-variable
    pulse_series = frame[pulse_series_name]

    if isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapMask): # pylint: disable=no-member
        pulse_series = pulse_series.apply(frame)

    if not isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMap): # pylint: disable=no-member
        raise TypeError(type(pulse_series))

    pulses = []

    for (string, dom, _), pinfo in pulse_series:
        sd_idx = const.get_sd_idx(string, dom)

        pls = []
        for pulse in pinfo:
            pls.append((pulse.time, pulse.charge, pulse.width))
        pls = np.array(pls, dtype=PULSE_T)

        pulses.append((sd_idx, pls))

    return pulses


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
    photons : list of (sd_idx, phot) tuples
        `sd_idx` is from get_sd_idx and each `phot` object is a 1D array of
        dtype retro_types.PHOTON_T with length the number of photons recorded
        in that DOM.

    """
    from icecube import dataclasses, simclasses # pylint: disable=unused-variable

    photon_series = frame[photon_key]
    photons = OrderedDict()
    photons = []
    #strs_doms = []
    for (string, dom), pinfos in photon_series:
        sd_idx = const.get_sd_idx(string, dom)

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

        photons.append((sd_idx, phot))

    return photons


def extract_mc_truth(frame, run_id, event_id):
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

    # Extract info from trueCascade
    try:
        true_cascade = frame['trueCascade']
        et['cascade_pdg'] = true_cascade.pdg_encoding
        et['cascade_energy'] = true_cascade.energy
        et['cascade_coszen'] = np.cos(true_cascade.dir.zenith)
        et['cascade_azimuth'] = true_cascade.dir.azimuth
    except KeyError:
        pass

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
        photon_names=tuple(),
        pulse_names=tuple(),
        reco_names=tuple(),
        trigger_hierarchy_names=tuple(),
        mc_truth=False
    ):
    """Extract information from an i3 file.

    Parameters
    ----------
    fpath : str

    outdir : str, optional

    photon_names : None, str, or iterable of str
        Names of photons series' to extract from each event

    pulse_names : None, str, or iterable of str
        Names of pulse series' to extract from each event

    reco_names : None, str, or iterable of str
        Names of reconstructions to extract from each event

    trigger_hierarchy_names : None, str, or iterable of str
        Names of trigger hierarchies to extract from each event

    mc_truth : bool
        Whether or not Monte Carlo truth for the event should be extracted for
        each event

    Returns
    -------
    list of OrderedDict, one per event
        Each dict contains key "meta" with sub-dict containing file metadata.
        Depending on which arguments are provided, OrderedDicts for each named
        key passed will appear as a key within the OrderedDicts named "pulses",
        "recos", "photons", and "mc_truth". E.g.:

        .. python ::

            {
                'i3_metadata': {...},
                'events': [{...}, {...}, ...],
                'mc_truth': [{'pdg': ..., 'daughters': [{...}, ...]}],
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
    mc_truths = []

    photons = OrderedDict()
    for name in photon_names:
        photons[name] = []

    pulses = OrderedDict()
    for name in pulse_names:
        pulses[name] = []

    recos = OrderedDict()
    for name in reco_names:
        recos[name] = []

    trigger_hierarchies = OrderedDict()
    for name in trigger_hierarchy_names:
        trigger_hierarchies[name] = []

    if outdir is None:
        outdir = fpath.rstrip('.bz2').rstrip('.gz').rstrip('.i3')
    mkdir(outdir)

    frame_buffer = []
    finished = False
    while not finished:
        if i3file.more():
            next_frame = i3file.pop_frame()
        else:
            finished = True

        if len(frame_buffer) == 0 or next_frame.Stop != icetray.I3Frame.DAQ:
            if next_frame.Stop in [icetray.I3Frame.DAQ,
                                   icetray.I3Frame.Physics]:
                frame_buffer.append(next_frame)
            if not finished:
                continue

        frame = frame_buffer[-1]

        i3header = frame['I3EventHeader']

        event = OrderedDict()
        event['run_id'] = run_id = i3header.run_id
        event['event_id'] = event_id = i3header.event_id
        event['start_time'] = i3header.start_time.utc_daq_time
        event['end_time'] = i3header.end_time.utc_daq_time

        if mc_truth:
            mc_truths.append(extract_mc_truth(frame, run_id=run_id,
                                              event_id=event_id))
            event['unique_id'] = unique_id = mc_truths[-1]['unique_id']

        events.append(event)

        for photon_name in photon_names:
            photons[photon_name].append(extract_photons(frame, photon_name))

        for pulse_series_name in pulse_names:
            pulses[pulse_series_name].append(extract_pulses(frame, pulse_series_name))

        for reco_name in reco_names:
            recos[reco_name].append(extract_reco(frame, reco_name))

        for trigger_hierarchy_name in trigger_hierarchy_names:
            trigger_hierarchies[trigger_hierarchy_name].append(
                extract_trigger_hierarchy(frame, trigger_hierarchy_name)
            )

        # Clear frame buffer and start a new "chain" with the next frame
        frame_buffer = [next_frame]

    photon_series_dir = join(outdir, 'photons')
    pulse_series_dir = join(outdir, 'pulses')
    recos_dir = join(outdir, 'recos')
    trigger_hierarchy_dir = join(outdir, 'triggers')

    if photon_names:
        mkdir(photon_series_dir)
    if pulse_names:
        mkdir(pulse_series_dir)
    if reco_names:
        mkdir(recos_dir)
    if trigger_hierarchy_names:
        mkdir(trigger_hierarchy_dir)

    pickle.dump(events,
                open(join(outdir, 'events.pkl'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    if mc_truth:
        pickle.dump(mc_truths,
                    open(join(outdir, 'mc_truth.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in photon_names:
        pickle.dump(photons[name],
                    open(join(photon_series_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in pulse_names:
        pickle.dump(pulses[name],
                    open(join(pulse_series_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in reco_names:
        pickle.dump(recos[name],
                    open(join(recos_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in trigger_hierarchy_names:
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
        '--photon-names', nargs='+', default=[],
        help='''Photon series names to extract from each event'''
    )
    parser.add_argument(
        '--pulse-names', nargs='+', default=[],
        help='''Pulse series names to extract from each event'''
    )
    parser.add_argument(
        '--reco-names', nargs='+', default=[],
        help='''Pulse series names to extract from each event'''
    )
    parser.add_argument(
        '--trigger-hierarchy-names', nargs='+', default=[],
        help='''Trigger hierarchy names to extract from each event'''
    )
    parser.add_argument(
        '--mc-truth', action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    extract_events(**vars(parse_args()))
