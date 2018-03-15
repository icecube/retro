#!/usr/bin/env python

"""
Extract information on events from an i3 file needed for running Retro Reco.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this i3_file except in compliance with the License.
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
from os.path import basename
import re

import numpy as np

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
    raise NotImplementedError('extract_reco not implemented')
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

    return None


def extract_pulses(frame, pulse_series_name):
    """Extract pulses from an I3 frame.

    Parameters
    ----------
    frame : icetray.I3Frame
        Frame object from which to extract the pulses.

    pulse_series_name : str
        Name of the pulse series to retrieve. If it represents a mask, the mask
        will be applied and appropriate pulses retrieved.

    Returns
    -------
    strings : shape (n_hit_doms,) int array
        IceCube string #, in [1, 86]

    doms :  shape (n_hit_doms,) int array
        IceCube DOM position on the string, in [1, 60]

    times : length n_hit_doms list of shape (n_hits_dom[i],) float arrays
        Time of the recorded hit, in nanoseconds.

    charges : length n_hit_doms list of shape (n_hits_dom[i],) float arrays
        Charge of the recorded hit, in photoelectrons.

    widths : length n_hit_doms list of shape (n_hits_dom[i],) float arrays
        Width of the recorded hit, in nanoseconds.

    """
    pulse_series = frame[pulse_series_name]

    if isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMapMask):
        pulse_series = pulse_series.apply(frame)

    if not isinstance(pulse_series, dataclasses.I3RecoPulseSeriesMap):
        raise TypeError(type(pulse_series))

    strings = []
    doms = []
    times = []
    charges = []
    widths = []

    for omkey, pulses in pulse_series:
        string = np.uint8(omkey.string)
        dom = np.uint8(omkey.om)
        this_times = []
        this_charges = []
        this_widths = []
        for pulse in pulses:
            this_times.append(pulse.time)
            this_charges.append(pulse.charge)
            this_widths.append(pulse.width)

        strings.append(string)
        doms.append(dom)
        times.append(np.float32(this_times))
        charges.append(np.float32(this_charges))
        widths.append(np.float32(this_widths))

    pulse_info = OrderedDict()
    pulse_info['strings'] = np.array(strings, dtype=np.uint8)
    pulse_info['doms'] = np.array(doms, dtype=np.uint8)
    pulse_info['times'] = times
    pulse_info['charges'] = charges
    pulse_info['widths'] = widths

    return pulse_info


def extract_photons(frame, photon_key):
    return None


def extract_mc_truth(frame, file_info=None):
    """Get event truth information from a frame.

    Parameters
    ----------
    frame : I3Frame

    Returns
    -------
    event_truth : OrderedDict

    """
    from icecube import icetray
    from icecube import dataclasses, multinest_icetray # pylint: disable=unused-variable

    event_truth = OrderedDict()

    event_truth = et = OrderedDict()

    # Extract info from I3MCTree: ...
    mctree = frame['I3MCTree']

    # ... primary particle
    primary = mctree.primaries[0]
    pdg = primary.pdg_encoding
    abs_pdg = np.abs(pdg)

    is_nu, is_charged_lepton = False, False
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
        int(1e13) * abs_pdg
        + int(1e7) * int(et['run'])
        + int(et['event_id'])
    )
    et['uid'] = uid

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


def extract_metadata_from_frame(frame, event_id_is_unique):
    event_meta = OrderedDict()
    event_header = frame['I3EventHeader']
    event_meta['run'] = event_header.run_id
    event_meta['event_id'] = event_header.event_id

    return event_meta


def extract_events(
        fpath,
        photon_series_names=tuple(),
        pulse_series_names=tuple(),
        reco_names=tuple(),
        mc_truth=False
    ):
    """Extract information from an i3 file.

    Parameters
    ----------
    fpath : str
    photon_series_names : None, str, or iterable of str
        Names of photons series' to extract from each event

    pulse_series_names : None, str, or iterable of str
        Names of pulse series' to extract from each event

    reco_names : None, str, or iterable of str
        Names of reconstructions to extract from each event

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
                'pulses': {
                    'SRTOfflinePulses': [[...], [...], ...],
                    'WaveDeformPulses': [[...], [...], ...]
                },
                'recos': {
                    'PegLeg8D': [{...}, ...],
                    'Retro8D': [{...}, ...]
                },
            }

    """
    file_meta = None
    try:
        file_meta = extract_metadata_from_filename(fpath)
    except (KeyError, ValueError):
        pass

    runs = []
    event_ids = []
    event_uids = []

    photons = OrderedDict()
    for name in photon_series_names:
        photons[name] = []

    pulses = OrderedDict()
    for name in pulse_series_names:
        pulses[name] = []

    recos = OrderedDict()
    for name in reco_names:
        recos[name] = []

    while True:
        try:
            frame = i3_file.pop_frame()
        except Exception as err:
            print(err)
            break

        frame_stop = frame.Stop
        frame_keys = frame.keys()

        if frame_stop == icetray.I3Frame.DAQ:
            if mc_truth:
                mc_truth.append(extract_mc_truth(frame, file_info=file_info))

            for photon_key in photon_series_names:
                photons[photon_key] = extract_photons(frame, photon_key)
            continue

        elif frame_stop != icetray.I3Frame.Physics:
            continue

        # The remainiing fields are to be gotten from physics frames

        for pulse_series_name in pulse_series_names:
            pulses[pulse_series_name] = extract_pulses(frame, pulse_series_name)

        for reco_name in reco_names:
            recos[reco_name] = extract_reco(frame, reco_name)

    #if run == 1 and file_info is not None:
    #    run = file_info['run']
    #else:
    #    if run != file_info['run']:
    #        raise ValueError(
    #            'Run number mismatch: filename => %d'
    #            ' while file contents => %d' % (file_info['run'], run)
    #        )


def parse_args(description=__doc__):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--fpath', required=True,
        help='''Path to i3 file'''
    )
    parser.add_argument(
        '--photons', default=[], nargs='+',
        help='''Photon series names to extract from each event'''
    )
    parser.add_argument(
    )
    parser.add_argument(
    )
    parser.add_argument(
    )
