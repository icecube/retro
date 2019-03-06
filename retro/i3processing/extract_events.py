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

__all__ = [
    'EM_CASCADE_PTYPES',
    'HADR_CASCADE_PTYPES',
    'CASCADE_PTYPES',
    'TRACK_PTYPES',
    'INVISIBLE_PTYPES',
    'ELECTRONS',
    'MUONS',
    'TAUS',
    'NUES',
    'NUMUS',
    'NUTAUS',
    'FILENAME_INFO_RE',
    'GENERIC_I3_FNAME_RE',
    'I3PARTICLE_ATTRS',
    'extract_file_metadata',
    'extract_reco',
    'extract_trigger_hierarchy',
    'extract_pulses',
    'extract_photons',
    'get_cascade_and_track_info',
    'populate_track_t',
    'extract_truth',
    'extract_metadata_from_frame',
    'extract_events',
    'parse_args',
]

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
from os.path import abspath, basename, dirname, join
import cPickle as pickle
import re
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import (
    PHOTON_T, PULSE_T, TRIGGER_T, ParticleType, ParticleShape, InteractionType,
    LocationType, FitStatus, ExtractionError, TRACK_T, NO_TRACK, INVALID_TRACK,
    CASCADE_T, NO_CASCADE, INVALID_CASCADE,
)
from retro.utils.cascade_energy_conversion import em2hadr, hadr2em
from retro.utils.misc import expand, mkdir
from retro.utils.geom import cart2sph_np, sph2cart_np


EM_CASCADE_PTYPES = (
    ParticleType.EMinus,
    ParticleType.EPlus,
    ParticleType.Brems,
    ParticleType.DeltaE,
    ParticleType.PairProd,
    ParticleType.Gamma,
    ParticleType.Pi0,
)
"""Particle types parameterized as electromagnetic cascades,
from clsim/python/GetHybridParameterizationList.py"""


HADR_CASCADE_PTYPES = (
    ParticleType.Hadrons,
    ParticleType.Neutron,
    ParticleType.PiPlus,
    ParticleType.PiMinus,
    ParticleType.K0_Long,
    ParticleType.KPlus,
    ParticleType.KMinus,
    ParticleType.PPlus,
    ParticleType.PMinus,
    ParticleType.K0_Short,
    ParticleType.Eta,
    ParticleType.Lambda,
    ParticleType.SigmaPlus,
    ParticleType.Sigma0,
    ParticleType.SigmaMinus,
    ParticleType.Xi0,
    ParticleType.XiMinus,
    ParticleType.OmegaMinus,
    ParticleType.NeutronBar,
    ParticleType.LambdaBar,
    ParticleType.SigmaMinusBar,
    ParticleType.Sigma0Bar,
    ParticleType.SigmaPlusBar,
    ParticleType.Xi0Bar,
    ParticleType.XiPlusBar,
    ParticleType.OmegaPlusBar,
    ParticleType.DPlus,
    ParticleType.DMinus,
    ParticleType.D0,
    ParticleType.D0Bar,
    ParticleType.DsPlus,
    ParticleType.DsMinusBar,
    ParticleType.LambdacPlus,
    ParticleType.WPlus,
    ParticleType.WMinus,
    ParticleType.Z0,
    ParticleType.NuclInt,
    ParticleType.TauPlus,
    ParticleType.TauMinus,
)
"""Particle types parameterized as hadronic cascades,
from clsim/CLSimLightSourceToStepConverterPPC.cxx with addition of TauPlus and
TauMinus"""


CASCADE_PTYPES = EM_CASCADE_PTYPES + HADR_CASCADE_PTYPES
"""Particle types classified as either EM or hadronic cascades"""


TRACK_PTYPES = (ParticleType.MuPlus, ParticleType.MuMinus)
"""Particle types classified as tracks"""


INVISIBLE_PTYPES = (
    ParticleType.Neutron, # long decay time exceeds trigger window
    ParticleType.K0,
    ParticleType.K0Bar,
    ParticleType.NuE,
    ParticleType.NuEBar,
    ParticleType.NuMu,
    ParticleType.NuMuBar,
    ParticleType.NuTau,
    ParticleType.NuTauBar,
)
"""Invisible particles (at least to low-energy IceCube triggers)"""

ELECTRONS = (ParticleType.EPlus, ParticleType.EMinus)
MUONS = (ParticleType.MuPlus, ParticleType.MuMinus)
TAUS = (ParticleType.TauPlus, ParticleType.TauMinus)
NUES = (ParticleType.NuE, ParticleType.NuEBar)
NUMUS = (ParticleType.NuMu, ParticleType.NuMuBar)
NUTAUS = (ParticleType.NuTau, ParticleType.NuTauBar)

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

GENERIC_I3_FNAME_RE = re.compile(
    r'''
    ^                              # Anchor to beginning of string
    (?P<base>.*)                   # Any number of any character
    (?P<i3ext>\.i3)                # Must have ".i3" as extension
    (?P<compext>\.gz|bz2|zst|zstd) # Optional extension indicating compression
    $                              # End of string
    ''', (re.VERBOSE | re.IGNORECASE)
)

I3PARTICLE_ATTRS = OrderedDict([
    ('major_id', dict(dtype=np.uint64, default=0)),
    ('minor_id', dict(dtype=np.int32, default=0)),
    ('zenith', dict(path='dir.zenith', dtype=np.float32, default=np.nan)),
    ('azimuth', dict(path='dir.azimuth', dtype=np.float32, default=np.nan)),
    ('x', dict(path='pos.x', dtype=np.float32, default=np.nan)),
    ('y', dict(path='pos.y', dtype=np.float32, default=np.nan)),
    ('z', dict(path='pos.z', dtype=np.float32, default=np.nan)),
    ('time', dict(dtype=np.float32, default=np.nan)),
    ('energy', dict(dtype=np.float32, default=np.nan)),
    ('speed', dict(dtype=np.float32, default=np.nan)),
    ('length', dict(dtype=np.float32, default=np.nan)),
    ('type', dict(enum=ParticleType, dtype=np.int32, default=ParticleType.unknown)),
    ('pdg_encoding', dict(enum=ParticleType, dtype=np.int32, default=ParticleType.unknown)),
    ('shape', dict(enum=ParticleShape, dtype=np.uint8, default=ParticleShape.Null)),
    ('fit_status', dict(enum=FitStatus, dtype=np.int8, default=FitStatus.NotSet)),
    ('location_type', dict(enum=LocationType, dtype=np.uint8, default=LocationType.Anywhere)),
])


def extract_file_metadata(fname):
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

        dt_spec = [(k, np.float32) for k in reco_dict.keys()]

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

        dt_spec = [(k, np.float32) for k in reco_dict.keys()]

    # MultiNest8D fits a cascade & track, with casscade in the track direction
    elif reco.endswith('MultiNest8D'):
        if reco + '_Neutrino' in frame:
            neutrino = frame[reco + '_Neutrino']
        elif reco + '_NumuCC' in frame:
            neutrino = frame[reco + '_NumuCC']

        if reco + '_Cascade' in frame:
            casc = frame[reco + '_Cascade']
        elif reco + '_HDCasc' in frame:
            casc = frame[reco + '_HDCasc']
        else:
            raise ValueError('Cannot find cascade in frame')

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

        dt_spec = [(k, np.float32) for k in reco_dict.keys()]

    # MultiNest10D fits a cascade & track with their directions independnent
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

        dt_spec = [(k, np.float32) for k in reco_dict.keys()]

    # -- Anything else -- #

    else:
        dt_spec = [(attr, info['dtype']) for attr, info in I3PARTICLE_ATTRS.items()]
        if reco in frame:
            i3particle = frame[reco]
            for attr, info in I3PARTICLE_ATTRS.items():
                # If "path" key present in `info`, get its value; otherwise,
                # path is just attr's name
                path = info.get('path', attr)

                # Recursively apply getattr on the i3particle for each path
                reco_dict[attr] = reduce(getattr, path.split('.'), i3particle)
        else:
            for attr, info in I3PARTICLE_ATTRS.items():
                reco_dict[attr] = info['default']

    # -- (subset of) PID and cut vars -- #

    if reco.startswith('IC86_Dunkman_L6') and 'IC86_Dunkman_L6' in frame:
        cutvars = frame['IC86_Dunkman_L6']
        for var in ['delta_LLH', 'mn_start_contained', 'mn_stop_contained']:
            try:
                reco_dict[var] = getattr(cutvars, var)
            except AttributeError:
                pass

    reco = np.array(tuple(reco_dict.values()), dtype=dt_spec)

    return reco


def extract_trigger_hierarchy(frame, path):
    """Extract trigger hierarchy from an I3 frame.

    Parameters
    ----------
    frame : icetray.I3Frame
        Frame object from which to extract the trigger hierarchy

    path : string
        Path to trigger hierarchy

    Returns
    -------
    triggers : length n_triggers array of dtype retro_types.TRIGGER_T

    """
    from icecube import dataclasses, recclasses, simclasses  # pylint: disable=unused-variable

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
        sys.stderr.write('triggers: {}\n'.format(triggers))
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

    time_range : tuple of two floats or None
        None is returned if the <pulses>TimeRange field is missing

    """
    from icecube import dataclasses, recclasses, simclasses  # pylint: disable=unused-variable
    from icecube.dataclasses import I3RecoPulseSeriesMap, I3RecoPulseSeriesMapMask  # pylint: disable=no-name-in-module

    pulse_series = frame[pulse_series_name]

     time_range_name = None
    if pulse_series_name + 'TimeRange' in frame:
        time_range_name = pulse_series_name + 'TimeRange'
    elif 'UncleanedInIcePulsesTimeRange' in frame:
        # TODO: use WaveformRange instead?
        time_range_name = 'UncleanedInIcePulsesTimeRange'

    if time_range_name is not None:
        i3_time_range = frame[time_range_name]
        time_range = i3_time_range.start, i3_time_range.stop
    else:
        time_range = None

    if isinstance(pulse_series, I3RecoPulseSeriesMapMask):
        pulse_series = pulse_series.apply(frame)

    if not isinstance(pulse_series, I3RecoPulseSeriesMap):
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
    from icecube import dataclasses, recclasses, simclasses  # pylint: disable=unused-variable

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


def get_cascade_and_track_info(particles, mctree):
    """Extract summary information about an event's "true" cascade(s) and
    track(s), insofar as we can discern from the IceCube low-en MC.

    Parameters
    ----------
    particles : icecube.dataclasses.I3Particle or iterable thereof
        Particles in the passed `mctree` to be analyzed for their "cascade-"
        and "track-ness."

    mctree : icecube.dataclasses.I3MCTree

    Returns
    -------
    cascade_and_track_info : OrderedDict
        Values are numpy struct arrays, and keys are
            "total_track"
                Simple sum of all particles that self-report as `is_track`, exept taus
                (see Notes); total track has length only as long as the longest
                component (it doesn't make sense to report a sum of lengths since they
                all start from the same point), but energy is summed over all
                components and directionality is the length-weighted average of all
                component particles

            "longest_track"
                Single longest particle classified as a track; filled with nan if no
                tracks

            "total_cascade"
                Simple sum of all particles that self-report as `is_cascade` plus taus
                which are grouped into cascades despite self-reporting as `is_track`
                since they decay so quickly

            "vis_em_equiv_cascade"
                Simply apply the inverse of the `F` factor from ref. [1] to the
                energies of the hadronic cascades and average all resulting cascading
                particles, including taus but omitting neutrons

    Notes
    -----
    Direction is reported by the energy-weighted average of all grouped
    particles with `directionality` a normalized (0-1) measure of the resulting
    direction vector.

    Taus are simplistically grouped with hadronic cascades since these do not
    (the majority of the time) produce a track detectable as such in the ice by
    the IceCube detector. This ignores the details of the tau propagation and
    decay products that will produce electromagnetic showers and possibly a
    detectable track (a muon byproduct), but we have to make a simplistic
    choice since we don't have further information about the tau's behavior in
    our MC files.

    The `vis_em_equiv_cascade` can be improved by applying separate factors
    from ref. [2] for different hadrons (and potentially deriving new factors
    from simulations of all cascading particles).

    In the end, without detailed reporting of the among of Cherenkov light
    produced in each event and detailed accounting of this for what we would
    call "tracks" or "cascades," the best we might hope to achieve in reporting
    "truth" here is to report expectation values for the last step reported in
    the I3MCTree. For each event, truth will be incorrect, but looking at many
    events, bias will average out (although the standard deviation between
    actual-truth and average-truth introduced by our uncertainty of what
    actually occurs in MC will still impact reconstruction errors).

    References
    ----------
    [1] D. Chirkin for the IceCube Collaboration, "Study of South Pole ice
    transparency with IceCube flashers"

    [2] M. Kowalski, "On the Čerenkov light emission of hadronic and
    electro-magnetic cascades," AMANDA-IR/20020803 August 12, 2002.

    """
    from icecube.dataclasses import I3Particle  # pylint: disable=no-name-in-module

    ignore_ptypes = (
        ParticleType.NuE,
        ParticleType.NuEBar,
        ParticleType.NuMu,
        ParticleType.NuMuBar,
        ParticleType.NuTau,
        ParticleType.NuTauBar,
        ParticleType.O16Nucleus,
        ParticleType.K0,
        ParticleType.K0Bar,
    )
    if isinstance(particles, I3Particle):
        particles = [particles]

    # Storage for particles we classify as each basic/simplistic topology
    cascades = []
    tracks = []

    for particle in particles:
        ptype = ParticleType(int(particle.pdg_encoding))
        if ptype in EM_CASCADE_PTYPES:
            assert particle.is_cascade, repr(ptype)
            cascades.append((particle, False))
        elif ptype in HADR_CASCADE_PTYPES:
            assert particle.is_cascade or ptype in TAUS, repr(ptype)
            cascades.append((particle, True))
        elif ptype in TRACK_PTYPES:
            assert particle.is_track, repr(ptype)
            tracks.append(particle)
        elif ptype in ignore_ptypes:
            pass
        else:
            #raise ValueError("{} is not track or cascade".format(ptype))
            sys.stderr.write("{!r} is neither track nor cascade\n".format(ptype))

    if tracks:
        longest_track_particle = None
        total_track = np.zeros(shape=1, dtype=TRACK_T)
        wtd_dir = np.zeros(3) # direction cosines (x, y, & z)
        sum_of_lengths = 0.
        secondaries = []
        for particle in tracks:
            if particle.length >= total_track['length']:
                total_track['length'] = particle.length
                longest_track_particle = particle
            total_track['energy'] += particle.energy
            # Note negative sign is due to dir.x, dir.y, dir.z indicating
            # direction _toward which particle points_, while icecube zenith
            # and azimuth describe direction _from which particle came_.
            wtd_dir -= particle.length * np.array((particle.dir.x, particle.dir.y, particle.dir.z))
            sum_of_lengths += particle.length
            secondaries.extend(mctree.get_daughters(particle))

        info = get_cascade_and_track_info(
            particles=secondaries,
            mctree=mctree,
        )

        wtd_dir /= sum_of_lengths
        directionality, zenith, azimuth = cart2sph_np(x=wtd_dir[0], y=wtd_dir[1], z=wtd_dir[2])

        longest_track = populate_track_t(mctree=mctree, particle=longest_track_particle)

        total_track['time'] = tracks[0]['time']
        total_track['x'] = tracks[0]['x']
        total_track['y'] = tracks[0]['y']
        total_track['z'] = tracks[0]['z']
        total_track['zenith'] = zenith
        total_track['azimuth'] = azimuth
        total_track['directionality'] = directionality
        # (energy and length already set inside above loop)
        total_track['stochastic_loss'] = info['total_cascade']['energy']
        total_track['vis_em_equiv_stochastic_loss'] = info['vis_em_equiv_cascade']['energy']
        if len(tracks) == 1:
            total_track['pdg'] = tracks[0]['pdg_encoding']
        else:
            total_track['pdg'] = ParticleType.unknown

    else:
        total_track = deepcopy(NO_TRACK)
        longest_track = deepcopy(NO_TRACK)

    if cascades:
        total_cascade = np.zeros(shape=1, dtype=CASCADE_T)
        vis_em_equiv_cascade = np.zeros(shape=1, dtype=CASCADE_T)
        vis_em_wtd_dir = np.zeros(3) # direction cosines (x, y, & z)
        energy_wtd_dir = np.zeros(3) # direction cosines (x, y, & z)
        for particle, is_hadr in cascades:
            total_cascade['energy'] += particle.energy
            raw_dir = np.array((particle.dir.x, particle.dir.y, particle.dir.z))
            energy_wtd_dir -= particle.energy * raw_dir

            if particle in INVISIBLE_PTYPES:
                continue

            if is_hadr:
                try:
                    vis_em_equiv_energy = hadr2em(particle.energy)
                except:
                    sys.stderr.write(
                        'pdg={}, energy={}\n'.format(particle.pdg_encoding, particle.energy)
                    )
                    raise
            else:
                vis_em_equiv_energy = particle.energy
            vis_em_wtd_dir -= vis_em_equiv_energy * raw_dir
            vis_em_equiv_cascade['energy'] += vis_em_equiv_energy
    else:
        total_cascade = deepcopy(NO_CASCADE)
        vis_em_equiv_cascade = deepcopy(NO_CASCADE)

    cascade_and_track_info = OrderedDict(
        [
            ('total_track', total_track),
            ('longest_track', longest_track),
            ('total_cascade', total_cascade),
            ('vis_em_equiv_cascade', vis_em_equiv_cascade),
        ]
    )

    return cascade_and_track_info


def populate_track_t(mctree, particle):
    """Populate a Retro TRACK_T from a track-like IceCube I3Particle

    Parameters
    ----------
    mctree : icecube.dataclasses.I3MCTree
    particle : icecube.dataclasses.I3Particle

    Returns
    -------
    track : length-1 array of dtype retro.retro_types.TRACK_T

    """
    # Start with an invalid track, so fields not explicitly populated are
    # explicitly invalid values (as much as we can define such values)
    track = deepcopy(NO_TRACK)

    # Populate the basics
    track['time'] = particle.time
    track['x'] = particle.pos.x
    track['y'] = particle.pos.y
    track['z'] = particle.pos.z
    track['zenith'] = particle.dir.zenith
    track['azimuth'] = particle.dir.azimuth
    track['directionality'] = 1
    track['energy'] = particle.energy
    track['length'] = particle.length
    track['pdg'] = particle.pdg_encoding

    # Get info about stochastics recorded as daughters of the track particle
    info = get_cascade_and_track_info(particles=mctree.get_daughters(particle), mctree=mctree)
    track['stochastic_loss'] = info['total_cascade']['energy']
    track['vis_em_equiv_stochastic_loss'] = info['vis_em_equiv_cascade']['energy']

    return track


def extract_truth(frame, run_id, event_id):
    """Get event truth information from a frame.

    Parameters
    ----------
    frame : I3Frame

    Returns
    -------
    event_truth : OrderedDict

    """
    from icecube import dataclasses, icetray  # pylint: disable=unused-variable
    from icecube.dataclasses import I3Direction  # pylint: disable=no-name-in-module
    try:
        from icecube import multinest_icetray  # pylint: disable=unused-variable
    except ImportError:
        multinest_icetray = None

    # Anything not listed defaults to float32; note this is augmented when
    # cascades and tracks are added
    truth_dtypes = dict(
        pdg=np.int32,
        unique_id=np.uint64,
        highest_energy_daughter_pdg=np.int32,
        longest_daughter_pdg=np.int32,
        InteractionType=np.int8,
        TargetPDGCode=np.int32,
        extraction_error=np.uint8,
    )

    event_truth = OrderedDict()

    # Extract info from I3MCTree: ...
    mctree = frame['I3MCTree']

    # ... primary particle
    primary = mctree.primaries[0]
    primary_pdg = primary.pdg_encoding

    # TODO: deal with charged leptons e.g. for CORSIKA/MuonGun

    is_nu = False
    is_charged_lepton = False
    if np.abs(primary_pdg) in [11, 13, 15]:
        is_charged_lepton = True
    elif np.abs(primary_pdg) in [12, 14, 16]:
        is_nu = True

    event_truth['pdg'] = primary_pdg
    event_truth['x'] = primary.pos.x
    event_truth['y'] = primary.pos.y
    event_truth['z'] = primary.pos.z
    event_truth['time'] = primary.time
    event_truth['energy'] = primary.energy
    event_truth['coszen'] = np.cos(primary.dir.zenith)
    event_truth['azimuth'] = primary.dir.azimuth
    event_truth['extraction_error'] = ExtractionError.NO_ERROR

    # Get event number and generate a unique ID
    unique_id = (
        int(1e13) * np.abs(primary_pdg) + int(1e7) * run_id + event_id
    )
    event_truth['unique_id'] = unique_id

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
        weight
    '''.split()
    mcwd = frame['I3MCWeightDict']
    for key in mcwd_copy_keys:
        try:
            event_truth[key] = mcwd[key]
        except KeyError:
            sys.stderr.write('Could not get "{}" from I3MCWeightDict\n'.format(key))
            raise

    # If neutrino, get charged lepton daughter particles
    if is_nu:
        # By default, track and cascades all "zero"; convention is that
        # cascade0 is on lepton side of interaction
        track = deepcopy(NO_TRACK)
        cascade0 = deepcopy(NO_CASCADE)
        cascade1 = deepcopy(NO_CASCADE)
        total_cascade = deepcopy(NO_CASCADE)

        secondaries = mctree.get_daughters(primary)
        interaction_type = InteractionType(int(event_truth['InteractionType']))

        # neutrino 4-momentum
        primary_p4 = (
            primary.energy * np.array([1, primary.dir.x, primary.dir.y, primary.dir.z])
        )

        if interaction_type is InteractionType.CC:
            # Find the charged lepton generated in the interaction
            charged_lepton = None
            for secondary in secondaries:
                # charged lepton PDG one less than corresponding neutrino's PDG
                if (
                    abs(int(secondary.pdg_encoding)) == abs(primary_pdg) - 1
                    and secondary.time == primary.time
                    and secondary.pos == primary.pos
                ):
                    if charged_lepton is None:
                        charged_lepton = secondary
                    elif secondary.energy > charged_lepton.energy:
                        charged_lepton = secondary

            if charged_lepton is None:
                msg = "ERROR: Couldn't find charged lepton daughter in CC MCTree"
                event_truth['extraction_error'] = ExtractionError.NU_CC_LEPTON_SECONDARY_MISSING
                sys.stderr.write(msg + "\n")
                #raise ValueError(msg)
                if primary_pdg in NUES + NUTAUS:
                    cascade0 = INVALID_CASCADE
                elif primary_pdg in NUMUS:
                    track = INVALID_TRACK
                else:
                    raise ValueError()

            else:
                charged_lepton_pdg = ParticleType(int(charged_lepton.pdg_encoding))

                # 4-momentum; note I3Particle.energy is particle's kinetic energy
                # and I3Particle.dir.{x, y, z} point in direction of particle's
                # travel while I3Particle.dir.{zenith, azimuth} point oppositely
                # to particle's direction of travel
                charged_lepton_p4 = np.array([
                    charged_lepton.energy + charged_lepton.mass,
                    charged_lepton.dir.x * charged_lepton.energy,
                    charged_lepton.dir.y * charged_lepton.energy,
                    charged_lepton.dir.z * charged_lepton.energy,
                ])
                remaining_p4 = primary_p4 - charged_lepton_p4
                remaining_energy = remaining_p4[0]
                remaining_dir = I3Direction(*remaining_p4[1:])

                # All these fields are the same for all CC events
                for cascade in (cascade0, cascade1, total_cascade):
                    cascade['time'] = charged_lepton.time
                    cascade['x'] = charged_lepton.pos.x
                    cascade['y'] = charged_lepton.pos.y
                    cascade['z'] = charged_lepton.pos.z

                if charged_lepton_pdg in ELECTRONS:
                    cascade0['pdg'] = charged_lepton_pdg
                    cascade0['zenith'] = charged_lepton.dir.zenith
                    cascade0['azimuth'] = charged_lepton.dir.azimuth
                    cascade0['directionality'] = 1
                    cascade0['energy'] = charged_lepton.energy
                    cascade0['hadr_fraction'] = 0
                    cascade0['em_equiv_energy'] = charged_lepton.energy
                    cascade0['hadr_equiv_energy'] = em2hadr(charged_lepton.energy)

                    cascade1['pdg'] = ParticleType.unknown
                    cascade1['zenith'] = remaining_dir.zenith
                    cascade1['azimuth'] = remaining_dir.azimuth
                    cascade1['directionality'] = 1
                    cascade1['energy'] = remaining_energy
                    cascade1['hadr_fraction'] = 1
                    cascade1['em_equiv_energy'] = hadr2em(remaining_energy)
                    cascade1['hadr_equiv_energy'] = remaining_energy

                elif charged_lepton_pdg in MUONS:
                    track = populate_track_t(mctree=mctree, particle=charged_lepton)

                    cascade1['pdg'] = ParticleType.unknown
                    cascade1['zenith'] = remaining_dir.zenith
                    cascade1['azimuth'] = remaining_dir.azimuth
                    cascade1['directionality'] = np.nan
                    cascade1['energy'] = remaining_energy
                    cascade1['hadr_fraction'] = 1
                    cascade1['em_equiv_energy'] = hadr2em(remaining_energy)
                    cascade1['hadr_equiv_energy'] = remaining_energy

                elif charged_lepton_pdg in TAUS:
                    # Until we can see which taus have muon decay product, this is
                    # as good as we can do (i.e. keep track as NO_TRACK, assume all
                    # tau's energy in tau-based hadronic cascade and remaining
                    # energy in a separate hadronic cascade)
                    cascade0['pdg'] = charged_lepton_pdg
                    cascade0['zenith'] = charged_lepton.dir.zenith
                    cascade0['azimuth'] = charged_lepton.dir.azimuth
                    cascade0['directionality'] = 1
                    cascade0['energy'] = charged_lepton.mass + charged_lepton.energy
                    cascade0['hadr_fraction'] = 1
                    cascade0['em_equiv_energy'] = hadr2em(charged_lepton.energy)
                    cascade0['hadr_equiv_energy'] = charged_lepton.energy

                    cascade1['pdg'] = ParticleType.unknown
                    cascade1['zenith'] = remaining_dir.zenith
                    cascade1['azimuth'] = remaining_dir.azimuth
                    cascade1['directionality'] = None
                    cascade1['energy'] = remaining_energy
                    cascade1['hadr_fraction'] = 1
                    cascade1['em_equiv_energy'] = hadr2em(remaining_energy)
                    cascade1['hadr_equiv_energy'] = remaining_energy

                else:
                    raise ValueError("unrecognized PDG code : {}".format(charged_lepton_pdg))

        elif interaction_type is InteractionType.NC:
            outgoing_nu = None
            for secondary in secondaries:
                if (
                    int(secondary.pdg_encoding) == primary_pdg
                    and secondary.time == primary.time
                    and secondary.pos == primary.pos
                ):
                    if outgoing_nu is None:
                        outgoing_nu = secondary
                    elif secondary.energy > outgoing_nu.energy:
                        outgoing_nu = secondary
            if outgoing_nu is None:
                msg = "ERROR: Couldn't find outgoing neutrino in NC MCTree"
                event_truth['extraction_error'] = ExtractionError.NU_NC_OUTOING_NU_MISSING
                sys.stderr.write(msg + "\n")
                cascade1 = INVALID_CASCADE
                #raise ValueError(msg)

            else:
                # No track and no cascade from lepton (escaping neutrino is
                # invisible); energy not carried off by neutrino is dumped into a
                # hadronic cascade

                # 4-momentum
                outgoing_nu_p4 = (
                    outgoing_nu.energy
                    * np.array([1, outgoing_nu.dir.x, outgoing_nu.dir.y, outgoing_nu.dir.z])
                )
                remaining_p4 = primary_p4 - outgoing_nu_p4
                remaining_energy = remaining_p4[0]
                remaining_dir = I3Direction(*remaining_p4[1:])

                cascade1['pdg'] = ParticleType.unknown
                cascade1['time'] = outgoing_nu.time
                cascade1['x'] = outgoing_nu.pos.x
                cascade1['y'] = outgoing_nu.pos.y
                cascade1['z'] = outgoing_nu.pos.z
                cascade1['zenith'] = remaining_dir.zenith
                cascade1['azimuth'] = remaining_dir.azimuth
                cascade1['directionality'] = 1
                cascade1['energy'] = remaining_energy
                cascade1['hadr_fraction'] = 1
                cascade1['em_equiv_energy'] = hadr2em(remaining_energy)
                cascade1['hadr_equiv_energy'] = remaining_energy

        else: # interaction_type is InteractionType.undefined
            from icecube import genie_icetray  # pylint: disable=unused-variable
            grd = frame['I3GENIEResultDict']
            if not grd['nuel']: # nuel=True means elastic collision, apparently
                grd_s = '{{{}}}'.format(
                    ', '.join("'{}': {!r}".format(k, grd[k]) for k in sorted(grd.keys()))
                )
                raise ValueError(
                    "Not recognized as NC, CC, or elastic. I3GENIEResultDict:\n{}".format(grd_s)
                )
            if len(secondaries) > 1:
                raise NotImplementedError("more than one secondary reported in elastic collision")
            secondary = secondaries[0]
            secondary_pdg = ParticleType(int(secondary.pdg_encoding))
            if secondary_pdg in EM_CASCADE_PTYPES:
                is_em = True
            elif secondary_pdg in HADR_CASCADE_PTYPES:
                is_em = False
            else:
                raise NotImplementedError("{!r} not handled".format(secondary_pdg))

            if secondary_pdg in MUONS:
                raise NotImplementedError()

            else:
                if secondary_pdg in TAUS:
                    secondary_energy = secondary.energy + secondary.mass
                else:
                    secondary_energy = secondary.energy

                cascade1['pdg'] = int(secondary_pdg)
                cascade1['time'] = secondary.time
                cascade1['x'] = secondary.pos.x
                cascade1['y'] = secondary.pos.y
                cascade1['z'] = secondary.pos.z
                cascade1['zenith'] = secondary.dir.zenith
                cascade1['azimuth'] = secondary.dir.azimuth
                cascade1['directionality'] = 1
                cascade1['energy'] = secondary_energy
                cascade1['hadr_fraction'] = 0 if is_em else 1
                cascade1['em_equiv_energy'] = (
                    secondary_energy if is_em else hadr2em(secondary_energy)
                )
                cascade1['hadr_equiv_energy'] = (
                    em2hadr(secondary_energy) if is_em else secondary_energy
                )

        num_cascades = 0
        total_cascade_pdg = None
        total_energy = 0
        total_hadr_equiv_energy = 0
        total_em_equiv_energy = 0
        total_hadr_fraction = 0
        em_equiv_energy_weighted_dirvec = np.zeros(shape=3)

        for cascade in (cascade0, cascade1):
            if cascade['energy'] == 0:
                continue
            num_cascades += 1

            if total_cascade_pdg is None:
                total_cascade_pdg = cascade['pdg']
            elif cascade['pdg'] != total_cascade_pdg:
                total_cascade_pdg = ParticleType.unknown

            total_energy += cascade['energy']
            total_hadr_fraction += cascade['hadr_fraction']
            em_equiv_energy = cascade['em_equiv_energy']
            total_em_equiv_energy += em_equiv_energy
            total_hadr_equiv_energy += cascade['hadr_equiv_energy']

            # note that (theta, phi) points oppositely to direction defined by
            # (zenith, azimuth)
            neg_dirvec = sph2cart_np(
                r=em_equiv_energy,
                theta=cascade['zenith'],
                phi=cascade['azimuth'],
            )
            em_equiv_energy_weighted_dirvec -= np.concatenate(neg_dirvec)

        total_hadr_fraction /= num_cascades
        em_equiv_energy_weighted_dirvec /= total_em_equiv_energy

        directionality, zenith, azimuth = cart2sph_np(
            x=-em_equiv_energy_weighted_dirvec[0],
            y=-em_equiv_energy_weighted_dirvec[1],
            z=-em_equiv_energy_weighted_dirvec[2],
        )

        total_cascade['pdg'] = total_cascade_pdg
        total_cascade['time'] = primary.time
        total_cascade['x'] = primary.pos.x
        total_cascade['y'] = primary.pos.y
        total_cascade['z'] = primary.pos.z
        total_cascade['zenith'] = zenith
        total_cascade['azimuth'] = azimuth
        total_cascade['directionality'] = directionality
        total_cascade['energy'] = total_energy
        total_cascade['hadr_fraction'] = total_hadr_fraction
        total_cascade['em_equiv_energy'] = total_em_equiv_energy
        total_cascade['hadr_equiv_energy'] = total_hadr_equiv_energy

        record_particles = OrderedDict(
            (
                ('track', track),
                ('cascade0', cascade0),
                ('cascade1', cascade1),
                ('total_cascade', total_cascade),
            )
        )
        for particle_name, particle in record_particles.items():
            dtype = particle.dtype
            # note `fields` attr is un-ordered, while `names` IS ordered
            fields = dtype.fields
            for field_name in dtype.names:
                key = '{}_{}'.format(particle_name, field_name)
                event_truth[key] = particle[field_name]
                # `fields` contains dtype and byte offset; just want dtype
                truth_dtypes[key] = fields[field_name][0]

    else:  # is not neutrino:
        raise NotImplementedError("Only neutrino primaries are implemented")

    struct_dtype_spec = []
    for key in event_truth.keys():
        # default to float32 if dtype not explicitly defined
        struct_dtype_spec.append((key, truth_dtypes.get(key, np.float32)))

    event_truth = np.array(tuple(event_truth.values()), dtype=struct_dtype_spec)

    return event_truth


def extract_metadata_from_frame(frame):
    """Extract metadata from I3 frame.

    Parameters
    ----------
    frame : icetray.I3Frame

    Returns
    -------
    event_meta : OrderedDict
        Keys are 'I3EventHeader', 'run', and 'event_id'

    """
    event_meta = OrderedDict()
    event_header = frame['I3EventHeader']
    event_meta['run_id'] = event_header.run_id
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
    """Extract event information from an i3 file.

    Parameters
    ----------
    fpath : str
        Path to I3 file

    outdir : str, optional
        Directory in which to place generated files

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
    from icecube import dataclasses, recclasses, simclasses  # pylint: disable=unused-variable
    from icecube.icetray import I3Frame  # pylint: disable=no-name-in-module
    from icecube.dataio import I3File  # pylint: disable=no-name-in-module

    # Anything not listed defaults to float32
    event_dtypes = dict(
        run_id=np.uint32,
        sub_run_id=np.uint32,
        event_id=np.uint32,
        sub_event_id=np.uint32,
        unique_id=np.uint64,
        state=np.uint32,
        start_time=np.uint64,
        stop_time=np.uint64,
    )

    fpath = expand(fpath)
    sha256_sum = sha256(open(fpath, 'rb').read()).hexdigest()
    i3file = I3File(fpath, 'r')

    events = []
    truths = []

    photons_d = OrderedDict()
    for name in photons:
        photons_d[name] = []

    pulses_d = OrderedDict()
    for name in pulses:
        pulses_d[name] = []

    recos_d = OrderedDict()
    for name in recos:
        recos_d[name] = []

    trigger_hierarchies = OrderedDict()
    for name in triggers:
        trigger_hierarchies[name] = []

    # Default to dir same path as I3 file but with ".i3<compr ext>" removed
    if outdir is None:
        fname_parts = GENERIC_I3_FNAME_RE.match(fpath).groupdict()
        outdir = fname_parts['base']
    mkdir(outdir)

    # Write SHA-256 in format compatible with `sha256` Linux utility
    with open(join(outdir, "source_i3_file_sha256sum.txt"), "w") as sha_file:
        sha_file.write("{}  {}\n".format(sha256_sum, fpath))

    frame_buffer = []
    frame_counter = 0
    finished = False
    while not finished:
        try:
            if i3file.more():
                try:
                    next_frame = i3file.pop_frame()
                    frame_counter += 1
                except:
                    sys.stderr.write('Failed to pop frame #{}\n'.format(frame_counter + 1))
                    raise
            else:
                finished = True

            if len(frame_buffer) == 0 or next_frame.Stop != I3Frame.DAQ:
                if next_frame.Stop in [I3Frame.DAQ, I3Frame.Physics]:
                    frame_buffer.append(next_frame)
                if not finished:
                    continue

            if not frame_buffer:
                raise ValueError(
                    'Empty frame buffer; possibly no Q frames in file "{}"'
                    .format(fpath)
                )

            frame = frame_buffer[-1]

            i3header = frame['I3EventHeader']

            event = OrderedDict()
            event['run_id'] = run_id = i3header.run_id
            event['sub_run_id'] = i3header.sub_run_id
            event['event_id'] = event_id = i3header.event_id
            event['sub_event_id'] = i3header.sub_event_id
            # TODO: map "state" string to uint enum value if defined in I3
            # software? np dtypes don't handle ragged data like strings but I don't
            # want to invent an encoding of our own if at all possible
            #event['sub_event_stream'] = sub_event_stream
            event['state'] = i3header.state

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
                    sys.stderr.write(
                        'Failed to get truth from "{}" event_id {} (frame {})\n'
                        .format(fpath, event_id, frame_counter)
                    )
                    raise
                truths.append(event_truth)
                event['unique_id'] = truths[-1]['unique_id']

            events.append(event)

            for photon_name in photons:
                photons_d[photon_name].append(extract_photons(frame, photon_name))

            for pulse_series_name in pulses:
                pulses_list, time_range = extract_pulses(frame, pulse_series_name)
                pulses_d[pulse_series_name].append(pulses_list)
                tr_key = pulse_series_name + 'TimeRange'
                if time_range is not None:
                    if tr_key not in pulses_d:
                        pulses_d[tr_key] = []
                    pulses_d[tr_key].append(time_range)

            for reco_name in recos:
                recos_d[reco_name].append(extract_reco(frame, reco_name))

            for trigger_hierarchy_name in triggers:
                trigger_hierarchies[trigger_hierarchy_name].append(
                    extract_trigger_hierarchy(frame, trigger_hierarchy_name)
                )

            # Clear frame buffer and start a new "chain" with the next frame
            frame_buffer = [next_frame]
        except:
            sys.stderr.write('ERROR! file "{}", frame #{}\n'.format(fpath, frame_counter + 1))
            raise

    # Sanity check that a pulse series that has corresponding TimeRange field
    # exists in all frames, since this is populated by appending to a list (so
    # a single frame missing this will cause all others to be misaligned in the
    # resulting array)
    for pulse_series_name in pulses:
        tr_key = pulse_series_name + 'TimeRange'
        if tr_key not in pulses_d:
            continue
        if len(pulses_d[pulse_series_name]) != len(pulses_d[tr_key]):
            raise ValueError(
                "{} present in some frames but not present in other frames"
                .format(tr_key)
            )

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

    struct_dtype_spec = []
    for key in event.keys():
        struct_dtype_spec.append((key, event_dtypes.get(key, np.float32)))

    events = np.array([tuple(ev.values()) for ev in events], dtype=struct_dtype_spec)
    np.save(join(outdir, 'events.npy'), events)

    if truth:
        np.save(join(outdir, 'truth.npy'), np.array(truths))

    for name in photons:
        pickle.dump(photons_d[name],
                    open(join(photon_series_dir, name + '.pkl'), 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    for name in pulses:
        pickle.dump(
            pulses_d[name],
            open(join(pulse_series_dir, name + '.pkl'), 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )
        key = name + 'TimeRange'
        if key in pulses_d and pulses_d[key]:
            np.save(
                join(pulse_series_dir, key + '.npy'),
                np.array(pulses_d[key], dtype=np.float32)
            )

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
