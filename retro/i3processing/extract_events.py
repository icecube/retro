#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order

"""
Extract information on events from an i3 file needed for running Retro Reco.
"""

from __future__ import absolute_import, division, print_function

__author__ = "P. Eller, J.L. Lanfranchi"
__license__ = """Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = [
    "GENERIC_I3_FNAME_RE",
    "I3_FNAME_INFO_RE",
    "START_END_TIME_SPEC",
    "I3PARTICLE_SPECS",
    "MissingPhysicsFrameError",
    "extract_file_metadata",
    "get_frame_item",
    "extract_reco",
    "extract_trigger_hierarchy",
    "extract_pulses",
    "extract_photons",
    "get_cascade_and_track_info",
    "populate_track_t",
    "record_particles",
    "process_true_neutrino",
    "extract_truth",
    "extract_metadata_from_frame",
    "extract_events",
    "main",
]

from argparse import ArgumentParser
from collections import OrderedDict
try:
    from collections import Iterable, Sequence
except ImportError:
    from collections.abc import Iterable, Sequence
from copy import deepcopy
from os.path import abspath, basename, dirname, join
import pickle
import re
import sys

import numpy as np
from six import PY2, string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import (
    PHOTON_T,
    PULSE_T,
    TRIGGER_T,
    TRACK_T,
    CASCADE_T,
    I3TIME_T,
    FitStatus,
    InteractionType,
    LocationType,
    ParticleType,
    EM_CASCADE_PTYPES,
    HADR_CASCADE_PTYPES,
    TRACK_PTYPES,
    INVISIBLE_PTYPES,
    ELECTRONS,
    MUONS,
    TAUS,
    NUES,
    NUMUS,
    NUTAUS,
    NEUTRINOS,
    ParticleShape,
    ExtractionError,
    NO_TRACK,
    INVALID_TRACK,
    NO_CASCADE,
    INVALID_CASCADE,
)
from retro.utils.cascade_energy_conversion import em2hadr, hadr2em
from retro.utils.misc import (
    dict2struct, expand, get_file_md5, mkdir, set_explicit_dtype
)
from retro.utils.geom import cart2sph_np, sph2cart_np


GENERIC_I3_FNAME_RE = re.compile(
    r"""
    ^                 # Anchor to beginning of string
    (?P<base>.*)      # Any number of any character
    (?P<i3ext>\.i3)   # Must have ".i3" as extension
    (?P<compext>\..*) # Optional extension indicating compression
    $                 # End of string
    """,
    (re.VERBOSE | re.IGNORECASE),
)

I3_FNAME_INFO_RE = re.compile(
    r"""
    Level(?P<proc_level>.+) # processing level e.g. 5p or 5pt (???)
    _(?P<detector>[^.]+)    # detector, e.g. IC86
    \.(?P<year>\d+)         # year
    _(?P<generator>.+)      # generator, e.g. genie
    _(?P<flavor>.+)         # flavor, e.g. nue
    \.(?P<run>\d+)          # run, e.g. 012600
    \.(?P<filenum>\d+)      # file number, e.g. 000000
    """,
    (re.VERBOSE | re.IGNORECASE),
)

START_END_TIME_SPEC = OrderedDict(
    [
        (
            "start_time",
            dict(
                paths=("start_time.utc_year", "start_time.utc_daq_time"), dtype=I3TIME_T
            ),
        ),
        (
            "end_time",
            dict(paths=("end_time.utc_year", "end_time.utc_daq_time"), dtype=I3TIME_T),
        ),
    ]
)


I3EVENTHEADER_SPECS = OrderedDict(
    [
        ("run_id", dict(dtype=np.uint32)),
        ("sub_run_id", dict(dtype=np.uint32)),
        ("event_id", dict(dtype=np.uint32)),
        ("sub_event_id", dict(dtype=np.uint32)),
        ("sub_event_stream", dict(dtype=np.dtype("S20"))),
        ("state", dict(dtype=np.uint8)),
    ]
)
"""See: dataclasses/public/dataclasses/physics/I3EventHeader.h"""
I3EVENTHEADER_SPECS.update(START_END_TIME_SPEC)

I3PARTICLE_SPECS = OrderedDict(
    [
        ("major_id", dict(dtype=np.uint64, default=0)),
        ("minor_id", dict(dtype=np.int32, default=0)),
        ("x", dict(paths="pos.x", dtype=np.float64, default=np.nan)),
        ("y", dict(paths="pos.y", dtype=np.float64, default=np.nan)),
        ("z", dict(paths="pos.z", dtype=np.float64, default=np.nan)),
        ("time", dict(dtype=np.float64, default=np.nan)),
        ("zenith", dict(paths="dir.zenith", dtype=np.float64, default=np.nan)),
        (
            "coszen",
            dict(paths="dir.zenith", xform=np.cos, dtype=np.float64, default=np.nan),
        ),
        ("azimuth", dict(paths="dir.azimuth", dtype=np.float64, default=np.nan)),
        ("energy", dict(dtype=np.float64, default=np.nan)),
        ("speed", dict(dtype=np.float64, default=np.nan)),
        ("length", dict(dtype=np.float64, default=np.nan)),
        (
            "pdg_encoding",
            dict(enum=ParticleType, dtype=np.int32, default=ParticleType.unknown),
        ),
        ("type", dict(enum=ParticleType, dtype=np.int32, default=ParticleType.unknown)),
        ("shape", dict(enum=ParticleShape, dtype=np.uint8, default=ParticleShape.Null)),
        (
            "location_type",
            dict(enum=LocationType, dtype=np.uint8, default=LocationType.Anywhere),
        ),
        ("fit_status", dict(enum=FitStatus, dtype=np.int8, default=FitStatus.NotSet)),
    ]
)

MILLIPEDE_FIT_PARAMS_SPECS = OrderedDict(
    [
        ("logl", dict(dtype=np.float64, default=np.nan)),
        ("rlogl", dict(dtype=np.float64, default=np.nan)),
        ("ndof", dict(dtype=np.int32, default=-1)),
        ("nmini", dict(dtype=np.int32, default=-1)),
        ("qtotal", dict(dtype=np.float64, default=np.nan)),
        ("predicted_qtotal", dict(dtype=np.float64, default=np.nan)),
        ("squared_residuals", dict(dtype=np.float64, default=np.nan)),
        ("chi_squared", dict(dtype=np.float64, default=np.nan)),
        ("chi_squared_dof", dict(dtype=np.float64, default=np.nan)),
        # Note logl_ratio is not present in pEgLeg
    ]
)
"""See millipede/private/millipede/converter/MillipedeFitParamsConverter.cxx"""

HIT_MULTIPLICITY_SPECS = OrderedDict(
    [
        ("n_hit_doms", dict(dtype=np.int32, default=-1)),
        ("n_hit_doms_one_pulse", dict(dtype=np.int32, default=-1)),
        ("n_hit_strings", dict(dtype=np.int32, default=-1)),
        ("n_pulses", dict(dtype=np.int32, default=-1)),
    ]
)

HIT_STATISTICS_SPECS = OrderedDict(
    [
        ("cog_x", dict(path="cog.x", dtype=np.float32, default=-1)),
        ("cog_y", dict(path="cog.y", dtype=np.float32, default=-1)),
        ("cog_z", dict(path="cog.z", dtype=np.float32, default=-1)),
        ("cog_z_sigma", dict(dtype=np.float32, default=np.nan)),
        ("max_pulse_time", dict(dtype=np.float32, default=np.nan)),
        ("min_pulse_time", dict(dtype=np.float32, default=np.nan)),
        ("q_max_doms", dict(dtype=np.float32, default=np.nan)),
        ("q_tot_pulses", dict(dtype=np.float32, default=np.nan)),
        ("z_max", dict(dtype=np.float32, default=np.nan)),
        ("z_mean", dict(dtype=np.float32, default=np.nan)),
        ("z_min", dict(dtype=np.float32, default=np.nan)),
        ("z_sigma", dict(dtype=np.float32, default=np.nan)),
        ("z_travel", dict(dtype=np.float32, default=np.nan)),
    ]
)

TIME_WINDOW_SPECS = OrderedDict(
    [
        ("start", dict(dtype=np.float64)),
        ("stop", dict(dtype=np.float64)),
        ("length", dict(dtype=np.float64)),
    ]
)

CLAST_FIT_PARAMS_SPECS = OrderedDict(
    [
        ("mineval", dict(dtype=np.float64, default=np.nan)),
        ("evalratio", dict(dtype=np.float64, default=np.nan)),
        ("eval2", dict(dtype=np.float64, default=np.nan)),
        ("eval3", dict(dtype=np.float64, default=np.nan)),
    ]
)

FINITE_CUTS_SPECS = OrderedDict(
    [
        ("Length", dict(dtype=np.float64, default=np.nan)),
        ("Lend", dict(dtype=np.float64, default=np.nan)),
        ("Lstart", dict(dtype=np.float64, default=np.nan)),
        ("Sdet", dict(dtype=np.float64, default=np.nan)),
        ("finiteCut", dict(dtype=np.float64, default=np.nan)),
        ("DetectorLength", dict(dtype=np.float64, default=np.nan)),
    ]
)

START_STOP_PARAMS_SPECS = OrderedDict(
    [
        ("LLHStartingTrack", dict(dtype=np.float64, default=np.nan)),
        ("LLHStoppingTrack", dict(dtype=np.float64, default=np.nan)),
        ("LLHInfTrack", dict(dtype=np.float64, default=np.nan)),
    ]
)

DIPOLE_FIT_PARAMS_SPECS = OrderedDict(
    [
        ("Magnet", dict(dtype=np.float64, default=np.nan)),
        ("MagnetX", dict(dtype=np.float64, default=np.nan)),
        ("MagnetY", dict(dtype=np.float64, default=np.nan)),
        ("MagnetZ", dict(dtype=np.float64, default=np.nan)),
        ("AmpSum", dict(dtype=np.float64, default=np.nan)),
        ("NHits", dict(dtype=np.int32, default=-1)),
        ("NPairs", dict(dtype=np.int32, default=-1)),
        ("MaxAmp", dict(dtype=np.float64, default=np.nan)),
    ]
)

DST_PARAM_SPECS = OrderedDict(
    [
        ("cog_x", dict(path="cog.x", dtype=np.int8)),
        ("cog_y", dict(path="cog.y", dtype=np.int8)),
        ("cog_z", dict(path="cog.z", dtype=np.int8)),
        ("n_dir", dict(dtype=np.uint8)),
        ("n_string", dict(dtype=np.uint8)),
        ("ndom", dict(dtype=np.uint16)),
        ("reco_label", dict(dtype=np.uint8)),
        ("time", dict(dtype=np.uint32)),
        ("trigger_tag", dict(dtype=np.uint16)),
    ]
)


class MissingPhysicsFrameError(Exception):
    """Processing a frame buffer via the `process_frame_buffer` closure
    (function) defined within ::func::`extract_events` requires a physics
    frame."""

    pass


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
    finfo_match = next(I3_FNAME_INFO_RE.finditer(fname))
    if not finfo_match:
        raise ValueError('Could not interpret file path "%s"' % fname)

    finfo_dict = finfo_match.groupdict()

    file_info = OrderedDict(
        [
            ("detector", finfo_dict["detector"].lower()),
            ("year", int(finfo_dict["year"] or -1)),
            ("generator", finfo_dict["generator"].lower()),
            ("run", int(finfo_dict["run"] or -1)),
            ("filenum", int(finfo_dict["filenum"] or -1)),
            ("proc_level", finfo_dict["proc_level"].lower()),
        ]
    )

    return file_info


def auto_get_frame_item(frame, key):
    from icecube import dataclasses, icetray, recclasses

    type_specs = {
        recclasses.I3HitMultiplicityValues: HIT_MULTIPLICITY_SPECS,
        recclasses.I3HitStatisticsValues: HIT_STATISTICS_SPECS,
        dataclasses.I3TimeWindow: TIME_WINDOW_SPECS,
        dataclasses.I3Particle: I3PARTICLE_SPECS,
        recclasses.I3CLastFitParams: CLAST_FIT_PARAMS_SPECS,
        recclasses.I3FiniteCuts: FINITE_CUTS_SPECS,
        recclasses.I3DipoleFitParams: DIPOLE_FIT_PARAMS_SPECS,
        recclasses.I3StartStopParams: START_STOP_PARAMS_SPECS,
        recclasses.I3DST16: DST_PARAM_SPECS,
        icetray.I3Bool: np.bool8,
    }

    if key in ADVANCED_KEY_SPECS:
        pass
    elif isinstance(key, tuple(type_specs.keys())):
        pass


def get_frame_item(frame, key, specs, allow_missing):
    """
    Parameters
    ----------
    frame : icecube.icetray.I3Frame
    key : str
    specs : OrderedDict
    allow_missing : bool

    Returns
    -------
    out_d : OrderedDict

    """
    if key not in frame and not allow_missing:
        raise KeyError("'{}' not in frame".format(key))

    out_d = OrderedDict()
    for output_name, spec in specs.items():
        # Get dtype, otherwise infer one of "already defined", int64, float64,
        # or bool8
        dtype = spec.get("dtype", None)

        if key in frame:
            # If "paths" key present in `spec`, get its value(s); otherwise,
            # `paths` is just output_name
            paths = spec.get("paths", output_name)
            if isinstance(paths, string_types):
                paths = [paths]

            obj = frame[key]  # pylint: disable=unused-variable

            values = []
            for path in paths:
                assert isinstance(path, string_types)

                # Assume it's an attribute if the string does not explicitly
                # define attribute (".") or item ("[") access syntax
                if path[0] not in (".", "["):
                    path = "." + path
                value = eval("obj" + path)  # pylint: disable=eval-used
                values.append(value)
            values = tuple(values)

            # Get transform if present, otherwise set to None
            xform = spec.get("xform", None)

            if xform is None:
                if len(paths) == 1:
                    value = values[0]
                else:
                    value = values
            else:
                value = xform(*values)

        else:
            value = spec["default"]

        if dtype is None:
            value = set_explicit_dtype(value)
        else:
            if isinstance(value, Sequence):
                value = np.array(value, dtype=dtype)
            else:
                value = dtype(value)

        out_d[output_name] = value

    return out_d


def extract_reco(frame, reco):
    """Extract a reconstruction from a frame."""
    reco_dict = None

    if reco.startswith("Pegleg_Fit"):
        from icecube import millipede  # pylint: disable=unused-import

        reco_dict = OrderedDict()
        reco_dict["fit_params"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "FitParams",
                specs=MILLIPEDE_FIT_PARAMS_SPECS,
                allow_missing=True,
            )
        )
        reco_dict["PLMN_timing"] = np.float64(frame["PLMN_timing"].value)
        reco_dict["MO_timing"] = np.float64(frame["MO_timing"].value)
        reco_dict["Neutrino"] = dict2struct(
            get_frame_item(
                frame=frame, key=reco, specs=I3PARTICLE_SPECS, allow_missing=True
            )
        )
        reco_dict["EMCasc"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "EMCasc",
                specs=I3PARTICLE_SPECS,
                allow_missing=True,
            )
        )
        reco_dict["HDCasc"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "HDCasc",
                specs=I3PARTICLE_SPECS,
                allow_missing=True,
            )
        )
        reco_dict["Track"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "Track",
                specs=I3PARTICLE_SPECS,
                allow_missing=True,
            )
        )

    elif reco.startswith("Monopod_"):
        from icecube import millipede  # pylint: disable=unused-import

        reco_dict = OrderedDict()
        reco_dict["fit_params"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "FitParams",
                specs=MILLIPEDE_FIT_PARAMS_SPECS,
                allow_missing=True,
            )
        )
        reco_dict["Neutrino"] = dict2struct(
            get_frame_item(
                frame=frame, key=reco, specs=I3PARTICLE_SPECS, allow_missing=True
            )
        )

    # -- HybridReco, as seen in DRAGON 1{2,4,6}60 Monte Carlo -- #

    # MultiNest7D is a cascade-only fit to the event
    elif reco.endswith("MultiNest7D"):
        reco_dict = OrderedDict()
        reco_dict["Neutrino"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "_Neutrino",
                specs=I3PARTICLE_SPECS,
                allow_missing=False,
            )
        )
        reco_dict["Cascade"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "_Cascade",
                specs=I3PARTICLE_SPECS,
                allow_missing=False,
            )
        )

    # MultiNest8D fits a cascade & track, with casscade in the track direction
    elif reco.endswith("MultiNest8D") or reco.endswith("MultiNest10D"):
        if reco + "_Neutrino" in frame:
            nu_key = reco + "_Neutrino"
        elif reco + "_NumuCC" in frame:
            nu_key = reco + "_NumuCC"

        if reco + "_Cascade" in frame:
            casc_key = reco + "_Cascade"
        elif reco + "_HDCasc" in frame:
            casc_key = reco + "_HDCasc"
        else:
            raise ValueError("Cannot find cascade in frame")

        reco_dict = OrderedDict()
        reco_dict["Neutrino"] = dict2struct(
            get_frame_item(
                frame=frame, key=nu_key, specs=I3PARTICLE_SPECS, allow_missing=False
            )
        )
        reco_dict["Cascade"] = dict2struct(
            get_frame_item(
                frame=frame, key=casc_key, specs=I3PARTICLE_SPECS, allow_missing=False
            )
        )
        reco_dict["Track"] = dict2struct(
            get_frame_item(
                frame=frame,
                key=reco + "_Track",
                specs=I3PARTICLE_SPECS,
                allow_missing=False,
            )
        )

    # -- Retro recos -- #

    elif reco.startswith("retro_"):
        from icecube.dataclasses import I3MapStringDouble, I3Particle
        reco_dict = OrderedDict()
        for key in sorted(frame.keys()):
            if not key.startswith(reco):
                continue

            value = frame[key]

            if isinstance(value, I3Particle):
                continue

            if isinstance(value, I3MapStringDouble):
                value = dict2struct(value)
            else:
                value = set_explicit_dtype(value)

            reco_dict[key[len(reco) + 2 ::]] = value

    # -- Anything else assume it's a single I3Particle -- #

    else:
        reco_dict = get_frame_item(
            frame=frame, key=reco, specs=I3PARTICLE_SPECS, allow_missing=True
        )

    # TODO: why is PID here?
    # -- (subset of) PID and cut vars -- #

    if reco.startswith("IC86_Dunkman_L6") and "IC86_Dunkman_L6" in frame:
        cutvars = frame["IC86_Dunkman_L6"]
        for var in ["delta_LLH", "mn_start_contained", "mn_stop_contained"]:
            try:
                reco_dict[var] = getattr(cutvars, var)
            except AttributeError:
                pass

    reco = dict2struct(reco_dict)
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
    from icecube import (  # pylint: disable=unused-import
        dataclasses,
        recclasses,
        simclasses,
    )

    trigger_hierarchy = frame[path]
    triggers = []
    iterattr = getattr(
        trigger_hierarchy,
        "items" if "items" in dir(trigger_hierarchy) else "iteritems"
    )
    for _, trigger in iterattr():
        config_id = trigger.key.config_id or 0
        triggers.append(
            (
                int(trigger.key.type),
                int(trigger.key.subtype),
                int(trigger.key.source),
                config_id,
                trigger.fired,
                trigger.time,
                trigger.length,
            )
        )
    try:
        triggers = np.array(triggers, dtype=TRIGGER_T)
    except TypeError:
        sys.stderr.write("triggers: {}\n".format(triggers))
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
    from icecube import (  # pylint: disable=unused-import
        dataclasses,
        recclasses,
        simclasses,
    )
    from icecube.dataclasses import (  # pylint: disable=no-name-in-module
        I3RecoPulseSeriesMap,
        I3RecoPulseSeriesMapMask,
        I3RecoPulseSeriesMapUnion,
    )

    pulse_series = frame[pulse_series_name]

    time_range_name = pulse_series_name + "TimeRange"

    if time_range_name in frame:
        i3_time_range = frame[time_range_name]
        time_range = i3_time_range.start, i3_time_range.stop
    else:
        time_range = np.nan, np.nan

    if isinstance(pulse_series, (I3RecoPulseSeriesMapMask, I3RecoPulseSeriesMapUnion)):
        pulse_series = pulse_series.apply(frame)

    if not isinstance(pulse_series, I3RecoPulseSeriesMap):
        raise TypeError(type(pulse_series))

    pulses_list = []

    for (string, dom, pmt), pinfo in pulse_series:
        pls = []
        for pulse in pinfo:
            pls.append((pulse.time, pulse.charge, pulse.width, pulse.flags))
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
    from icecube import (  # pylint: disable=unused-import
        dataclasses,
        recclasses,
        simclasses,
    )

    photon_series = frame[photon_key]
    photons = []
    for omkey, pinfos in photon_series:
        if len(omkey) == 2:
            string, dom = omkey
            pmt = 0
        else:
            string, dom, pmt = omkey

        phot = []  # Photon(*([] for _ in range(len(Photon._fields))))
        for pinfo in pinfos:
            phot.append(
                (
                    pinfo.time,
                    pinfo.pos.x,
                    pinfo.pos.y,
                    pinfo.pos.z,
                    np.cos(pinfo.dir.zenith),
                    pinfo.dir.azimuth,
                    pinfo.wavelength,
                )
            )
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
            # raise ValueError("{} is not track or cascade".format(ptype))
            sys.stderr.write("{!r} is neither track nor cascade\n".format(ptype))

    if tracks:
        longest_track_particle = None
        total_track = np.zeros(shape=1, dtype=TRACK_T)
        wtd_dir = np.zeros(3)  # direction cosines (x, y, & z)
        sum_of_lengths = 0.0
        secondaries = []
        for particle in tracks:
            if particle.length >= total_track["length"]:
                total_track["length"] = particle.length
                longest_track_particle = particle
            total_track["energy"] += particle.energy
            # Note negative sign is due to dir.x, dir.y, dir.z indicating
            # direction _toward which particle points_, while icecube zenith
            # and azimuth describe direction _from which particle came_.
            wtd_dir -= particle.length * np.array(
                (particle.dir.x, particle.dir.y, particle.dir.z)
            )
            sum_of_lengths += particle.length
            secondaries.extend(mctree.get_daughters(particle))

        info = get_cascade_and_track_info(particles=secondaries, mctree=mctree)

        wtd_dir /= sum_of_lengths
        directionality, zenith, azimuth = cart2sph_np(
            x=wtd_dir[0], y=wtd_dir[1], z=wtd_dir[2]
        )

        longest_track = populate_track_t(mctree=mctree, particle=longest_track_particle)

        total_track["time"] = tracks[0]["time"]
        total_track["x"] = tracks[0]["x"]
        total_track["y"] = tracks[0]["y"]
        total_track["z"] = tracks[0]["z"]
        total_track["zenith"] = zenith
        total_track["coszen"] = np.cos(zenith)
        total_track["azimuth"] = azimuth
        total_track["directionality"] = directionality
        # (energy and length already set inside above loop)
        total_track["stochastic_loss"] = info["total_cascade"]["energy"]
        total_track["vis_em_equiv_stochastic_loss"] = info["vis_em_equiv_cascade"][
            "energy"
        ]
        if len(tracks) == 1:
            total_track["pdg_encoding"] = tracks[0]["pdg_encoding"]
        else:
            total_track["pdg_encoding"] = ParticleType.unknown

    else:
        total_track = deepcopy(NO_TRACK)
        longest_track = deepcopy(NO_TRACK)

    if cascades:
        total_cascade = np.zeros(shape=1, dtype=CASCADE_T)
        vis_em_equiv_cascade = np.zeros(shape=1, dtype=CASCADE_T)
        vis_em_wtd_dir = np.zeros(3)  # direction cosines (x, y, & z)
        energy_wtd_dir = np.zeros(3)  # direction cosines (x, y, & z)
        for particle, is_hadr in cascades:
            total_cascade["energy"] += particle.energy
            raw_dir = np.array((particle.dir.x, particle.dir.y, particle.dir.z))
            energy_wtd_dir -= particle.energy * raw_dir

            if particle in INVISIBLE_PTYPES:
                continue

            if is_hadr:
                try:
                    vis_em_equiv_energy = hadr2em(particle.energy)
                except:
                    sys.stderr.write(
                        "pdg={}, energy={}\n".format(
                            particle.pdg_encoding, particle.energy
                        )
                    )
                    raise
            else:
                vis_em_equiv_energy = particle.energy
            vis_em_wtd_dir -= vis_em_equiv_energy * raw_dir
            vis_em_equiv_cascade["energy"] += vis_em_equiv_energy
    else:
        total_cascade = deepcopy(NO_CASCADE)
        vis_em_equiv_cascade = deepcopy(NO_CASCADE)

    cascade_and_track_info = OrderedDict(
        [
            ("total_track", total_track),
            ("longest_track", longest_track),
            ("total_cascade", total_cascade),
            ("vis_em_equiv_cascade", vis_em_equiv_cascade),
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
    track["time"] = particle.time
    track["x"] = particle.pos.x
    track["y"] = particle.pos.y
    track["z"] = particle.pos.z
    track["zenith"] = particle.dir.zenith
    track["coszen"] = np.cos(particle.dir.zenith)
    track["azimuth"] = particle.dir.azimuth
    track["directionality"] = 1
    track["energy"] = particle.energy
    track["length"] = particle.length
    track["pdg_encoding"] = particle.pdg_encoding

    # Get info about stochastics recorded as daughters of the track particle
    info = get_cascade_and_track_info(
        particles=mctree.get_daughters(particle), mctree=mctree
    )
    track["stochastic_loss"] = info["total_cascade"]["energy"]
    track["vis_em_equiv_stochastic_loss"] = info["vis_em_equiv_cascade"]["energy"]

    return track


def record_particles(particles):
    """
    Parameters
    ----------
    particles : OrderedDict

    Returns
    -------
    values_dict : OrderedDict
    dtypes_dict : OrderedDict

    """
    values_dict = OrderedDict()
    dtypes_dict = OrderedDict()

    for particle_name, particle in particles.items():
        dtype = particle.dtype
        # note `fields` attr is un-ordered, while `names` IS ordered
        fields = dtype.fields
        for field_name in dtype.names:
            key = "{}_{}".format(particle_name, field_name)
            values_dict[key] = particle[field_name]
            # `fields` contains dtype and byte offset; just want dtype
            dtypes_dict[key] = fields[field_name][0]

    return values_dict, dtypes_dict


def process_true_neutrino(nu, mctree, frame, event_truth):
    """Get particles to record "truth" information from an MC neutrino.

    Parameters
    ----------
    nu : icecube.dataclasses.I3Particle
    mctree : icecube.dataclasses.I3MCTree
    frame : icecube.icetray.I3Frame
    event_truth : OrderedDict

    Returns
    -------
    particles_to_record : OrderedDict

    """
    from icecube.dataclasses import I3Direction  # pylint: disable=no-name-in-module

    # By default, track and cascades all "zero"; convention is that
    # cascade0 is on lepton side of interaction
    track = deepcopy(NO_TRACK)
    cascade0 = deepcopy(NO_CASCADE)
    cascade1 = deepcopy(NO_CASCADE)
    total_cascade = deepcopy(NO_CASCADE)

    nu_pdg = nu.pdg_encoding
    secondaries = mctree.get_daughters(nu)
    interaction_type = InteractionType(int(event_truth["InteractionType"]))

    # neutrino 4-momentum
    nu_p4 = nu.energy * np.array([1, nu.dir.x, nu.dir.y, nu.dir.z])

    if interaction_type == InteractionType.CC:
        # Find the charged lepton generated in the interaction
        charged_lepton = None
        for secondary in secondaries:
            # charged lepton PDG one less than corresponding neutrino's PDG
            if (
                abs(int(secondary.pdg_encoding)) == abs(nu_pdg) - 1
                and secondary.time == nu.time
                and secondary.pos == nu.pos
            ):
                if charged_lepton is None:
                    charged_lepton = secondary
                elif secondary.energy > charged_lepton.energy:
                    charged_lepton = secondary

        if charged_lepton is None:
            msg = "ERROR: Couldn't find charged lepton daughter in CC MCTree"
            event_truth[
                "extraction_error"
            ] = ExtractionError.NU_CC_LEPTON_SECONDARY_MISSING
            sys.stderr.write(msg + "\n")
            # raise ValueError(msg)
            if nu_pdg in NUES + NUTAUS:
                cascade0 = INVALID_CASCADE
            elif nu_pdg in NUMUS:
                track = INVALID_TRACK
            else:
                raise ValueError()

        else:
            charged_lepton_pdg = ParticleType(int(charged_lepton.pdg_encoding))

            # 4-momentum; note I3Particle.energy is particle's kinetic energy
            # and I3Particle.dir.{x, y, z} point in direction of particle's
            # travel while I3Particle.dir.{zenith, azimuth} point oppositely
            # to particle's direction of travel
            charged_lepton_p4 = np.array(
                [
                    charged_lepton.energy + charged_lepton.mass,
                    charged_lepton.dir.x * charged_lepton.energy,
                    charged_lepton.dir.y * charged_lepton.energy,
                    charged_lepton.dir.z * charged_lepton.energy,
                ]
            )
            remaining_p4 = nu_p4 - charged_lepton_p4
            remaining_energy = remaining_p4[0]
            remaining_dir = I3Direction(*remaining_p4[1:])

            # All these fields are the same for all CC events
            for cascade in (cascade0, cascade1, total_cascade):
                cascade["time"] = charged_lepton.time
                cascade["x"] = charged_lepton.pos.x
                cascade["y"] = charged_lepton.pos.y
                cascade["z"] = charged_lepton.pos.z

            if charged_lepton_pdg in ELECTRONS:
                cascade0["pdg_encoding"] = charged_lepton_pdg
                cascade0["zenith"] = charged_lepton.dir.zenith
                cascade0["coszen"] = np.cos(charged_lepton.dir.zenith)
                cascade0["azimuth"] = charged_lepton.dir.azimuth
                cascade0["directionality"] = 1
                cascade0["energy"] = charged_lepton.energy
                cascade0["hadr_fraction"] = 0
                cascade0["em_equiv_energy"] = charged_lepton.energy
                cascade0["hadr_equiv_energy"] = em2hadr(charged_lepton.energy)

                cascade1["pdg_encoding"] = ParticleType.unknown
                cascade1["zenith"] = remaining_dir.zenith
                cascade1["coszen"] = np.cos(remaining_dir.zenith)
                cascade1["azimuth"] = remaining_dir.azimuth
                cascade1["directionality"] = 1
                cascade1["energy"] = remaining_energy
                cascade1["hadr_fraction"] = 1
                cascade1["em_equiv_energy"] = hadr2em(remaining_energy)
                cascade1["hadr_equiv_energy"] = remaining_energy

            elif charged_lepton_pdg in MUONS:
                track = populate_track_t(mctree=mctree, particle=charged_lepton)

                cascade1["pdg_encoding"] = ParticleType.unknown
                cascade1["zenith"] = remaining_dir.zenith
                cascade1["coszen"] = np.cos(remaining_dir.zenith)
                cascade1["azimuth"] = remaining_dir.azimuth
                cascade1["directionality"] = np.nan
                cascade1["energy"] = remaining_energy
                cascade1["hadr_fraction"] = 1
                cascade1["em_equiv_energy"] = hadr2em(remaining_energy)
                cascade1["hadr_equiv_energy"] = remaining_energy

            elif charged_lepton_pdg in TAUS:
                # Until we can see which taus have muon decay product, this is
                # as good as we can do (i.e. keep track as NO_TRACK, assume all
                # tau's energy in tau-based hadronic cascade and remaining
                # energy in a separate hadronic cascade)
                cascade0["pdg_encoding"] = charged_lepton_pdg
                cascade0["zenith"] = charged_lepton.dir.zenith
                cascade0["coszen"] = np.cos(charged_lepton.dir.zenith)
                cascade0["azimuth"] = charged_lepton.dir.azimuth
                cascade0["directionality"] = 1
                cascade0["energy"] = charged_lepton.mass + charged_lepton.energy
                cascade0["hadr_fraction"] = 1
                cascade0["em_equiv_energy"] = hadr2em(charged_lepton.energy)
                cascade0["hadr_equiv_energy"] = charged_lepton.energy

                cascade1["pdg_encoding"] = ParticleType.unknown
                cascade1["zenith"] = remaining_dir.zenith
                cascade1["coszen"] = np.cos(remaining_dir.zenith)
                cascade1["azimuth"] = remaining_dir.azimuth
                cascade1["directionality"] = None
                cascade1["energy"] = remaining_energy
                cascade1["hadr_fraction"] = 1
                cascade1["em_equiv_energy"] = hadr2em(remaining_energy)
                cascade1["hadr_equiv_energy"] = remaining_energy

            else:
                raise ValueError(
                    "unrecognized PDG code : {}".format(charged_lepton_pdg)
                )

    elif interaction_type == InteractionType.NC:
        outgoing_nu = None
        for secondary in secondaries:
            if (
                int(secondary.pdg_encoding) == nu_pdg
                and secondary.time == nu.time
                and secondary.pos == nu.pos
            ):
                if outgoing_nu is None:
                    outgoing_nu = secondary
                elif secondary.energy > outgoing_nu.energy:
                    outgoing_nu = secondary
        if outgoing_nu is None:
            msg = "ERROR: Couldn't find outgoing neutrino in NC MCTree"
            event_truth["extraction_error"] = ExtractionError.NU_NC_OUTOING_NU_MISSING
            sys.stderr.write(msg + "\n")
            cascade1 = INVALID_CASCADE
            # raise ValueError(msg)

        else:
            # No track and no cascade from lepton (escaping neutrino is
            # invisible); energy not carried off by neutrino is dumped into a
            # hadronic cascade

            # 4-momentum
            outgoing_nu_p4 = outgoing_nu.energy * np.array(
                [1, outgoing_nu.dir.x, outgoing_nu.dir.y, outgoing_nu.dir.z]
            )
            remaining_p4 = nu_p4 - outgoing_nu_p4
            remaining_energy = remaining_p4[0]
            remaining_dir = I3Direction(*remaining_p4[1:])

            cascade1["pdg_encoding"] = ParticleType.unknown
            cascade1["time"] = outgoing_nu.time
            cascade1["x"] = outgoing_nu.pos.x
            cascade1["y"] = outgoing_nu.pos.y
            cascade1["z"] = outgoing_nu.pos.z
            cascade1["zenith"] = remaining_dir.zenith
            cascade1["coszen"] = np.cos(remaining_dir.zenith)
            cascade1["azimuth"] = remaining_dir.azimuth
            cascade1["directionality"] = 1
            cascade1["energy"] = remaining_energy
            cascade1["hadr_fraction"] = 1
            cascade1["em_equiv_energy"] = hadr2em(remaining_energy)
            cascade1["hadr_equiv_energy"] = remaining_energy

    else:  # interaction_type == InteractionType.undefined
        from icecube import genie_icetray  # pylint: disable=unused-import

        grd = frame["I3GENIEResultDict"]
        if not grd["nuel"]:  # nuel=True means elastic collision, apparently
            grd_s = "{{{}}}".format(
                ", ".join("'{}': {!r}".format(k, grd[k]) for k in sorted(grd.keys()))
            )
            raise ValueError(
                "Not recognized as NC, CC, or elastic. I3GENIEResultDict:\n{}".format(
                    grd_s
                )
            )
        if len(secondaries) > 1:
            raise NotImplementedError(
                "more than one secondary reported in elastic collision"
            )
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

            cascade1["pdg_encoding"] = int(secondary_pdg)
            cascade1["time"] = secondary.time
            cascade1["x"] = secondary.pos.x
            cascade1["y"] = secondary.pos.y
            cascade1["z"] = secondary.pos.z
            cascade1["zenith"] = secondary.dir.zenith
            cascade1["coszen"] = np.cos(secondary.dir.zenith)
            cascade1["azimuth"] = secondary.dir.azimuth
            cascade1["directionality"] = 1
            cascade1["energy"] = secondary_energy
            cascade1["hadr_fraction"] = 0 if is_em else 1
            cascade1["em_equiv_energy"] = (
                secondary_energy if is_em else hadr2em(secondary_energy)
            )
            cascade1["hadr_equiv_energy"] = (
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
        if cascade["energy"] == 0:
            continue
        num_cascades += 1

        if total_cascade_pdg is None:
            total_cascade_pdg = cascade["pdg_encoding"]
        elif cascade["pdg_encoding"] != total_cascade_pdg:
            total_cascade_pdg = ParticleType.unknown

        total_energy += cascade["energy"]
        total_hadr_fraction += cascade["hadr_fraction"]
        em_equiv_energy = cascade["em_equiv_energy"]
        total_em_equiv_energy += em_equiv_energy
        total_hadr_equiv_energy += cascade["hadr_equiv_energy"]

        # note that (theta, phi) points oppositely to direction defined by
        # (zenith, azimuth)
        neg_dirvec = sph2cart_np(
            r=em_equiv_energy, theta=cascade["zenith"], phi=cascade["azimuth"]
        )
        em_equiv_energy_weighted_dirvec -= np.concatenate(neg_dirvec)

    total_hadr_fraction /= num_cascades
    em_equiv_energy_weighted_dirvec /= total_em_equiv_energy

    directionality, zenith, azimuth = cart2sph_np(
        x=-em_equiv_energy_weighted_dirvec[0],
        y=-em_equiv_energy_weighted_dirvec[1],
        z=-em_equiv_energy_weighted_dirvec[2],
    )

    total_cascade["pdg_encoding"] = total_cascade_pdg
    total_cascade["time"] = nu.time
    total_cascade["x"] = nu.pos.x
    total_cascade["y"] = nu.pos.y
    total_cascade["z"] = nu.pos.z
    total_cascade["zenith"] = zenith
    total_cascade["coszen"] = np.cos(zenith)
    total_cascade["azimuth"] = azimuth
    total_cascade["directionality"] = directionality
    total_cascade["energy"] = total_energy
    total_cascade["hadr_fraction"] = total_hadr_fraction
    total_cascade["em_equiv_energy"] = total_em_equiv_energy
    total_cascade["hadr_equiv_energy"] = total_hadr_equiv_energy

    particles_to_record = OrderedDict(
        [
            ("track", track),
            ("cascade0", cascade0),
            ("cascade1", cascade1),
            ("total_cascade", total_cascade),
        ]
    )

    return particles_to_record


def extract_truth(frame):
    """Get event truth information from a frame.

    Parameters
    ----------
    frame : I3Frame

    Returns
    -------
    event_truth : OrderedDict

    """
    from icecube import dataclasses, icetray  # pylint: disable=unused-import

    try:
        from icecube import multinest_icetray  # pylint: disable=unused-import
    except ImportError:
        multinest_icetray = None

    # Anything not listed defaults to float32; note this is augmented when
    # cascades and tracks are added
    truth_dtypes = dict(
        pdg_encoding=np.int32,
        highest_energy_daughter_pdg=np.int32,
        longest_daughter_pdg=np.int32,
        InteractionType=np.int8,
        TargetPDGCode=np.int32,
        extraction_error=np.uint8,
    )

    event_truth = OrderedDict()

    # Extract info from I3MCTree: ...
    mctree = frame["I3MCTree"]

    # ... primary particle, if there is one ("noise" sim has none)
    primary_pdg = None
    if mctree.primaries:
        primary = mctree.primaries[0]
        primary_pdg = primary.pdg_encoding

        event_truth["pdg_encoding"] = primary_pdg
        event_truth["time"] = primary.time
        event_truth["x"] = primary.pos.x
        event_truth["y"] = primary.pos.y
        event_truth["z"] = primary.pos.z
        event_truth["energy"] = primary.energy
        event_truth["zenith"] = primary.dir.zenith
        event_truth["coszen"] = np.cos(primary.dir.zenith)
        event_truth["azimuth"] = primary.dir.azimuth
        event_truth["extraction_error"] = ExtractionError.NO_ERROR

    # TODO: should we prefix I3MCWeightDict items to avoid overwriting
    # something else?

    # Extract per-event info from I3MCWeightDict
    i3_mcwd = frame["I3MCWeightDict"]
    for key in sorted(i3_mcwd.keys()):
        if key in event_truth:
            raise ValueError("key '{}' already in event_truth".format(key))
        event_truth[key] = i3_mcwd[key]

    particles_to_record = None
    if primary_pdg in NEUTRINOS:
        particles_to_record = process_true_neutrino(
            nu=primary, mctree=mctree, frame=frame, event_truth=event_truth
        )

    elif primary_pdg == ParticleType.unknown:  # This (can) indicate a muon
        # TODO: how to handle muon bundles?

        secondaries = mctree.get_daughters(primary)
        muon = None
        if len(secondaries) == 1:
            secondary_pdg = secondaries[0].pdg_encoding
            if secondary_pdg in MUONS:
                muon = secondaries[0]
            else:
                raise NotImplementedError(
                    "primary of type {} and secondary of type {} not"
                    "implemented".format(
                        ParticleType.unknown, ParticleType(secondary_pdg)
                    )
                )
        else:
            raise NotImplementedError(
                "Unknown primary with multiple secondaries not implemented"
            )

        # If we get here, we have a single muon

        event_truth["pdg_encoding"] = secondary_pdg
        track = populate_track_t(mctree=mctree, particle=muon)
        particles_to_record = OrderedDict([("track", track)])

    elif primary_pdg is not None:  # is not neutrino, not noise (None):
        raise NotImplementedError("Only neutrino primaries are implemented")

    if particles_to_record:
        values_dict, dtypes_dict = record_particles(particles_to_record)
        event_truth.update(values_dict)
        truth_dtypes.update(dtypes_dict)

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
    event_header = frame["I3EventHeader"]
    event_meta["run_id"] = event_header.run_id
    event_meta["event_id"] = event_header.event_id
    return event_meta


def _extract_events_from_single_file(
    i3_fpath,
    retro_gcd_dir,
    gcd_md5_hex=None,
    gcd=None,
    outdir=None,
    photons=tuple(),
    pulses=tuple(),
    recos=tuple(),
    triggers=tuple(),
    truth=False,
    additional_keys=None,
):
    """Extract event information from an i3 file.

    Parameters
    ----------
    fpath : str
        Path to I3 file

    external_gcd : bool, optional
        If `True`, force loading a (single) GCD file to be found in the same
        directory as the data file(s) (e.g., use this option for actual data
        run files). If `None` (the default), a single GCD file in the same
        directory is automatically loaded but if no file is found, no error
        results. If `False`, then no GCD file in the same directory is loaded,
        even if one is present. If `True` or `None` and multiple GCD files are
        found, a ValueError is raised.

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

    additional_keys : str, iterable thereof, or None

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
                'truth': [{'pdg_encoding': ..., 'daughters': [{...}, ...]}],
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
    from icecube import (  # pylint: disable=unused-import
        dataclasses,
        recclasses,
        simclasses,
    )
    from icecube.dataio import I3FrameSequence  # pylint: disable=no-name-in-module
    from icecube.icetray import I3Frame  # pylint: disable=no-name-in-module
    from retro.i3processing.extract_gcd import MD5_HEX_RE, extract_gcd_frames

    i3_fpath = expand(i3_fpath)
    if gcd is not None:
        gcd = expand(gcd)

    if additional_keys is None:
        additional_keys = tuple()
    elif isinstance(additional_keys, string_types):
        additional_keys = (additional_keys,)
    else:
        assert isinstance(additional_keys, Iterable)
        additional_keys = tuple(additional_keys)

    # Subdirectory will be named same as i3 file (with ".i3" and any
    # compression extensions like .bz2, .zst, etc. removed)
    fname_parts = GENERIC_I3_FNAME_RE.match(basename(i3_fpath)).groupdict()
    leafdir_basename = fname_parts["base"]
    if outdir is None:
        # Default to dir same root directory as I3 file
        outdir = join(dirname(i3_fpath), leafdir_basename)
    else:
        outdir = join(outdir, leafdir_basename)

    i3file_md5_hex = get_file_md5(i3_fpath)
    i3_fpaths = [i3_fpath]
    if gcd is not None:
        i3_fpaths = [gcd] + i3_fpaths
    i3file_iterator = I3FrameSequence(i3_fpaths)

    events = []
    truths = []

    photons_d = OrderedDict()
    for name in photons:
        photons_d[name] = []

    pulses_d = OrderedDict()
    for name in pulses:
        pulses_d[name] = []
        pulses_d[name + "TimeRange"] = []

    recos_d = OrderedDict()
    for name in recos:
        recos_d[name] = []

    trigger_hierarchies = OrderedDict()
    for name in triggers:
        trigger_hierarchies[name] = []

    def process_frame_buffer(frame_buffer, gcd_md5_hex):
        """Get event information from a set of frames that, together, should
        completely describe a single event.

        Information gathered about the event is added to the (pre-existing)
        data structures listed in the "Out" section below.

        Parameters
        ----------
        frame_buffer : list
        gcd_md5_hex : length-32 string

        Out
        ---
        events : list
        photons_d : OrderedDict
        pulses_d : OrderedDict
        recos_d : OrderedDict
        trigger_hierarchies : OrderedDict

        """
        gcd_md5_hex = gcd_md5_hex.strip().lower()
        assert MD5_HEX_RE.match(gcd_md5_hex)
        num_qframes = 0
        for frame in frame_buffer:
            if frame.Stop == I3Frame.DAQ:
                num_qframes += 1
        if num_qframes > 1:
            raise ValueError(
                "Found {} DAQ (Q) frames in chain, must have only one".format(
                    num_qframes
                )
            )

        pframe = None
        for frame in frame_buffer[::-1]:
            if frame.Stop == I3Frame.DAQ:
                raise ValueError("DAQ (Q) frame found after last physics (P) frame")
            if frame.Stop == I3Frame.Physics:
                pframe = frame
                break
        if pframe is None:
            raise MissingPhysicsFrameError()
        if num_qframes == 0:
            raise ValueError("Found a physics (P) frame but no DAQ (Q) frame")

        event = get_frame_item(
            frame=frame,
            key="I3EventHeader",
            specs=I3EVENTHEADER_SPECS,
            allow_missing=False,
        )
        event["gcd_md5_hex"] = gcd_md5_hex
        if len(events) > 2 ** 32 - 1:
            raise ValueError(
                "only using uint32 to store event index, but have event index of {}".format(
                    len(events)
                )
            )
        event["index"] = np.uint32(len(events))

        if "TimeShift" in pframe:
            event["TimeShift"] = pframe["TimeShift"].value

        if truth:
            try:
                event_truth = extract_truth(pframe)
            except:
                sys.stderr.write("Failed to get truth from frame buffer\n")
                raise
            truths.append(event_truth)

        for frame_key in additional_keys:
            event[frame_key] = pframe[frame_key].value

        events.append(event)

        for photon_name in photons:
            photons_d[photon_name].append(extract_photons(pframe, photon_name))

        for pulse_series_name in pulses:
            pulses_list, time_range = extract_pulses(pframe, pulse_series_name)
            pulses_d[pulse_series_name].append(pulses_list)
            pulses_d[pulse_series_name + "TimeRange"].append(time_range)

        for reco_name in recos:
            recos_d[reco_name].append(extract_reco(pframe, reco_name))

        for trigger_hierarchy_name in triggers:
            trigger_hierarchies[trigger_hierarchy_name].append(
                extract_trigger_hierarchy(pframe, trigger_hierarchy_name)
            )

    mkdir(outdir)

    with open(join(outdir, "sourcefile_md5sum.txt"), "w") as md5sum_file:
        md5sum_file.write("{}  {}\n".format(i3file_md5_hex, i3_fpath))

    frame_buffer = []
    frame_counter = 0

    gcd_frames = OrderedDict([("g_frame", None), ("c_frame", None), ("d_frame", None)])
    gcd_changed = False

    while i3file_iterator.more():
        try:
            frame = None
            try:
                frame = i3file_iterator.pop_frame()
            except:
                sys.stderr.write("Failed to pop frame #{}\n".format(frame_counter + 1))
                raise
            frame_counter += 1

            if frame is None:
                break

            if frame.Stop in (I3Frame.Physics, I3Frame.DAQ):
                if gcd_changed:
                    gcd_md5_hex = extract_gcd_frames(retro_gcd_dir=retro_gcd_dir, **gcd_frames)
                    gcd_changed = False
                if frame.Stop == I3Frame.Physics:
                    frame_buffer.append(frame)
                    process_frame_buffer(frame_buffer, gcd_md5_hex=gcd_md5_hex)
                    frame_buffer.pop()
                elif frame.Stop == I3Frame.DAQ:
                    frame_buffer = [frame]
            elif frame.Stop == I3Frame.Geometry:
                gcd_frames["g_frame"] = frame
                gcd_changed = True
            elif frame.Stop == I3Frame.Calibration:
                gcd_frames["c_frame"] = frame
                gcd_changed = True
            elif frame.Stop == I3Frame.DetectorStatus:
                gcd_frames["d_frame"] = frame
                gcd_changed = True

        except Exception as err:
            sys.stderr.write(
                'ERROR! file "{}", frame #{}, error: {}\n'.format(
                    i3_fpath, frame_counter + 1, err
                )
            )
            raise

    # Sanity check that a pulse series that has corresponding TimeRange field
    # exists in all frames, since this is populated by appending to a list (so
    # a single frame missing this will cause all others to be misaligned in the
    # resulting array)
    for pulse_series_name in pulses:
        tr_key = pulse_series_name + "TimeRange"
        if tr_key in pulses_d and len(pulses_d[pulse_series_name]) != len(
            pulses_d[tr_key]
        ):
            raise ValueError(
                "{} present in some frames but not present in other frames".format(
                    tr_key
                )
            )

    if not events:
        assert not truths
        assert not photons
        for key, val in pulses_d.items():
            assert not val, "'{}': {}".format(name, val)
        for key, val in recos_d.items():
            assert not val, "'{}': {}".format(name, val)
        for key, val in trigger_hierarchies.items():
            assert not val, "'{}': {}".format(name, val)
        sys.stderr.write('WARNING: No events found in i3 file "{}"\n'.format(i3_fpath))

    photon_series_dir = join(outdir, "photons")
    pulse_series_dir = join(outdir, "pulses")
    recos_dir = join(outdir, "recos")
    trigger_hierarchy_dir = join(outdir, "triggers")

    if photons:
        mkdir(photon_series_dir)
    if pulses:
        mkdir(pulse_series_dir)
    if recos:
        mkdir(recos_dir)
    if triggers:
        mkdir(trigger_hierarchy_dir)

    event_dtype = None
    if events:
        for key, val in events[0].items():
            if event_dtype is None:
                event_dtype = []
            event_dtype.append((key, set_explicit_dtype(val).dtype))

    events = np.array([tuple(ev.values()) for ev in events], dtype=event_dtype)
    np.save(join(outdir, "events.npy"), events)

    if truth:
        np.save(join(outdir, "truth.npy"), np.array(truths))

    for name in photons:
        pickle.dump(
            photons_d[name],
            open(join(photon_series_dir, name + ".pkl"), "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    for name in pulses:
        pickle.dump(
            pulses_d[name],
            open(join(pulse_series_dir, name + ".pkl"), "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        tr_key = name + "TimeRange"
        np.save(
            join(pulse_series_dir, tr_key + ".npy"),
            np.array(pulses_d[tr_key], dtype=np.float32),
        )

    for name in recos:
        np.save(join(recos_dir, name + ".npy"), np.array(recos_d[name]))

    for name in triggers:
        pickle.dump(
            trigger_hierarchies[name],
            open(join(trigger_hierarchy_dir, name + ".pkl"), "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def extract_events(
    i3_files,
    retro_gcd_dir,
    gcd=None,
    outdir=None,
    photons=tuple(),
    pulses=tuple(),
    recos=tuple(),
    triggers=tuple(),
    truth=False,
    additional_keys=None,
):
    """Extract event information from an i3 file.

    Parameters
    ----------
    i3_files : str or iterable thereof
        Path(s) to I3 file(s). If a `gcd` is specified, it will apply to all i3
        files specified that do not GCD frames embedded in them.

    retro_gcd_dir : str
        Path to directory containing Retro-extracted GCD files (each
        subdirectory is named by the md5sum of the corresponding GCD file)

    gcd : str or None
        GCD information for the events _must_ be provided one way or another.
        If GCD frames are embedded in the i3 file(s) passed via `i3_files`, then
        nothing needs to be specified for `gcd`. Otherwise, `gcd` must be a
        string specifying either a path to a GCD file or the MD5 sum derived
        from storing the GCD frames (in that order, with no other frames) to an
        (uncompressed) i3 file (see
        `retro.i3processing.extract_gcd.extract_gcd_frames`); a GCD specified
        in this way is applied to i3 files specified by the `i3_files` argument
        except for those files containing embedded GCD frames (which supersede
        the `gcd` argument).

    outdir : str, optional
        Directory in which to place generated leaf directory (named same as i3
        file but with extensions removed). Default is same directory(ies) where
        the i3 file(s) are stored

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

    additional_keys : str, iterable thereof, or None

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
                'truth': [{'pdg_encoding': ..., 'daughters': [{...}, ...]}],
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
    from retro.i3processing.extract_gcd import MD5_HEX_RE, extract_gcd_files

    if isinstance(i3_files, string_types):
        i3_files = [i3_files]

    if gcd is None:
        gcd_md5_hex = None
    elif isinstance(gcd, string_types):
        if MD5_HEX_RE.match(gcd.strip().lower()):
            gcd_md5_hex = gcd.strip().lower()
            gcd = None
        else:
            gcd_md5_hexs = extract_gcd_files(
                gcd_files=gcd, retro_gcd_dir=retro_gcd_dir
            )
            assert len(gcd_md5_hexs) == 1
            gcd_md5_hex = gcd_md5_hexs[0]
    else:
        raise TypeError("Cannot handle `gcd` arg of type {}".format(type(gcd)))

    for i3_fpath in i3_files:
        _extract_events_from_single_file(
            i3_fpath=i3_fpath,
            retro_gcd_dir=retro_gcd_dir,
            gcd=gcd,
            gcd_md5_hex=gcd_md5_hex,
            outdir=outdir,
            photons=photons,
            pulses=pulses,
            recos=recos,
            triggers=triggers,
            truth=truth,
            additional_keys=additional_keys,
        )


class ExceptionWrapper(object):
    """From https://stackoverflow.com/a/26096355/9632900, user rocksportrocker"""
    def __init__(self, err):
        self.err = err
        __,  __, self.tb = sys.exc_info()

    def re_raise(self):
        if PY2:
            raise self.err, None, self.tb
        else:
            raise self.err.with_traceback(self.tb)


def wrapped_extract_events(*args, **kwargs):
    """`extract_events` function wrapped in a try/except that keeps full
    traceback information. Use if calling from multiprocessing module."""
    try:
        extract_events(*args, **kwargs)
    except Exception as err:
        return ExceptionWrapper(err)


def main(description=__doc__):
    """Script interface to `extract_events` function: Parse command line args
    and call function."""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--i3-files",
        nargs="+",
        required=True,
        help="""Paths to i3 files to extract""",
    )
    parser.add_argument(
        "--retro-gcd-dir",
        required=True,
        help="""Directory into which to store any extracted GCD info""",
    )
    parser.add_argument(
        "--gcd",
        required=False,
        default=None,
        help="""Specify an external GCD file or md5sum (as returned by
        `retro.i3processing.extract_gcd_frames`, i.e., the md5sum of an
        uncompressed i3 file containing _only_ the G, C, and D frames). It is
        not required to specify --gcd if the G, C, and D frames are embedded in
        all files specified by --i3-files. Any GCD frames within said files
        will also take precedent if --gcd _is_ specified.""",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        help="""Directory into which to store the extracted directories and
        files. If not specified, the directory where each i3 file is stored is
        used (a leaf directory is created with the same name as each i3 file
        but with .i3 and any compression extensions removed)."""
    )
    parser.add_argument(
        "--photons",
        nargs="+",
        default=[],
        help="""Photon series names to extract from each event""",
    )
    parser.add_argument(
        "--pulses",
        nargs="+",
        default=[],
        help="""Pulse series names to extract from each event""",
    )
    parser.add_argument(
        "--recos",
        nargs="+",
        default=[],
        help="""Reco names to extract from each event""",
    )
    parser.add_argument(
        "--triggers",
        nargs="+",
        default=[],
        help="""Trigger hierarchy names to extract from each event""",
    )
    parser.add_argument(
        "--truth",
        action="store_true",
        help="""Extract truth information from Monte Carlo events""",
    )
    parser.add_argument(
        "--additional-keys",
        default=None,
        nargs="+",
        help="""Additional keys to extract from event I3 frame""",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    extract_events(**kwargs)


if __name__ == "__main__":
    main()
