#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Populate Retro recos and metadata to I3 files.
"""

from __future__ import absolute_import, division, print_function

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2019 Philipp Eller and Justin L. Lanfranchi

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
    "GMS_LEN2EN",
    "const_en2len",
    "const_en_to_gms_en",
    "DEFAULT_I3PARTICLE_ATTRS",
    "NEUTRINO_ATTRS",
    "TRACK_ATTRS",
    "CASCADE_ATTRS",
    "make_i3_particles",
    "extract_all_reco_info",
    "particle_from_reco",
    "populate_pframe",
    "retro_recos_to_i3files",
    "main",
]

from argparse import ArgumentParser
from collections import OrderedDict
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
from copy import deepcopy
from glob import glob
from os import remove, walk
from os.path import (
    abspath,
    basename,
    dirname,
    expanduser,
    expandvars,
    isfile,
    join,
    relpath,
    splitext,
)
import sys

import numpy as np
from six import string_types

from icecube import (  # pylint: disable=unused-import
    dataclasses,
    recclasses,
    simclasses,
    dataio,
)
from icecube.dataclasses import (  # pylint: disable=no-name-in-module
    I3Constants,
    I3Direction,
    I3MapStringDouble,
    I3Particle,
    I3Double,
)
from icecube.icetray import I3Frame, I3Units, I3Int, I3Bool  # pylint: disable=no-name-in-module
from icecube.dataio import I3File  # pylint: disable=no-name-in-module


if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.muon_hypo import TRACK_M_PER_GEV, generate_gms_table_converters
from retro.retro_types import FitStatus
from retro.utils.misc import nsort_key_func


_, GMS_LEN2EN, _ = generate_gms_table_converters(losses="all")

def const_en2len(energy):
    return energy * TRACK_M_PER_GEV


def const_en_to_gms_en(energy):
    length = const_en2len(energy)
    gms_energy_est = GMS_LEN2EN(length)
    return gms_energy_est


DEFAULT_I3PARTICLE_ATTRS = {
    "dir": dict(
        fields=["zenith", "azimuth"], func=lambda z, a: I3Direction(float(z), float(a))
    ),
    "energy": dict(fields="energy", func=const_en_to_gms_en, units=I3Units.GeV),
    "fit_status": dict(fields="fit_status", func=I3Particle.FitStatus),
    "length": dict(value=np.nan, units=I3Units.m),
    # "location_type" deprecated according to `dataclasses/resources/docs/particle.rst`
    "pdg_encoding": dict(value=I3Particle.ParticleType.unknown),
    "pos.x": dict(fields="x", units=I3Units.m),
    "pos.y": dict(fields="y", units=I3Units.m),
    "pos.z": dict(fields="z", units=I3Units.m),
    "shape": dict(value=I3Particle.ParticleShape.Null),
    "time": dict(fields="time", units=I3Units.ns),
    "speed": dict(value=I3Constants.c),
}

NEUTRINO_ATTRS = deepcopy(DEFAULT_I3PARTICLE_ATTRS)
NEUTRINO_ATTRS["shape"] = dict(value=I3Particle.ParticleShape.Primary)

TRACK_ATTRS = deepcopy(DEFAULT_I3PARTICLE_ATTRS)
TRACK_ATTRS["length"] = dict(
    fields="energy", func=const_en2len, units=I3Units.m
)

CASCADE_ATTRS = deepcopy(DEFAULT_I3PARTICLE_ATTRS)
CASCADE_ATTRS["shape"] = dict(value=I3Particle.ParticleShape.Cascade)


def particle_from_reco(reco, kind, point_estimator, field_format="{field}"):
    """Create an I3Particle from part of a retro reco hypothesis.

    Parameters
    ----------
    reco : numpy.array of struct dtype
    kind : str in {"neutrino", "track", "cascade"}
    point_estimator : str in {"mean", "medain", "max", ...}
    field_format : str, optional

    Returns
    -------
    particle : icecube.dataclasses.I3Particle
    consumed_fields : set

    """
    if kind == "neutrino":
        attrs = NEUTRINO_ATTRS
    elif kind == "track":
        attrs = TRACK_ATTRS
    elif kind == "cascade":
        attrs = CASCADE_ATTRS
    else:
        raise ValueError(kind)

    if field_format is None:
        field_format = "{field}"

    # Create the particle that is to be populated
    particle = I3Particle()

    consumed_fields = set()
    for attr, info in attrs.items():
        if "value" in info:
            value = info["value"]
            try:
                value = value[point_estimator]
            except (KeyError, ValueError, IndexError, TypeError):
                pass
            if hasattr(value, "tolist"):
                value = value.tolist()
            if np.isscalar(value):
                value = [value]
        else:
            fields = info["fields"]
            if isinstance(fields, string_types):
                fields = [fields]

            value = []
            for field in fields:
                # Try the field with attr in the `field_format`, else fallback
                # to "bare" name of the attr.

                # KeyError if Mapping missing field, ValueError if numpy.array
                # w/ struct dtype and name is missing
                consumed_field = field_format.format(field=field)
                try:
                    value_ = reco[consumed_field]
                except (KeyError, ValueError):
                    consumed_field = field
                    value_ = reco[consumed_field]

                consumed_fields.add(consumed_field)

                # Attempt to retrieve the point estimator
                try:
                    value_ = value_[point_estimator]
                except (KeyError, ValueError, IndexError):
                    pass

                # Test to see if we got multiple values for 1 thing
                if (
                    not np.isscalar(value_)
                    or hasattr(value_, "dtype")
                    and len(value_.dtype) > 0
                ):
                    raise ValueError(
                        "value {!r} type {} is not a simple scalar".format(
                            value_, type(value)
                        )
                    )

                value.append(value_)

        # If a function was specified, apply it
        if "func" in info:
            func = info["func"]
            value = func(*value)

        # Should have a single value by now
        if isinstance(value, Sequence):
            assert len(value) == 1
            value = value[0]

        # Test to see if we actually have a scalar and not have a struct dtype
        if not isinstance(value, I3Direction) and (
            not np.isscalar(value) or hasattr(value, "dtype") and len(value.dtype) > 0
        ):
            raise ValueError(
                "value {!r} type {} is not a simple scalar".format(value, type(value))
            )

        # Apply units if specified
        if "units" in info:
            units = info["units"]
            value *= units

        split_attrs = attr.split(".")
        attrs_to_get = split_attrs[:-1]
        attr_to_set = split_attrs[-1]

        obj = particle
        for attr_to_get in attrs_to_get:
            obj = getattr(obj, attr_to_get)
        setattr(obj, attr_to_set, value)

    return particle, consumed_fields


def setitem_pframe(frame, key, val, event_index=None, overwrite=False):
    """Put value in frame, with wrapper for warn or error if the key is already
    present.

    Parameters
    ----------
    frame
    key
    val
    event_index : print-able object, optional
        Some form of event identifier
    overwrite : bool, optional

    """
    if key in frame:
        if not overwrite:
            raise KeyError(
                "frame for event index {} has key '{}' already".format(event_index, key)
            )
        print(
            "WARNING: frame for event index {} has key '{}' already; will be"
            " overwritten".format(event_index, key)
        )
    frame[key] = val


def make_i3_particles(reco, point_estimator):
    """Populate I3Particles as a summary of the reco

    This makes getting a quick (albeit incomplete) summary of retro reco
    results "easy" and somewhat standard in comparison to other IceCube recos
    which populate particles. Keep in mind that all info is NOT able to be
    populated, here, though, so be sure to check the other fields

    Parameters
    ----------
    reco
    point_estimator

    Returns
    -------
    particles_identifiers

    """
    fields_to_consume = set(reco.dtype.names)

    particles_identifiers = []

    particle, consumed_fields = particle_from_reco(
        reco, kind="neutrino", field_format=None, point_estimator=point_estimator
    )
    particles_identifiers.append((particle, "neutrino"))
    fields_to_consume -= consumed_fields

    # TODO: make treatment recognize different track / cascade names, group
    # like names together, and treat as separate tracks / cascades. For now
    # we only have a single track and cascade, so not necessary, but this
    # will eventually be an issue.

    if any("track" in f for f in fields_to_consume):
        particle, consumed_fields = particle_from_reco(
            reco,
            kind="track",
            field_format="track_{field}",
            point_estimator="median",
        )
        particles_identifiers.append((particle, "track"))
        fields_to_consume -= consumed_fields

    if any("cascade" in f for f in fields_to_consume):
        particle, consumed_fields = particle_from_reco(
            reco,
            kind="cascade",
            field_format="cascade_{field}",
            point_estimator="median",
        )
        particles_identifiers.append((particle, "cascade"))
        fields_to_consume -= consumed_fields

    return particles_identifiers


def extract_all_reco_info(reco, reco_name):
    """Populate ALL Retro reco information

    Note we use length-one i3vector types because there aren't standard
    scalar types for anything besides I3Float, and if we want information
    to live on, we don't want to have to maintain custom datatypes for
    Retro inside the IceCube codebase, because who has time for that?

    Parameters
    ----------
    reco

    Returns
    -------
    all_reco_info : dict

    """
    all_reco_info = OrderedDict()

    for field in reco.dtype.names:
        key = "{}__{}".format(reco_name, field)
        val = reco[field]
        if hasattr(val, "dtype") and len(val.dtype) > 0:
            # TODO: handle I3MapStringBool, I3MapStringInt?
            val = I3MapStringDouble(list(zip(val.dtype.names, val.tolist())))
        else:
            val_type = getattr(val, "dtype", type(val))

            # floating types
            if val_type in (
                float,
                np.float,
                np.float_,
                np.float16,
                np.float32,
                np.float64,
            ):
                i3type = I3Double
                pytype = float

            # integer types
            elif val_type in (
                int,
                np.int,
                np.int_,
                np.intp,
                np.integer,
                np.int0,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint,
                np.uintp,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                i3type = I3Int
                pytype = int

            # boolean types
            elif val_type in (
                bool,
                np.bool,
                np.bool_,
                np.bool8,
            ):
                i3type = I3Bool
                pytype = bool

            else:
                raise TypeError("Don't know how to handle type {}".format(val_type))

            val = i3type(pytype(val))

        all_reco_info[key] = val

    return all_reco_info


def populate_pframe(event_index, frame_buffer, recos_d, point_estimator):
    """Create I3Particles from the major components of the reco; all other
    info, populate opportunistically with dict-like I3 objects, by type.

    Result is placed in the last physics frame in `frame_buffer` if one is
    present, otherwise a new physics frame is created and populated and
    appended to the `frame_buffer`.

    Parameters
    ----------
    event_index
    frame_buffer
    recos_d

    Notes
    -----
    For now, only handles one or no track and one or no cascade in the
    hypothesis.

    """
    # -- Look for last physics frame in buffer -- #
    pframe = None
    for frame in frame_buffer[::-1]:
        if frame.Stop == I3Frame.Physics:
            pframe = frame
            break
    if pframe is None:
        pframe = I3Frame(I3Frame.Physics)
        frame_buffer.append(pframe)

    for reco_name, recos in recos_d.items():
        reco = recos[event_index]

        # Do not populate recos that were not performed
        if "fit_status" in reco.dtype.names and reco["fit_status"] == FitStatus.NotSet:
            continue

        particles_identifiers = make_i3_particles(reco, point_estimator)

        for particle, identifier in particles_identifiers:
            key = "__".join([reco_name, point_estimator, identifier])
            setitem_pframe(pframe, key, particle, event_index, overwrite=False)

        all_reco_info = extract_all_reco_info(reco, reco_name)

        for key, val in all_reco_info.items():
            setitem_pframe(pframe, key, val, event_index, overwrite=False)


def retro_recos_to_i3files(
    eventsdir, point_estimator, recos=None, i3dir=None, overwrite=False
):
    """Take retro recos found in .npy files / retro directory structure and
    corresponding i3 files and generate new i3 files like the original but
    populated with the retro reco information.

    Parameters
    ----------
    eventsdir : str
    point_estimator : str in {"mean", "median", "max"}
    recos : str or iterable thereof, optional
        If not specified, all "retro_*" recos found will be populated
    i3dir : str, optional
        If None or not specified, defaults to `eventsdir`
    overwrite : bool

    """
    eventsdir = abspath(expanduser(expandvars(eventsdir)))
    # If the leaf reco/events/truth dir "recos" was specified, must go one up
    # to find events/truth
    if basename(eventsdir) == "recos":
        eventsdir = dirname(eventsdir)

    if recos is None:
        recos = [
            splitext(basename(n))[0] for n in glob(join(eventsdir, "recos", "retro_*.npy"))
        ]

    if isinstance(recos, string_types):
        recos = [recos]
    else:
        recos = sorted(list(recos))
    for reco in recos:
        if not reco.startswith("retro_"):
            raise ValueError(
                'Can only populate "retro_*" recos; "{}" is invalid'.format(reco)
            )

    if i3dir is None:
        i3dir = eventsdir
    else:
        i3dir = abspath(expanduser(expandvars(i3dir)))

    # -- Walk directories and match (events, recos) to i3 paths -- #

    for events_dirpath, dirs, filenames in walk(eventsdir):
        dirs.sort(key=nsort_key_func)
        if "events.npy" not in filenames:
            continue

        missing_recos = []
        reco_filepaths = {}
        for reco in recos:
            reco_filepath = join(events_dirpath, "recos", "{}.npy".format(reco))
            if isfile(reco_filepath):
                reco_filepaths[reco] = reco_filepath
            else:
                missing_recos.append(reco)

        if missing_recos:
            print(
                'WARNING: Missing recos {} in dir "{}"'.format(
                    missing_recos, events_dirpath
                )
            )
            if set(missing_recos) == set(recos):
                continue

        eventsdir_basename = basename(events_dirpath)
        i3filedir = join(i3dir, relpath(dirname(events_dirpath), start=eventsdir))
        i3filepaths = sorted(
            glob(join(i3filedir, "{}.i3*".format(eventsdir_basename))),
            key=nsort_key_func,
        )
        if not i3filepaths:
            raise IOError(
                'No matching i3 file "{}.i3*" in directory "{}"'.format(
                    eventsdir_basename, i3filedir
                )
            )
        input_i3filepath = i3filepaths[0]
        if len(i3filepaths) > 1:
            print(
                'WARNING: found multiple i3 files in dir, picking first one "{}"'.format(
                    i3filepaths
                )
            )
        print("input_i3filepath:", input_i3filepath)

        suffix = "__" + "__".join(sorted(reco_filepaths.keys()))
        output_i3filepath = join(
            i3filedir,
            "{base}{suffix}{extensions}".format(
                base=basename(input_i3filepath)[: len(eventsdir_basename)],
                suffix=suffix,
                extensions=".i3.zst",
            ),
        )
        if not overwrite and isfile(output_i3filepath):
            print(
                'WARNING: skipping writing output path that already exists: "{}"'.format(
                    output_i3filepath
                )
            )
            continue
        print("output_i3filepath:", output_i3filepath)

        print("events_dirpath:", events_dirpath)
        events = np.load(join(events_dirpath, "events.npy"))
        recos_d = OrderedDict()
        for reco, reco_filepath in reco_filepaths.items():
            recos_d[reco] = np.load(reco_filepath)
            if len(recos_d[reco]) != len(events):
                raise ValueError(
                    "{} has len {}, events has len {}".format(
                        reco, len(recos_d[reco]), len(events)
                    )
                )

        # Collect frames into an event chain until we hit a physics frame, a
        # second DAQ frame, or the end of the file.
        #
        # * If we have only a DAQ frame in the chain, create a new Physics
        #   frame and populate the reco(s) to it.
        #
        # * If we have a Physics frame, populate the recos to that frame.
        #
        # * If we have no DAQ or Physics frames in the chain, we should be
        #   done. Make sure we've accounted for all the recos in the npy files
        #   and quit.
        #
        # When done with the chain, push all frames in the chain to the output file.
        # physics frame, we have a new "event" to process; populate recos to
        # that frame. Then, regardless of why we finished the event chain,
        # write the frames in the chain out to the new i3 file.

        input_i3file = I3File(input_i3filepath, "r")
        output_i3file = I3File(output_i3filepath, "w")

        frame_buffer = []
        chain_has_daq_frame = False
        chain_has_physics_frame = False
        frame_counter = 0
        event_index = -1

        try:
            while True:
                if input_i3file.more():
                    try:
                        next_frame = input_i3file.pop_frame()
                    except:
                        sys.stderr.write(
                            "Failed to pop frame #{}\n".format(frame_counter + 1)
                        )
                        raise
                    frame_counter += 1
                else:
                    next_frame = None

                # Current chain has ended and a new one will have to be started
                # (or we're at the end of the file).

                # Populate the reco to the current chain, push all of the
                # current chain's frames to the output file, and start a new
                # chain with the next frame (or quit if we're at the end of the
                # file).
                if (
                    next_frame is None
                    or next_frame.Stop == I3Frame.DAQ
                    or (chain_has_physics_frame and next_frame.Stop == I3Frame.Physics)
                ):
                    if frame_buffer:
                        # Events are identified as a chain with daq frame being
                        # present with no physics frame, physics frame present
                        # with no daq frame, or both being present (existence
                        # of other frames is considered to be irrelevant)

                        # TODO: oscNext v01.01 by L5, i3 file processing was
                        # messed up, there were Q frames followed by I frames
                        # and no associated P frame. Therefore we have to only
                        # count chains with P frames in them as events, or else
                        # the recos won't be put back in the right place /
                        # indices run out.
                        #if chain_has_daq_frame or chain_has_physics_frame:

                        if chain_has_physics_frame:
                            event_index += 1
                            populate_pframe(
                                event_index=event_index,
                                frame_buffer=frame_buffer,
                                recos_d=recos_d,
                                point_estimator=point_estimator,
                            )

                        # Regardless if there was an event identified in the
                        # chain, push all frames to the output file
                        for frame in frame_buffer:
                            output_i3file.push(frame)

                    # No next frame indicates we hit the end of the file; quit
                    if next_frame is None:
                        break

                    # Create a new chain, starting with the next frame
                    frame_buffer = [next_frame]
                    chain_has_daq_frame = next_frame.Stop == I3Frame.DAQ
                    chain_has_physics_frame = next_frame.Stop == I3Frame.Physics

                # Otherwise, we have just another frame in the current chain;
                # append it and move on.
                else:
                    frame_buffer.append(next_frame)
                    chain_has_daq_frame |= next_frame.Stop == I3Frame.DAQ
                    chain_has_physics_frame |= next_frame.Stop == I3Frame.Physics

        except:
            output_i3file.close()
            del output_i3file
            remove(output_i3filepath)

            sys.stderr.write(
                'ERROR! file "{}", frame #{}\n'.format(
                    input_i3filepath, frame_counter + 1
                )
            )
            raise

        else:
            output_i3file.close()
            del output_i3file


def main(description=__doc__):
    """Script interface to `populate_recos` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--recos",
        default=None,
        nargs="+",
        help="""Reco names to populate to the i3 file(s)""",
    )
    parser.add_argument(
        "--eventsdir",
        required=True,
        help="""Parent directory in which to look for Retro events / reconstructions""",
    )
    parser.add_argument(
        "--i3dir",
        required=False,
        default=None,
        help="""Directory with parallel structure to --eventsdir in which to
        look for i3 files to populate""",
    )
    parser.add_argument(
        "--point-estimator",
        required=True,
        choices=("mean", "median", "max"),
        help="""Reconstructed values estimated from a posterior distribution
        have several point estimators; choose one to use for producing
        I3Particles describing the basic reconstructed varaibles of the
        reconstruction.""",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""Overwrite existing output file(s) if they exist""",
    )
    kwargs = vars(parser.parse_args())
    retro_recos_to_i3files(**kwargs)


if __name__ == "__main__":
    main()
