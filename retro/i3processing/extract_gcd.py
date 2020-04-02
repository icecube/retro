#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, line-too-long

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
    "GCD_README",
    "MD5_HEX_RE",
    "extract_i3_geometry",
    "extract_i3_calibration",
    "extract_i3_detector_status",
    "extract_bad_doms_lists",
    "extract_gcd_frames",
    "extract_gcd_files",
    "parse_args",
]

from argparse import ArgumentParser
from collections import OrderedDict
try:
    from collections import Mapping
except ImportError:
    from collections.abc import Mapping
import errno
import io
import json
from os.path import abspath, dirname, isdir, isfile, join
import pickle
import re
from shutil import copytree, rmtree
import socket
import sys
from tempfile import mkdtemp

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import (
    I3DOMCALIBRATION_T,
    I3DOMSTATUS_T,
    I3OMGEO_T,
    I3TRIGGERREADOUTCONFIG_T,
    I3TIME_T,
    OMKEY_T,
    TRIGGERKEY_T,
    CableType,
    DOMGain,
    LCMode,
    OnOff,
    ToroidType,
    TrigMode,
    OMType,
    TriggerSourceID,
    TriggerTypeID,
    TriggerSubtypeID,
    TriggerConfigID,
)
from retro.utils.misc import expand, get_file_md5, mkdir
from retro.i3processing.extract_common import dict2struct
from retro.i3processing.extract_events import START_END_TIME_SPEC, get_frame_item


GCD_README = """
Directory names are md5 hashes of the _decompressed_ GCD i3 files (or if G, C,
and D frames were found inline in an i3 data file, each unique combination of
GCD frames are output to an i3 file and hashed).
"""


MD5_HEX_RE = re.compile("^[0-9a-f]{32}$")


def extract_i3_geometry(frame):
    """Extract I3Geometry object from frame.

    Note that for now, the `stationgeo` attribute of the I3Geometry frame
    object is omitted.

    Parameters
    ----------
    frame : icecube.icetray.I3Frame
        Must contain key "I3Geometry" whose value is an
        ``icecube.dataclasses.I3Geometry`` object

    Returns
    -------
    geometry : OrderedDict
        Keys are the properties (excluding stationgeo) of the I3Geometry frame:
        "start_time", "end_time", and "omgeo". Values are Numpy arrays of
        structured dtypes containing (most) of the info from each.

    """
    # Get `start_time` & `end_time` using standard functions
    geometry = get_frame_item(
        frame, key="I3Geometry", specs=START_END_TIME_SPEC, allow_missing=False
    )

    # Get omgeo, which is not simply extractable using `get_frame_item` func
    omgeo_frame_obj = frame["I3Geometry"].omgeo

    omgeo = np.empty(shape=len(omgeo_frame_obj), dtype=I3OMGEO_T)
    for i, omkey in enumerate(sorted(omgeo_frame_obj.keys())):
        geo = omgeo_frame_obj[omkey]

        omgeo[i]["omkey"]["string"] = omkey.string
        omgeo[i]["omkey"]["om"] = omkey.om
        omgeo[i]["omkey"]["pmt"] = omkey.pmt
        omgeo[i]["omtype"] = OMType(int(geo.omtype))
        omgeo[i]["area"] = geo.area
        omgeo[i]["position"]["x"] = geo.position.x
        omgeo[i]["position"]["y"] = geo.position.y
        omgeo[i]["position"]["z"] = geo.position.z
        omgeo[i]["direction"]["azimuth"] = geo.direction.azimuth
        omgeo[i]["direction"]["zenith"] = geo.direction.zenith

    geometry["omgeo"] = omgeo

    # TODO: extract stationgeo

    return geometry


def extract_i3_calibration(frame):
    """Extract I3Calibration object from frame.

    Note that for now, the `vem_cal` attribute of the I3Calibration frame
    object is omitted.

    Parameters
    ----------
    frame : icecube.icetray.I3Frame
        Must contain key "I3Calibration", whose value is an
        ``icecube.dataclasses.I3Calibration`` object

    Returns
    -------
    calibration : OrderedDict
        Keys are the properties (excluding vem_cal) of the I3Calibration
        frame: "start_time", "end_time", and "dom_cal". Values are Numpy arrays
        of structured dtypes containing (some) of the info from each.

    """
    # Get `start_time` & `end_time` using standard functions
    calibration = get_frame_item(
        frame, key="I3Calibration", specs=START_END_TIME_SPEC, allow_missing=False
    )

    i3_calibration_frame_obj = frame["I3Calibration"]
    dom_cal_frame_obj = i3_calibration_frame_obj.dom_cal

    dom_cal = np.empty(shape=len(dom_cal_frame_obj), dtype=I3DOMCALIBRATION_T)
    for i, omkey in enumerate(sorted(dom_cal_frame_obj.keys())):
        cal = dom_cal_frame_obj[omkey]
        if cal.dom_cal_version == "unknown":
            ver = [-1, -1, -1]
        else:
            ver = [np.int8(int(x)) for x in cal.dom_cal_version.split(".")]

        dom_cal[i]["omkey"]["string"] = omkey.string
        dom_cal[i]["omkey"]["om"] = omkey.om
        dom_cal[i]["omkey"]["pmt"] = omkey.pmt
        dom_cal[i]["dom_cal_version"]["major"] = ver[0] if len(ver) > 0 else 0
        dom_cal[i]["dom_cal_version"]["minor"] = ver[1] if len(ver) > 1 else 0
        dom_cal[i]["dom_cal_version"]["rev"] = ver[2] if len(ver) > 2 else 0
        dom_cal[i]["dom_noise_decay_rate"] = cal.dom_noise_decay_rate
        dom_cal[i]["dom_noise_rate"] = cal.dom_noise_rate
        dom_cal[i]["dom_noise_scintillation_hits"] = cal.dom_noise_scintillation_hits
        dom_cal[i]["dom_noise_scintillation_mean"] = cal.dom_noise_scintillation_mean
        dom_cal[i]["dom_noise_scintillation_sigma"] = cal.dom_noise_scintillation_sigma
        dom_cal[i]["dom_noise_thermal_rate"] = cal.dom_noise_thermal_rate
        dom_cal[i]["fadc_beacon_baseline"] = cal.fadc_beacon_baseline
        dom_cal[i]["fadc_delta_t"] = cal.fadc_delta_t
        dom_cal[i]["fadc_gain"] = cal.fadc_gain
        dom_cal[i]["front_end_impedance"] = cal.front_end_impedance
        dom_cal[i]["is_mean_atwd_charge_valid"] = cal.is_mean_atwd_charge_valid
        dom_cal[i]["is_mean_fadc_charge_valid"] = cal.is_mean_fadc_charge_valid
        dom_cal[i]["mean_atwd_charge"] = cal.mean_atwd_charge
        dom_cal[i]["mean_fadc_charge"] = cal.mean_fadc_charge
        dom_cal[i]["relative_dom_eff"] = cal.relative_dom_eff
        dom_cal[i]["temperature"] = cal.temperature
        dom_cal[i]["toroid_type"] = ToroidType(cal.toroid_type)

    calibration["dom_cal"] = dom_cal

    return calibration


def extract_i3_detector_status(frame):
    """Extract (most, maybe?) items found in a I3Detector frame.

    Parameters
    ----------
    frame : icecube.icetray.I3Frame
        Must contain key "I3DetectorStatus", whose value is an
        ``icecube.dataclasses.I3DetectorStatus`` object

    Returns
    -------
    detector_status : OrderedDict
        Roughly equivalent Numpy representation of I3DetectorStatus frame object

    Notes
    -----
    Frame keys seen that aren't extracted: IceTopBadTanks and
    I3FlasherSubrunMap; there might be an infinite number of others missed.
    Caveat emptor.

    """
    # Get `start_time` & `end_time` using standard functions
    detector_status = get_frame_item(
        frame, key="I3DetectorStatus", specs=START_END_TIME_SPEC, allow_missing=False
    )

    # TODO: roll the following few items into a single spec incl.
    #       START_END_TIME_SPEC

    key = "GRLSnapshotId"
    if key in frame:
        detector_status[key] = np.float64(frame[key].value)

    key = "OfflineProductionVersion"
    if key in frame:
        detector_status[key] = np.float64(frame[key].value)

    for key in ["GoodRunStartTime", "GoodRunEndTime"]:
        if key not in frame:
            continue
        detector_status[key] = np.array(
            [tuple(getattr(frame[key], name) for name in I3TIME_T.names)],
            dtype=I3TIME_T,
        )

    # DETECTOR_STATUS_T = np.dtype(
    #    [
    #        ('start_time', I3Time(2011,116382900000000000L)),
    #        ('end_time', I3Time(2011,116672520000000000L)),
    #        ('daq_configuration_name', 'sps-IC86-mitigatedHVs-V175'),
    #        ('dom_status', <icecube.dataclasses.Map_OMKey_I3DOMStatus>),
    #        ('trigger_status', <icecube.dataclasses.Map_TriggerKey_I3TriggerStatus>),
    #    ]
    # )

    i3_detector_status_frame_obj = frame["I3DetectorStatus"]

    dom_status_frame_obj = i3_detector_status_frame_obj.dom_status
    dom_status = np.empty(shape=len(dom_status_frame_obj), dtype=I3DOMSTATUS_T)
    for i, omkey in enumerate(sorted(dom_status_frame_obj.keys())):
        this_dom_status = dom_status_frame_obj[omkey]
        dom_status[i]["omkey"]["string"] = omkey.string
        dom_status[i]["omkey"]["om"] = omkey.om
        dom_status[i]["omkey"]["pmt"] = omkey.pmt
        dom_status[i]["cable_type"] = CableType(this_dom_status.cable_type)
        dom_status[i]["dac_fadc_ref"] = this_dom_status.dac_fadc_ref
        dom_status[i]["dac_trigger_bias_0"] = this_dom_status.dac_trigger_bias_0
        dom_status[i]["dac_trigger_bias_1"] = this_dom_status.dac_trigger_bias_1
        dom_status[i]["delta_compress"] = OnOff(this_dom_status.delta_compress)
        dom_status[i]["dom_gain_type"] = DOMGain(this_dom_status.dom_gain_type)
        dom_status[i]["fe_pedestal"] = this_dom_status.fe_pedestal
        dom_status[i]["lc_mode"] = LCMode(this_dom_status.lc_mode)
        dom_status[i]["lc_span"] = this_dom_status.lc_span
        dom_status[i]["lc_window_post"] = this_dom_status.lc_window_post
        dom_status[i]["lc_window_pre"] = this_dom_status.lc_window_pre
        dom_status[i]["mpe_threshold"] = this_dom_status.mpe_threshold
        dom_status[i]["n_bins_atwd_0"] = this_dom_status.n_bins_atwd_0
        dom_status[i]["n_bins_atwd_1"] = this_dom_status.n_bins_atwd_1
        dom_status[i]["n_bins_atwd_2"] = this_dom_status.n_bins_atwd_2
        dom_status[i]["n_bins_atwd_3"] = this_dom_status.n_bins_atwd_3
        dom_status[i]["n_bins_fadc"] = this_dom_status.n_bins_fadc
        dom_status[i]["pmt_hv"] = this_dom_status.pmt_hv
        dom_status[i]["slc_active"] = this_dom_status.slc_active
        dom_status[i]["spe_threshold"] = this_dom_status.spe_threshold
        dom_status[i]["status_atwd_a"] = OnOff(this_dom_status.status_atwd_a)
        dom_status[i]["status_atwd_b"] = OnOff(this_dom_status.status_atwd_b)
        dom_status[i]["status_fadc"] = OnOff(this_dom_status.status_fadc)
        dom_status[i]["trig_mode"] = TrigMode(this_dom_status.trig_mode)
        dom_status[i]["tx_mode"] = LCMode(this_dom_status.tx_mode)

    detector_status["dom_status"] = dom_status

    # Trigger status does not have uniform sub-fields across all types, so
    # build up dict keyed by str(trigger index)
    trigger_status_frame_obj = i3_detector_status_frame_obj.trigger_status
    trigger_status = OrderedDict()
    for trigger_key_fobj, trigger_status_fobj in trigger_status_frame_obj.items():
        this_trigger_config = OrderedDict()

        trigger_key = np.empty(shape=1, dtype=TRIGGERKEY_T)
        trigger_key["source"] = TriggerSourceID(trigger_key_fobj.source)
        trigger_key["type"] = TriggerTypeID(trigger_key_fobj.type)
        trigger_key["subtype"] = TriggerSubtypeID(trigger_key_fobj.subtype)
        # TODO: some config ID's aren't defined in the TriggerConfigID enum, no
        # idea where they come from and whether or not it's a bug. For now,
        # simply accept all ID's.
        # trigger_key["config_id"] = TriggerConfigID(trigger_key_fobj.config_id)
        if (
            hasattr(trigger_key_fobj, "config_id")
            and trigger_key_fobj.config_id is not None
        ):
            config_id = trigger_key_fobj.config_id
        else:
            config_id = TriggerConfigID.NONE  # custom code; see `retro_types`
        trigger_key["config_id"] = config_id

        this_trigger_config["trigger_key"] = trigger_key

        readout_settings = OrderedDict()
        for subdet, settings in trigger_status_fobj.readout_settings.items():
            trigger_readout_config = np.empty(shape=1, dtype=I3TRIGGERREADOUTCONFIG_T)
            trigger_readout_config[0][
                "readout_time_minus"
            ] = settings.readout_time_minus
            trigger_readout_config[0]["readout_time_plus"] = settings.readout_time_plus
            trigger_readout_config[0][
                "readout_time_offset"
            ] = settings.readout_time_offset
            readout_settings[str(subdet)] = trigger_readout_config

        this_trigger_config["readout_settings"] = dict2struct(readout_settings)

        trigger_status[tuple(trigger_key[0])] = this_trigger_config

    detector_status["trigger_status"] = trigger_status
    detector_status["daq_configuration_name"] = np.string0(
        i3_detector_status_frame_obj.daq_configuration_name
    )

    return detector_status


def extract_bad_doms_lists(frame):
    """Extract frame objects named "*BadDoms*" (case insensitive), each of
    which must be a ``icecube.dataclasses.I3VectorOMKey``.

    Parameters
    ----------
    frame : icecube.icetray.I3Frame
        Must contain key(s) "*BadDomsList*" (case insensitive), each of which
        whose values are ``icecube.dataclasses.I3VectorOMKey`` objects

    Returns
    -------
    bad_doms_lists : OrderedDict
        Keys are names of frame objects, values are Numpy arrays with dtype
        `retro.retro_types.OMKEY_T`

    """
    bad_doms_lists = OrderedDict()
    for key, bad_doms_list_obj in frame.items():
        if "baddoms" not in key.lower():
            continue
        bad_doms = np.empty(shape=len(bad_doms_list_obj), dtype=OMKEY_T)
        for i, omkey in enumerate(bad_doms_list_obj):
            bad_doms[i]["string"] = omkey.string
            bad_doms[i]["om"] = omkey.om
            bad_doms[i]["pmt"] = omkey.pmt
        bad_doms_lists[key] = bad_doms
    return bad_doms_lists


def extract_gcd_frames(g_frame, c_frame, d_frame, retro_gcd_dir, metadata=None):
    """Extract GCD info to Python/Numpy-readable objects stored to a central
    GCD directory, subdirs of which are named by the hex md5sum of each
    extracted GCD file.

    Parameters
    ----------
    g_frame : icecube.icetray.I3Frame with stop I3Frame.Geometry
    c_frame : icecube.icetray.I3Frame with stop I3Frame.Calibration
    d_frame : icecube.icetray.I3Frame with stop I3Frame.DetectorStatus
    retro_gcd_dir : string
    metadata : None or mapping, optional
        If non-empty mapping (e.g., OrderedDict) is provided, the contents are
        written to the gcd file's subdirectory inside retro_gcd_dir as
        "metadata.json"

    Returns
    -------
    gcd_md5_hex : len-32 string of chars 0-9 and/or a-f
        MD5 sum of _only_ the G, C, and D frames (in that order) dumped to an
        uncompressed i3 file. Note that this can result in a hash value
        different from hashing the original GCD file if other frames were
        present besides the GCD frames (such as an I frame, or Q/P/etc. if the
        GCD is embedded in a data i3 file)

    """
    from icecube.dataio import I3File  # pylint: disable=import-outside-toplevel

    retro_gcd_dir = expand(retro_gcd_dir)

    # Create root dir for gcd subdirs if necessary
    if not isdir(retro_gcd_dir):
        mkdir(retro_gcd_dir)

    # Add a vaguely useful README to gcd root dir
    readme_fpath = join(retro_gcd_dir, "README")
    if not isfile(readme_fpath):
        with io.open(readme_fpath, "w", encoding="utf-8") as fhandle:
            fhandle.write(GCD_README.strip() + "\n")

    # Find md5sum of an uncompressed GCD file created by these G, C, & D frames
    tempdir_path = mkdtemp(suffix="gcd")
    try:
        gcd_i3file_path = join(tempdir_path, "gcd.i3")
        gcd_i3file = I3File(gcd_i3file_path, "w")
        gcd_i3file.push(g_frame)
        gcd_i3file.push(c_frame)
        gcd_i3file.push(d_frame)
        gcd_i3file.close()
        gcd_md5_hex = get_file_md5(gcd_i3file_path)
    finally:
        try:
            rmtree(tempdir_path)
        except Exception:
            pass

    this_gcd_dir_path = join(retro_gcd_dir, gcd_md5_hex)
    if isdir(this_gcd_dir_path):
        # already extracted this GCD
        sys.stderr.write("Already extracted GCD with md5sum {}\n".format(gcd_md5_hex))
        return gcd_md5_hex

    tempdir_path = mkdtemp(suffix="." + gcd_md5_hex)
    try:
        # Extract GCD info into Python/Numpy-readable things
        gcd_info = OrderedDict()
        gcd_info["I3Geometry"] = extract_i3_geometry(g_frame)
        gcd_info["I3Calibration"] = extract_i3_calibration(c_frame)
        gcd_info["I3DetectorStatus"] = extract_i3_detector_status(d_frame)
        gcd_info.update(extract_bad_doms_lists(d_frame))

        # Write info to files. Preferable to write a single array to a .npy file;
        # second most preferable is to write multiple arrays to (compressed) .npz
        # file (faster to load than pkl files); finally, I3DetectorStatus _has_ to
        # be stored as pickle to preserve varying-length items.
        for key, val in gcd_info.items():
            if isinstance(val, Mapping):
                if key == "I3DetectorStatus":
                    key_fpath = join(tempdir_path, key + ".pkl")
                    with io.open(key_fpath, "wb") as fhandle:
                        pickle.dump(val, fhandle, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    np.savez_compressed(join(tempdir_path, key + ".npz"), **val)
            else:
                assert isinstance(val, np.ndarray)
                np.save(join(tempdir_path, key + ".npy"), val)

        if metadata:
            metadata_fpath = join(tempdir_path, "metadata.json")
            with open(metadata_fpath, "w") as fhandle:
                json.dump(metadata, fhandle, sort_keys=False, indent=4)

        try:
            copytree(tempdir_path, this_gcd_dir_path)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

    finally:
        try:
            rmtree(tempdir_path)
        except Exception:
            pass

    return gcd_md5_hex


def extract_gcd_files(gcd_files, retro_gcd_dir, verbosity=0):
    """
    Parameters
    ----------
    gcd_files : string or iterable thereof

    retro_gcd_dir : string
        Path to communal Retro-extracted GCD dir

    verbosity : int in [0, 1]

    Returns
    -------
    gcd_md5_hexs : len(gcd_files)-list of strings

    """
    # Import here so module can be read without access to IceCube software
    from icecube.dataio import I3File  # pylint: disable=no-name-in-module, import-outside-toplevel
    from icecube.icetray import I3Frame  # pylint: disable=no-name-in-module, import-outside-toplevel

    if isinstance(gcd_files, string_types):
        gcd_files = [gcd_files]

    gcd_md5_hexs = []
    for gcd_fpath in gcd_files:
        gcd_fpath = expand(gcd_fpath)
        i3f = I3File(gcd_fpath)
        gcd_frames = OrderedDict()
        while i3f.more():
            frame = i3f.pop_frame()
            if frame.Stop == I3Frame.Geometry:
                if "g_frame" in gcd_frames:
                    raise ValueError(
                        'GCD file "{}" contains multiple G frames'.format(gcd_fpath)
                    )
                gcd_frames["g_frame"] = frame
            elif frame.Stop == I3Frame.Calibration:
                if "c_frame" in gcd_frames:
                    raise ValueError(
                        'GCD file "{}" contains multiple C frames'.format(gcd_fpath)
                    )
                gcd_frames["c_frame"] = frame
            elif frame.Stop == I3Frame.DetectorStatus:
                if "d_frame" in gcd_frames:
                    raise ValueError(
                        'GCD file "{}" contains multiple D frames'.format(gcd_fpath)
                    )
                gcd_frames["d_frame"] = frame
        for frame_type in ("g", "c", "d"):
            if "{}_frame".format(frame_type) not in gcd_frames:
                raise ValueError(
                    'No {} frame found in GCD file "{}"'.format(frame_type, gcd_fpath)
                )
        metadata = OrderedDict()
        metadata["extracted_on_fqdn"] = socket.getfqdn()
        metadata["path_to_sourcefile"] = abspath(gcd_fpath)
        metadata["sourcefile_md5sum"] = get_file_md5(gcd_fpath)
        try:
            gcd_md5_hex = extract_gcd_frames(
                retro_gcd_dir=retro_gcd_dir,
                metadata=metadata,
                **gcd_frames
            )
        except Exception:
            sys.stderr.write('failed to extract GCD file "{}"\n'.format(gcd_fpath))
            raise
        gcd_md5_hexs.append(gcd_md5_hex)
        if verbosity:
            sys.stdout.write("{}  {}\n".format(gcd_md5_hex, gcd_fpath))

    return gcd_md5_hexs


def parse_args(description=__doc__):
    """Script interface to `extract_events` function: Parse command line args
    and call function."""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--gcd-files",
        required=True,
        nargs="+",  # allow one or more GCD files
        help="""GCD i3 file(s) to extract (each optionally compressed in a
        manner icecube software understands... gz, bz2, zst).""",
    )
    parser.add_argument(
        "--retro-gcd-dir",
        required=True,
        help="""Directory into which to store the extracted GCD info""",
    )
    parser.add_argument(
        "-v", dest="verbosity", action="count", default=0, help="set verbosity level"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    extract_gcd_files(**vars(parse_args()))
