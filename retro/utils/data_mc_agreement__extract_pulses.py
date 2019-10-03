#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Produce 3 arrays for a given data (MC) set: events_array.npy, doms_array.npy,
and pulses_array.npy.

The first includes per-event information and indexes into the second. The
second includes per-DOM information and indexes into the third. The third is
just an array of pulse times and charges.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["process_events_dir", "produce_arrays", "main"]

from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool
from os import walk
from os.path import abspath, basename, dirname, isfile, join
import re
import sys

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.retro_types import NEUTRINOS
from retro.utils.misc import expand, mkdir, nsort_key_func


# DATA/MC pulses idx array contains per-event information and an index into
# DOMs array & number of DOMs

# DOMs array contains per-DOM information: index into Pulses array & number of
# pulses

# Pulses array contains per-pulse information: charge and time


UNIVERSAL_DOMS_IDX_T_FIELDS = [
    ("run_id", np.uint32),
    ("sub_run_id", np.uint32),  # always set to 0 in data 'cuz IceCube DGAF
    ("event_id", np.uint32),
    ("sub_event_id", np.uint32),

    ("dom_idx0", np.uint64),
    ("num_hit_doms", np.uint16),

    ("num_pulses", np.uint32),
    ("charge", np.float32),

    ("LineFit_DC__pos_x", np.float32),
    ("LineFit_DC__pos_y", np.float32),
    ("LineFit_DC__pos_z", np.float32),
    ("LineFit_DC__time", np.float32),
    ("LineFit_DC__dir_azimuth", np.float32),
    ("LineFit_DC__dir_zenith", np.float32),
]

DATA_DOMS_IDX_T = np.dtype(
    UNIVERSAL_DOMS_IDX_T_FIELDS
    + [
        ("season", np.uint8),  # e.g., 11, 12, ...
        ("actual_sub_run_id", np.uint32),  # need to get from filename

        ("utc_year", np.uint16),  # e.g., 2011, 2012, ...
        ("utc_daq_time", np.uint64),  # ns since beginning of utc year
    ]
)

MC_DOMS_IDX_T = np.dtype(
    UNIVERSAL_DOMS_IDX_T_FIELDS
    + [
        ("dataset", np.uint32),
        ("file_id", np.uint32),

        ("weight", np.float32),
        ("true_pdg", np.int8),
        ("true_int", np.uint8),
        ("true_energy", np.float32),
        ("true_time", np.float32),
    ]
)

DOM_PULSES_IDX_T = np.dtype(
    [
        ("string", np.uint8),
        ("om", np.uint8),

        ("pulses_idx0", np.uint64),
        ("num_pulses", np.uint8),

        ("charge", np.float32),
    ]
)

SIMPLE_PULSE_T = np.dtype(
    [
        ("charge", np.float32),
        ("time", np.float32),
    ]
)

DATA_DIRPATH_META_RE = re.compile(
    r"""
    oscNext_data_
    IC86\.(?P<season>[0-9]+)_
    level(?P<level>[0-9]+)_
    v(?P<version>[0-9.]+)_
    pass(?P<pass>[0-9])_
    Run(?P<run_id>[0-9]+)_
    Subrun(?P<sub_run_id>[0-9]+)
    """,
    (re.VERBOSE | re.IGNORECASE),
)

MC_DIRPATH_META_RE = re.compile(
    r"""
    oscNext_
    (?P<mc_type>[a-z0-9]+)_
    level(?P<level>[0-9]+)_
    v(?P<version>[0-9.]+)_
    pass(?P<pass>[0-9]+)\.
    (?P<dataset>[0-9]+)\.
    (?P<file_id>[0-9]+)
    """,
    (re.VERBOSE | re.IGNORECASE),
)

COPY_ID_FIELDS = ["run_id", "sub_run_id", "event_id", "sub_event_id"]

COPY_TIME_FIELDS = ["utc_year", "utc_daq_time"]

COPY_LINEFIT_DC_SRC_FIELDS = ["x", "y", "z", "time", "azimuth", "zenith"]

COPY_LINEFIT_DC_DST_FIELDS = [
    "LineFit_DC__pos_x",
    "LineFit_DC__pos_y",
    "LineFit_DC__pos_z",
    "LineFit_DC__time",
    "LineFit_DC__dir_azimuth",
    "LineFit_DC__dir_zenith",
]


def process_events_dir(events_dirpath, pulse_series):
    """
    Parameters
    ----------
    events_dirpath : string
    pulse_series : string

    Returns
    -------
    events_array : numpy ndarray
        ndarray dtype is `DATA_DOMS_IDX_T` if is data, otherwise `MC_DOMS_IDX_T`

    doms_array : numpy ndarray of dtype `DOM_PULSES_IDX_T`

    pulses_array : numpy ndarray of dtype `SIMPLE_PULSE_T`

    """
    events_dirpath = expand(events_dirpath)
    basedir = basename(events_dirpath)
    events = np.load(join(events_dirpath, "events.npy"), mmap_mode="r")
    if len(events) == 0:
        return None

    mask_vals = events["L5_oscNext_bool"]
    valid_event_indices = np.argwhere(mask_vals).flatten()
    num_valid_events = len(valid_event_indices)
    if num_valid_events == 0:
        return None

    truth = None
    weights = None
    is_noise = False
    if isfile(join(events_dirpath, "truth.npy")):  # is Monte Carlo simulation
        is_data = False
        truth = np.load(join(events_dirpath, "truth.npy"), mmap_mode="r")
        weights = truth["weight"]
        events_dtype = MC_DOMS_IDX_T
        is_noise = "pdg_encoding" not in truth.dtype.names
        match = MC_DIRPATH_META_RE.match(basedir)
        if not match:
            raise ValueError(events_dirpath)
        finfo_d = match.groupdict()
        finfo_d["dataset"] = int(finfo_d["dataset"])
        finfo_d["file_id"] = int(finfo_d["file_id"])
    else:  # is actual detector data
        is_data = True
        events_dtype = DATA_DOMS_IDX_T
        match = DATA_DIRPATH_META_RE.match(basename(events_dirpath))
        if not match:
            raise ValueError(events_dirpath)
        finfo_d = match.groupdict()
        finfo_d["season"] = int(finfo_d["season"])
        finfo_d["sub_run_id"] = int(finfo_d["sub_run_id"])

    events_array = np.empty(shape=num_valid_events, dtype=events_dtype)

    doms_arrays = []
    pulses_arrays = []

    dom_idx0 = 0
    pulses_idx0 = 0

    pulses = load_pickle(join(events_dirpath, "pulses", "{}.pkl".format(pulse_series)))
    linefit_dc = np.load(join(events_dirpath, "recos", "LineFit_DC.npy"))

    for rel_idx, valid_idx in enumerate(valid_event_indices):
        events_array[rel_idx:rel_idx+1][COPY_ID_FIELDS] = events[valid_idx][COPY_ID_FIELDS]
        events_array[rel_idx]["dom_idx0"] = dom_idx0

        if is_data:
            events_array[rel_idx:rel_idx+1][COPY_TIME_FIELDS] = (
                events[valid_idx]["start_time"][COPY_TIME_FIELDS]
            )
            events_array[rel_idx]["season"] = finfo_d["season"]
            events_array[rel_idx]["actual_sub_run_id"] = finfo_d["sub_run_id"]
        else:
            events_array[rel_idx]["dataset"] = finfo_d["dataset"]
            events_array[rel_idx]["file_id"] = finfo_d["file_id"]

            events_array[rel_idx]["weight"] = weights[valid_idx]
            if is_noise:
                true_pdg = 0
                true_energy = np.nan
                true_time = np.nan
            else:
                true_pdg = truth[valid_idx]["pdg_encoding"]
                true_energy = truth[valid_idx]["energy"]
                true_time = truth[valid_idx]["time"]
            events_array[rel_idx]["true_pdg"] = true_pdg
            #if abs(true_pdg) >= 128:
            #    print("true_pdg =", true_pdg)
            #    raise ValueError("true_pdg = {}".format(true_pdg))
            events_array[rel_idx]["true_energy"] = true_energy
            events_array[rel_idx]["true_time"] = true_time

            if true_pdg in NEUTRINOS:
                events_array[rel_idx]["true_int"] = truth[valid_idx]["InteractionType"]
            else:
                events_array[rel_idx]["true_int"] = 0

        event_pulses = pulses[valid_idx]

        events_array[rel_idx]["num_hit_doms"] = num_hit_doms = len(event_pulses)
        doms_array = np.empty(shape=num_hit_doms, dtype=DOM_PULSES_IDX_T)

        event_num_pulses = 0
        event_charge = 0.
        for dom_rel_idx, (omkey, dom_pulses) in enumerate(event_pulses):
            dom_num_pulses = len(dom_pulses)
            if dom_num_pulses >= 2**8:
                print("dom_num_pulses =", dom_num_pulses)
                raise ValueError("dom_num_pulses = {}".format(dom_num_pulses))

            dom_charge = np.sum(dom_pulses["charge"])

            event_num_pulses += dom_num_pulses
            event_charge += dom_charge

            doms_array[dom_rel_idx]["string"] = omkey[0]
            doms_array[dom_rel_idx]["om"] = omkey[1]
            doms_array[dom_rel_idx]["pulses_idx0"] = pulses_idx0
            doms_array[dom_rel_idx]["num_pulses"] = dom_num_pulses
            doms_array[dom_rel_idx]["charge"] = dom_charge

            simple_dom_pulses = np.empty(shape=dom_num_pulses, dtype=SIMPLE_PULSE_T)
            simple_dom_pulses["charge"] = dom_pulses["charge"]
            simple_dom_pulses["time"] = dom_pulses["time"]

            pulses_arrays.append(simple_dom_pulses)
            pulses_idx0 += dom_num_pulses

        if event_num_pulses >= 2**32:
            print("event_num_pulses =", event_num_pulses)
            raise ValueError("event_num_pulses = {}".format(event_num_pulses))

        events_array[rel_idx]["num_pulses"] = event_num_pulses
        events_array[rel_idx]["charge"] = event_charge

        doms_arrays.append(doms_array)
        dom_idx0 += num_hit_doms

    events_array[COPY_LINEFIT_DC_DST_FIELDS] = (
        linefit_dc[valid_event_indices][COPY_LINEFIT_DC_SRC_FIELDS]
    )

    doms_array = np.concatenate(doms_arrays)
    pulses_array = np.concatenate(pulses_arrays)

    return events_array, doms_array, pulses_array


def produce_arrays(
    indir,
    outdir,
    pulse_series,
    processes=None,
):
    """
    Parameters
    ----------
    indir
    outdir
    pulse_series
    processes : None or int > 0, optional

    """
    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)

    if processes is None:
        processes = cpu_count()
    assert processes >= 1
    serial = processes == 1

    if not serial:
        pool = Pool(processes=processes)

    # -- Define a closure as callback function -- #

    # Capture the following (must be non-scalar to be persistent between calls
    # of function)
    events_arrays = []
    doms_arrays = []
    pulses_arrays = []

    dom_idx0 = [0]
    pulses_idx0 = [0]

    def concatenate_results(result):
        """Closure"""
        if result is None:
            return

        events_array, doms_array, pulses_array = result

        if len(events_arrays) > 0:
            events_array["dom_idx0"] += dom_idx0[0]
            doms_array["pulses_idx0"] += pulses_idx0[0]

        events_arrays.append(events_array)
        doms_arrays.append(doms_array)
        pulses_arrays.append(pulses_array)

        dom_idx0[0] = events_array[-1]["dom_idx0"] + events_array[-1]["num_hit_doms"]
        pulses_idx0[0] = doms_array[-1]["pulses_idx0"] + doms_array[-1]["num_pulses"]

    # -- Find leaf directories to process -- #

    events_dirpaths = []
    for dirpath, dirs_, files in walk(indir, followlinks=True):
        dirs_.sort(key=nsort_key_func)
        if not "events.npy" in files:  # not a "leaf" dir
            continue
        events_dirpaths.append(dirpath)

    args = tuple()
    for events_dirpath in events_dirpaths:
        kwargs = dict(events_dirpath=events_dirpath, pulse_series=pulse_series)
        if serial:
            result = process_events_dir(*args, **kwargs)
            concatenate_results(result)
        else:
            pool.apply_async(
                process_events_dir,
                args,
                kwargs,
                concatenate_results,
            )

    if not serial:
        pool.close()
        pool.join()

    if len(events_arrays) == 0:
        assert len(doms_arrays) == 0
        assert len(pulses_arrays) == 0
        print("no events found in `indir`:", indir)
        return None

    events_array = np.concatenate(events_arrays)
    doms_array = np.concatenate(doms_arrays)
    pulses_array = np.concatenate(pulses_arrays)

    if outdir is not None:
        np.save(join(outdir, "events_array.npy"), events_array)
        np.save(join(outdir, "doms_array.npy"), doms_array)
        np.save(join(outdir, "pulses_array.npy"), pulses_array)

    return events_array, doms_array, pulses_array


def main(description=__doc__):
    """Script interface to produce_arrays function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--indir")
    parser.add_argument("--pulse-series")
    parser.add_argument("--outdir")
    parser.add_argument(
        "--processes",
        default=None,
        type=int,
        help="""Number of subprocesses to spawn to process directories. Setting
        to 1 avoids use of multiprocessing.Pool""",
    )
    args = parser.parse_args()
    kwargs = vars(args)
    produce_arrays(**kwargs)


if __name__ == "__main__":
    main()
