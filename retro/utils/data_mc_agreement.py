#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Produce data-MC (dis)agreement plots
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "LABELS",
    "UNITS",
    "PULSE_SERIES_NAME",
    "ROOT_DIR",
    "MC_NAME_DIRINFOS",
    "DATA_NAME_DIRINFOS",
    "get_stats",
    "get_all_stats",
]

from argparse import ArgumentParser
from collections import OrderedDict
try:
    from collections.abc import Mapping
except ImportError:  # py2 compatibility
    from collections import Mapping
from copy import deepcopy
from itertools import chain
import pickle
from multiprocessing import Pool
from os import walk
from os.path import (
    abspath,
    dirname,
    isfile,
    join,
)
import sys
import time

import numpy as np
from six import string_types
import matplotlib as mpl
#mpl.use("agg")
#mpl.rcParams['font.family'] = 'sans-serif'
#mpl.rcParams['font.sans-serif'] = ['Tahoma']

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.retro_types import PULSE_T
from retro.utils.misc import expand, mkdir, nsort_key_func, wstderr


STATS_PROTO = OrderedDict(
    [
        ("charge_per_hit", []),
        ("charge_per_dom", []),
        ("charge_per_event", []),

        ("hits_per_dom", []),
        ("hits_per_event", []),

        ("doms_per_event", []),  # aka "nchannel"

        #("time_diffs_between_hits", []),
        ("time_diffs_within_dom", []),
        ("time_diffs_within_event", []),

        ("weight_per_event", []),
        ("weight_per_dom", []),
        ("weight_per_hit", []),
    ]
)

LABELS = dict(
    nue=r'GENIE $\nu_e$',
    numu=r'GENIE $\nu_\mu$',
    nutau=r'GENIE $\nu_\tau$',
    mu=r'MuonGun',

    coszen=r'$\cos\theta_{\rm zen}$',
    zenith=r'$\theta_{\rm zen}$',
    azimuth=r'$\phi_{\rm az}$',
)

UNITS = dict(
    x="m",
    y="m",
    z="m",
    time="ns",
    azimuth="rad",
    zenith="rad",
    energy="GeV",
)

PULSE_SERIES_NAME = "SplitInIcePulses"

ROOT_DIR = "/data/icecube/ana/LE/oscNext/pass2/"

MC_NAME_DIRINFOS = OrderedDict(
    [
        ("noise", [
            dict(
                id="888003",
                path=join(ROOT_DIR, "noise", "level5_v01.03", "888003"),
                n_files=5000
            )
        ]),
        ("mu", [
            #dict(
            #    id="139010",
            #    path=join(ROOT_DIR, "muongun", "level5_v01.01", "139010"),
            #    n_files=1000,
            #),
            dict(
                id="139011",
                path=join(ROOT_DIR, "muongun", "level5_v01.03", "139011"),
                #n_files=1775,
                n_files=2996,
            ),
        ]),
        ("nue", [
            dict(
                id="120000",
                path=join(ROOT_DIR, "genie", "level5_v01.03", "120000"),
                n_files=601,
            ),
            #dict(
            #    id="120001",
            #    path=join(ROOT_DIR, "genie", "level5_v01.03", "120001"),
            #    n_files=602,
            #),
            #dict(
            #    id="120004",
            #    path=join(ROOT_DIR, "genie", "level5_v01.03", "120004"),
            #    n_files=602,
            #),
        ]),
        ("numu", [
            dict(
                id="140000",
                path=join(ROOT_DIR, "genie", "level5_v01.03", "140000"),
                n_files=1494,
            ),
            #dict(
            #    id="140001",
            #    path=join(ROOT_DIR, "genie", "level5_v01.03", "140001"),
            #    n_files=1520,
            #),
            #dict(
            #    id="140004",
            #    path=join(ROOT_DIR, "genie", "level5_v01.03", "140004"),
            #    n_files=1520,
            #),
        ]),
        ("nutau", [
            dict(
                id="160000",
                path=join(ROOT_DIR, "genie", "level5_v01.02", "160000"),
                n_files=335,
            ),
        ]),
    ]
)

DATA_N_FILES = {
    #"12": 21142,
    "12": 21141,
    "13": 21128,
    "14": 37305,
    "15": 23831,
    "16": 21939,
    "17": 30499,
    "18": 23589,
}

DATA_NAME_DIRINFOS = OrderedDict([("data", [])])
for _yr in range(12, 19):
    _data_year_dirname = "IC86.{:02d}".format(_yr)
    DATA_NAME_DIRINFOS["data"].append(
        dict(
            id=_data_year_dirname,
            path=join(ROOT_DIR, "data", "level5_v01.03",  _data_year_dirname),
            n_files=DATA_N_FILES[str(_yr)],
        )
    )


def quantize(x, quantum):
    return ((x.astype(np.float64) // quantum) * quantum + quantum / 2).astype(x.dtype)


def quantize_min_q_filter(pulses, qmin=0.4, quantum=0):
    """
    Parameters
    ----------
    pulses : array of dtype retro.retro_types.PULSE_T
    qmin : float >= 0
    quantum : float >= 0
        If bool(quantum) evaluates to False, quantization is disabled.
        Quantization is performed via truncation (verified to yield better
        agreement than via rounding)

    Returns
    -------
    new_pulses : list of ((str, om, pmt) tuple, pulse_array) tuples

    """
    new_pulses = []
    q_bool = np.isfinite(quantum) and quantum > 0
    for om, pulse_array in pulses:
        if qmin > 0:
            mask = pulse_array["charge"] >= qmin
            if np.count_nonzero(mask) == 0:
                continue
            new_pulse_array = deepcopy(pulse_array[mask])
        else:
            new_pulse_array = deepcopy(pulse_array)
        if q_bool:
            new_pulse_array["charge"] = quantize(
                new_pulse_array["charge"], quantum=quantum
            )
        new_pulses.append((om, new_pulse_array))
    return new_pulses


def fixed_charge_filter(pulses, fixed_charge=1.0):
    new_pulses = []
    for om, pulse_array in pulses:
        new_pulse_array = deepcopy(pulse_array)
        new_pulse_array["charge"] = fixed_charge
        new_pulses.append((om, new_pulse_array))
    return new_pulses


def irregular_quantize_min_q_filter(pulses, qmin=0.4, quantum=0.5, clip_above=2.25):
    """
    Parameters
    ----------
    pulses : array of dtype retro.retro_types.PULSE_T
    quantum : scalar > 0
    clip_above : 0 < scalar <= np.inf

    Returns
    -------
    new_pulses : list of ((str, om, pmt) tuple, pulse_array) tuples

    """
    new_pulses = []
    for om, pulse_array in pulses:
        if qmin > 0:
            mask = pulse_array["charge"] >= qmin
            if np.count_nonzero(mask) == 0:
                continue
            new_pulse_array = deepcopy(pulse_array[mask])
        else:
            new_pulse_array = deepcopy(pulse_array)
        new_pulse_array["charge"] = np.clip(
            quantize(new_pulse_array["charge"], quantum=quantum),
            a_min=None,
            a_max=clip_above,
        )
        new_pulses.append((om, new_pulse_array))
    return new_pulses


def do_flaring_dom_analysis(
    omkey,
    pulses,
    geo,
    causal_cut,
    min_charge_fraction,
    max_noncausal_hits,
):
    geo = frame['I3Geometry']
    pos = geo.omgeo[omkey].position
    keyCharge = sum(p.charge for p in pulses[omkey])
    t0 = pulses[omkey][0].time
    totalCharge = keyCharge
    nprior = 0
    for pulses_omkey in pulses.keys():
        if pulses_omkey == omkey:
            continue
        d = (geo.omgeo[pulses_omkey].position - pos).magnitude
        totalCharge += sum(p.charge for p in pulses[pulses_omkey])
        tExp = t0 + d / dataclasses.I3Constants.c_ice
        tRes = pulses[pulses_omkey][0].time - tExp
        isNeighbor = (
            pulses_omkey.string == omkey.string
            and abs(pulses_omkey.om - omkey.om) < 4
        )
        if tRes < causal_cut and isNeighbor:
            nprior += 1

    return (
        (keyCharge / totalCharge) > min_charge_fraction
        and nprior <= max_noncausal_hits
    )


def clean_pulses(pulses):
    """Sort pulses in time-order and remove the first 1% of charge.

    Parameters
    ----------
    pulses : array of dtype PULSE_T
        Pulses recorded by a single DOM

    Returns
    -------
    cleaned_pulses : array of dtype PULSE_T

    """
    sorted_pulses = np.sort(pulses, order=["time"])
    qsum = sorted_pulses["charge"].sum()
    threshold = qsum / 100.
    qsum = 0.
    for i, pulse in enumerate(sorted_pulses):
        qsum += pulse["charge"]
        if qsum >= threshold:
            return sorted_pulses[i:]


def flaring_dom_filter(pulses, charge_threshold):
    chargemap = OrderedDict(
        [(omkey, sum([p.charge for p in v])) for omkey, v in pulses.items()]
    )
    selectedKeys = set(omkey for omkey, q in chargemap.items() if q > charge_threshold)


def pulse_integrator(pulses, quantum, window_len):
    """Find total charge and weighted-average time of pulses (from a single
    PulseSeries, i.e. from within a single DOM) in adjacent time windows.

    The goal of this is to achieve better data/MC agreement where wavedeform
    places small pulse artifacts that are non-physical at different times and
    of different charges for data vs. MC.


    Parameters
    ----------
    pulses : array of dtype PULSE_T

    quantum : scalar > 0

    window_len : scalar, units of ns


    Returns
    -------
    integrated_pulses : array of dtype PULSE_T

    """
    pulses["charge"] = quantize(pulses["charge"], quantum=quantum)

    if len(pulses) <= 1:
        return pulses

    sorted_pulses = np.sort(pulses, order=["time"])

    t_lower, t_upper = sorted_pulses["time"][[0, -1]]
    t_width = t_upper - t_lower
    n_windows = int(np.ceil(t_width / window_len))

    integrated_pulse_charges = []
    integrated_pulse_times = []
    integrated_pulse_widths = []
    for window_i in range(n_windows):
        t0 = t_lower + window_i * window_len
        t1 = t0 + window_len
        if window_i == n_windows - 1:
            mask = (sorted_pulses["time"] >= t0) & (sorted_pulses["time"] <= t1)
        else:
            mask = (sorted_pulses["time"] >= t0) & (sorted_pulses["time"] < t1)
        if not np.count_nonzero(mask):
            continue
        these_pulses = sorted_pulses[mask]
        total_charge = np.sum(these_pulses["charge"])
        weighted_avg_time = np.sum(these_pulses["charge"] * these_pulses["time"]) / total_charge
        integrated_pulse_times.append(weighted_avg_time)
        integrated_pulse_charges.append(total_charge)
        first_pulse, last_pulse = these_pulses[[0, -1]]
        integrated_pulse_widths.append(
            last_pulse["time"] - first_pulse["time"]
            +
            (first_pulse["width"] + last_pulse["width"]) / 2
        )

    if len(integrated_pulse_times) == 0:
        raise ValueError("t_width = {}, pulses = {}".format(t_width, pulses))

    integrated_pulses = np.empty(shape=len(integrated_pulse_times), dtype=PULSE_T)
    integrated_pulses["time"] = integrated_pulse_times
    integrated_pulses["charge"] = integrated_pulse_charges
    integrated_pulses["width"] = integrated_pulse_widths

    return integrated_pulses


def pulse_integration_filter(pulses, quantum=0.05, window_len=1000):
    new_pulses = []
    for omkey, pulses_array in pulses:
        new_pulses.append(
            (
                omkey,
                pulse_integrator(pulses_array, quantum=quantum, window_len=window_len),
            )
        )
    return new_pulses


def process_dir(
    dirpath,
    n_files,
    min_pulses_per_event,
    pulses_filter,
    emax,
    verbosity=0,
):
    """
    Parameters
    ----------
    dirpath : string
    n_files : int > 0
    min_pulses_per_event : int >= 0
    pulses_filter : None or callable, optional
    emax : 0 <= scalar <= np.inf
    verbosity : int >= 0

    Returns
    -------
    stats : OrderedDict
        Keys are taken from STATS_PROTO, values are numpy arrays

    """
    stats = deepcopy(STATS_PROTO)

    events = np.load(join(dirpath, "events.npy"), mmap_mode="r")
    if len(events) == 0:
        return stats
    mask_vals = deepcopy(events["L5_oscNext_bool"])
    if np.count_nonzero(mask_vals) == 0:
        return stats

    if verbosity >= 2:
        wstderr(".")

    if isfile(join(dirpath, "truth.npy")):
        truth = np.load(join(dirpath, "truth.npy"), mmap_mode="r")
        weights = truth["weight"]
        use_weights = True
    else:
        weights = np.ones(shape=len(events))
        use_weights = False

    if np.isfinite(emax) and emax > 0:
        recos = np.load(
            join(dirpath, "recos", "retro_crs_prefit.npy"),
            mmap_mode="r",
        )
        with np.errstate(invalid='ignore'):
            mask_vals &= recos["energy"]["median"] <= emax
        if np.count_nonzero(mask_vals) == 0:
            return stats

    pulses = load_pickle(join(dirpath, "pulses", "{}.pkl".format(PULSE_SERIES_NAME)))

    for mask_val, event_pulses, weight in zip(mask_vals, pulses, weights):
        if not mask_val:
            continue

        if callable(pulses_filter):
            event_pulses = pulses_filter(event_pulses)
            if len(event_pulses) == 0:
                continue

        if use_weights:
            normed_weight = weight / n_files

        # qtot is sum of charge of all hits on all DOMs
        event_pulses_ = []
        tmp_hits_per_dom = []
        tmp_charge_per_dom = []
        tmp_time_diffs_within_dom = []
        tmp_weight_per_dom = []
        for omkey, dom_pulses in event_pulses:
            event_pulses_.append(dom_pulses)
            tmp_hits_per_dom.append(len(dom_pulses))
            tmp_charge_per_dom.append(dom_pulses["charge"].sum())
            #stats["time_diffs_between_hits"].append(
            #    np.concatenate([[0.], np.diff(np.sort(dom_pulses["time"]))])
            #)
            tmp_time_diffs_within_dom.append(dom_pulses["time"] - dom_pulses["time"].min())
            if use_weights:
                tmp_weight_per_dom.append(normed_weight)

        event_pulses = np.concatenate(event_pulses_)
        if len(event_pulses) < min_pulses_per_event:
            continue

        stats["doms_per_event"].append(len(event_pulses))

        stats["hits_per_dom"].extend(tmp_hits_per_dom)
        stats["charge_per_dom"].extend(tmp_charge_per_dom)
        stats["time_diffs_within_dom"].extend(tmp_time_diffs_within_dom)
        if use_weights:
            stats["weight_per_dom"].extend(tmp_weight_per_dom)

        charge = event_pulses["charge"]
        stats["charge_per_hit"].append(charge)
        stats["charge_per_event"].append(charge.sum())
        stats["hits_per_event"].append(len(event_pulses))
        stats["time_diffs_within_event"].append(
            event_pulses["time"] - event_pulses["time"].min()
        )
        if use_weights:
            stats["weight_per_event"].append(normed_weight)
            stats["weight_per_hit"].append(
                np.full(shape=len(event_pulses), fill_value=normed_weight)
            )

    return stats


def get_stats(dirinfo, min_pulses_per_event, processes=None, verbosity=0):
    """
    Parameters
    ----------
    dirinfo : dict
        Must contain keys / vals
            "id" : string
            "path" : string
            "n_files" : int

    min_pulses_per_event : int >= 0

    processes : None or int > 0, optional

    verbosity : int >= 0

    """
    if isinstance(dirinfo, string_types):
        dirinfo = [dirinfo]
    elif isinstance(dirinfo, Mapping):
        dirinfo = [dirinfo]

    #pulses_filter = fixed_charge_filter
    #pulses_filter = quantize_min_q_filter
    #pulses_filter = irregular_quantize_min_q_filter
    #pulses_filter = pulse_integration_filter
    pulses_filter = None

    #emax = 100
    emax = np.inf

    pool = Pool(processes=processes)
    results = []
    for root_dirinfo in dirinfo:
        root_dir = expand(root_dirinfo["path"])
        n_files = root_dirinfo["n_files"]
        for dirpath, dirs_, files in walk(root_dir, followlinks=True):
            dirs_.sort(key=nsort_key_func)
            if "events.npy" in files:
                results.append(
                    pool.apply_async(
                        process_dir,
                        tuple(),
                        dict(
                            dirpath=dirpath,
                            n_files=n_files,
                            pulses_filter=pulses_filter,
                            emax=emax,
                            min_pulses_per_event=min_pulses_per_event,
                            verbosity=verbosity,
                        ),
                    )
                )
    pool.close()
    pool.join()

    stats = deepcopy(STATS_PROTO)
    for result in results:
        result = result.get()
        for key in result.keys():
            stats[key].extend(result[key])

    # Concatenate and cull
    new_stats = OrderedDict()
    for stat_name in stats.keys():
        vals = stats[stat_name]
        if len(vals) == 0:
            if verbosity >= 1:
                wstderr('Not using stat "{}" for dirs {}\n'.format(stat_name, dirinfo))
        elif np.isscalar(vals[0]):
            new_stats[stat_name] = np.array(vals)
        else:
            new_stats[stat_name] = np.concatenate(vals)
    stats = new_stats

    return stats


def get_all_stats(
    outdir,
    min_pulses_per_event,
    overwrite=False,
    only_sets=None,
    processes=None,
    verbosity=0,
):
    """Get stats for all data and MC sets.

    Parameters
    ----------
    outdir : string

    min_pulses_per_event : int >= 0

    overwrite : bool, optional
        Whether to overwrite any existing stats files

    only_sets : string, iterable thereof, or None, optional
        If specified, string(s) must be keys of `MC_NAME_DIRINFOS` and/or
        `DATA_NAME_DIRINFOS` dicts.

    processes : None or int > 0, optional

    verbosity : int >= 0, optional

    Returns
    -------
    stats : OrderedDict
        Keys are dataset names and values are OrderedDicts containing the stats
        for the corresponding datasets.

    """
    outdir = expand(outdir)

    if isinstance(only_sets, string_types):
        only_sets = [only_sets]

    to_process = chain.from_iterable(
        [MC_NAME_DIRINFOS.items(),  DATA_NAME_DIRINFOS.items()]
    )
    if only_sets is not None:
        only_sets = [s.split("/") for s in only_sets]
        new_to_process = []
        for set_name, subsets_list in to_process:
            new_subsets_list = []
            for only_set in only_sets:
                if set_name != only_set[0]:
                    continue
                if len(only_set) == 1:
                    new_subsets_list = subsets_list
                    break
                else:
                    for subset in subsets_list:
                        if subset["id"] == only_set[1]:
                            new_subsets_list.append(subset)
            if len(new_subsets_list) > 0:
                new_to_process.append((set_name, new_subsets_list))
        to_process = new_to_process #((key, val) for key, val in to_process if key in only_sets)
        print(to_process)

    mkdir(outdir)
    stats = OrderedDict()
    for name, dirinfos in to_process:
        t0 = time.time()

        this_stats = OrderedDict()
        for dirinfo in dirinfos:
            augmented_name = "{}.{}".format(name, dirinfo["id"])
            outfile = join(outdir, "stats_{}.npz".format(augmented_name))
            if isfile(outfile) and not overwrite:
                contents = OrderedDict([(k, v) for k, v in np.load(outfile).items()])
                if verbosity >= 1:
                    wstderr(
                        'loaded stats for set "{}" from file "{}" ({} sec)\n'.format(
                            augmented_name, outfile, time.time() - t0
                        )
                    )
            else:
                contents = get_stats(
                    min_pulses_per_event=min_pulses_per_event,
                    dirinfo=dirinfo,
                    processes=processes,
                    verbosity=verbosity,
                )
                #np.savez_compressed(outfile, **contents)
                np.savez(outfile, **contents)
                if verbosity >= 1:
                    wstderr(
                        'saved stats for set "{}" to file "{}" ({} sec)\n'.format(
                            name, outfile, time.time() - t0
                        )
                    )

            if name == "data":
                stats[dirinfo["id"]] = contents
            else:
                for key, vals in contents.items():
                    if key not in this_stats:
                        this_stats[key] = []
                    this_stats[key].append(vals)

            del contents

        if name != "data":
            stats[name] = OrderedDict(
                [(k, np.concatenate(v)) for k, v in this_stats.items()]
            )

    return stats


def plot_distributions():
    all_stats = get_all_stats()
    for name in ["total_mc", "total_data"]:
        all_stats["total_mc"] = OrderedDict(
            [
                (name, []) for name in all_stats.values()[0].keys()
            ]
        )

    #for name, stats in all_stats.items():
    #    if name in MC_NAME_DIRINFOS:
    #        dmc = "total_mc"
    #    else:
    #        dmc = "total_data"

    #        for key


def main(description=__doc__):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--outdir", help="Load existing or save generated .npz files to this dir"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing stats *.npz files"
    )
    parser.add_argument(
        "--only-sets",
        nargs="+",
        #choices=list(MC_NAME_DIRINFOS) + list(DATA_NAME_DIRINFOS),
        default=None,
        help="""Only process a subset of the MC and data sets defined in
        `MC_NAME_DIRINFOS` and `DATA_NAME_DIRINFOS`. Optionally specify subsets
        by separating set from subsets with forward-slash, e.g.:
        data/IC86.12""",
    )
    parser.add_argument(
        "--processes",
        default=None,
        type=int,
        help="""Number of subprocesses to spawn to process directories""",
    )
    parser.add_argument(
        "--min-pulses-per-event",
        default=0,
        type=int,
        help="""Filter events by minimum number of pulses in the event""",
    )
    parser.add_argument(
        "-v", dest="verbosity", action="count", default=0, help="Message verbosity"
    )
    args = parser.parse_args()
    kwargs = vars(args)
    get_all_stats(**kwargs)


if __name__ == "__main__":
    main()
