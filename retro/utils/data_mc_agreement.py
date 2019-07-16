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
    "OUTDIR",
    "LEVEL",
    "PROC_VER",
    "LEVEL_PROC_VER",
    "ROOT_DIR",
    "DATA_ROOT_DIR",
    "GENIE_ROOT_DIR",
    "MUONGUN_ROOT_DIR",
    "MC_NAME_DIRS",
    "DATA_NAME_DIRS",
    "get_stats",
    "get_all_stats",
]

from argparse import ArgumentParser
from collections import OrderedDict
import cPickle as pickle
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

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, mkdir, nsort_key_func


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

OUTDIR = "/home/justin/cowen/oscNext/data_mc_agreement"

LEVEL = 5
PROC_VER = "v01.01"
LEVEL_PROC_VER = "level{}_{}".format(LEVEL, PROC_VER)
ROOT_DIR = "/data/icecube/ana/LE/oscNext/pass2/"

DATA_ROOT_DIR = join(ROOT_DIR, "data", LEVEL_PROC_VER)
GENIE_ROOT_DIR = join(ROOT_DIR, "genie", LEVEL_PROC_VER)
MUONGUN_ROOT_DIR = join(ROOT_DIR, "muongun", LEVEL_PROC_VER)

MC_NAME_DIRS = OrderedDict(
    [
        ("nutau", [join(GENIE_ROOT_DIR, "160000")]),
        ("numu", [join(GENIE_ROOT_DIR, "140000")]),
        ("nue", [join(GENIE_ROOT_DIR, "120000")]),
        ("mu", [join(MUONGUN_ROOT_DIR, "139010")]),
    ]
)

DATA_NAME_DIRS = OrderedDict()
for _yr in range(12, 19):
    _data_year_dirname = "IC86.{:02d}".format(_yr)
    DATA_NAME_DIRS[str(_yr)] = [join(DATA_ROOT_DIR, _data_year_dirname)]


def process_dir(dirpath):
    stats = OrderedDict(
        [
            ("charge_per_hit", []),
            ("charge_per_dom", []),
            ("charge_per_event", []),
            ("hits_per_dom", []),
            ("hits_per_event", []),
            ("doms_per_event", []),  # aka "nchannel"
            ("time_diffs", []),
            ("weight_per_event", []),
            ("weight_per_hit", []),
        ]
    )

    events = np.load(join(dirpath, "events.npy"), mmap_mode="r")
    if len(events) == 0:
        return stats

    sys.stderr.write('processing dir "{}"\n'.format(dirpath))

    l5_bools = events["L5_oscNext_bool"]

    if isfile(join(dirpath, "truth.npy")):
        truth = np.load(join(dirpath, "truth.npy"), mmap_mode="r")
        weights = truth["weight"]
        use_weights = True
    else:
        weights = np.ones(shape=len(events))
        use_weights = False

    pulses = pickle.load(
        file(join(dirpath, "pulses", "{}.pkl".format(PULSE_SERIES_NAME)), "r")
    )

    for l5_bool, event_pulses, weight in zip(l5_bools, pulses, weights):
        if not l5_bool:
            continue

        stats["doms_per_event"].append(len(event_pulses))

        # qtot is sum of charge of all hits on all DOMs
        event_pulses_ = []
        for _, dom_pulses in event_pulses:
            stats["hits_per_dom"].append(len(dom_pulses))
            stats["charge_per_dom"].append(dom_pulses["charge"].sum())
            event_pulses_.append(dom_pulses)
        event_pulses = np.concatenate(event_pulses_)
        charge = event_pulses["charge"]
        stats["charge_per_hit"].append(charge)
        stats["charge_per_event"].append(charge.sum())
        stats["hits_per_event"].append(len(event_pulses))
        first_hit_time = event_pulses["time"].min()
        time_diffs = event_pulses["time"] - first_hit_time
        stats["time_diffs"].append(time_diffs)
        if use_weights:
            stats["weight_per_event"].append(weight)
            stats["weight_per_hit"].append(
                np.full(shape=len(event_pulses), fill_value=weight)
            )

    return stats

def get_stats(dirs):
    if isinstance(dirs, string_types):
        dirs = [dirs]

    pool = Pool()
    dirpaths_to_process = []

    for root_dir in dirs:
        root_dir = expand(root_dir)
        for dirpath, dirs_, files in walk(root_dir, followlinks=True):
            dirs_.sort(key=nsort_key_func)

            if "events.npy" in files:
                dirpaths_to_process.append(dirpath)

    results = pool.map(process_dir, dirpaths_to_process)

    stats = OrderedDict(
        [
            ("charge_per_hit", []),
            ("charge_per_dom", []),
            ("charge_per_event", []),
            ("hits_per_dom", []),
            ("hits_per_event", []),
            ("doms_per_event", []),  # aka "nchannel"
            ("time_diffs", []),
            ("weight_per_event", []),
            ("weight_per_hit", []),
        ]
    )
    for result in results:
        for key in result.keys():
            stats[key].extend(result[key])

    #if len(stats["weight_per_event"]) == 0:
    #    stats.pop("weight_per_event")
    #if len(stats["weight_per_hit"]) == 0:
    #    stats.pop("weight_per_hit")

    for stat_name in stats.keys():
        vals = stats[stat_name]
        if len(vals) == 0:
            stats.pop(stat_name)
            sys.stderr.write(
                'Not using stat "{}" for dirs {}\n'.format(stat_name, dirs)
            )
        elif np.isscalar(vals[0]):
            stats[stat_name] = np.array(vals)
        else:
            stats[stat_name] = np.concatenate(vals)

    return stats


def get_all_stats(overwrite=False):
    """Get stats for all data and MC sets.

    Parameters
    ----------
    overwrite : bool, optional
        Whether to overwrite any existing stats files

    Returns
    -------
    stats : OrderedDict
        Keys are dataset names and values are OrderedDicts containing the stats
        for the corresponding datasets.

    """
    mkdir(OUTDIR)
    stats = OrderedDict()
    for name, dirs in list(MC_NAME_DIRS.items()) +  list(DATA_NAME_DIRS.items()):
        t0 = time.time()
        outfile = join(OUTDIR, "stats_{}.npz".format(name))
        if isfile(outfile) and not overwrite:
            contents = np.load(outfile)
            this_stats = OrderedDict([(k, contents[k]) for k in contents.keys()])
            del contents
            sys.stderr.write(
                'loaded stats for set "{}" from file "{}" ({} sec)\n'.format(
                    name, outfile, time.time() - t0
                )
            )
        else:
            this_stats = get_stats(dirs=dirs)
            np.savez_compressed(outfile, **this_stats)
            sys.stderr.write(
                'saved stats for set "{}" to file "{}" ({} sec)\n'.format(
                    name, outfile, time.time() - t0
                )
            )
        stats[name] = this_stats

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
    #    if


def main(description=__doc__):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing stats *.npz files"
    )
    args = parser.parse_args()
    kwargs = vars(args)
    get_all_stats(**kwargs)


if __name__ == "__main__":
    main()
