#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Count events in directories.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["count_events", "main"]

from argparse import ArgumentParser
from collections import OrderedDict
from os import walk
from os.path import (
    abspath,
    dirname,
    isfile,
    expanduser,
    expandvars,
    join,
)
import sys

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import FitStatus
from retro.utils.misc import nsort_key_func


def count_events(root_dirs, reco=None, verbosity=0):
    """Count number of events (look for only those passing L5_oscNext_bool, if
    present) in directories.

    Parameters
    ----------
    root_dirs : str or iterable thereof

    reco : str or None, optional
        If a reconstruction name is specified, successes, failures, and
        not-yet-run statistics will be returned for it.

    verbosity : int >= 0, optional

    Returns
    -------
    event_count : int

    """
    if isinstance(root_dirs, string_types):
        root_dirs = [root_dirs]

    if reco is not None and reco.endswith(".npy"):
        reco = reco[:-4]

    event_count = 0
    reco_stats = OrderedDict([("successes", 0), ("failures", 0), ("avg_run_time", 0)])
    for root_dir in root_dirs:
        root_dir = expanduser(expandvars(root_dir))
        for dirpath, dirs, files in walk(root_dir, followlinks=True):
            dirs.sort(key=nsort_key_func)

            if "events.npy" not in files:
                continue

            this_events = np.load(join(dirpath, "events.npy"), mmap_mode="r")

            if "L5_oscNext_bool" in this_events.dtype.names:
                l5_mask = this_events["L5_oscNext_bool"]
                this_events_count = np.count_nonzero(l5_mask)
            else:
                l5_mask = None
                this_events_count = len(this_events)

            if this_events_count == 0:
                continue

            event_count += this_events_count

            if reco is not None:
                reco_fpath = join(dirpath, "recos", reco + ".npy")
                if not isfile(reco_fpath):
                    if verbosity > 1:
                        print(dirpath)
                    continue
                recos = np.load(reco_fpath, mmap_mode="r")
                if l5_mask is not None:
                    recos = recos[l5_mask]
                success_mask = recos["fit_status"] == FitStatus.OK
                this_num_successes = np.count_nonzero(success_mask)
                reco_stats["successes"] += this_num_successes
                if "run_time" in recos.dtype.names:
                    reco_stats["avg_run_time"] += np.sum(
                        recos[success_mask]["run_time"]
                    )
                this_failures = np.count_nonzero(recos["fit_status"] > 0)
                if verbosity > 1 and this_num_successes < this_events_count:
                    print(dirpath)
                reco_stats["failures"] += this_failures

    if reco is not None:
        reco_stats["not_yet_run"] = (
            event_count - reco_stats["successes"] - reco_stats["failures"]
        )
        reco_stats["event_count"] = event_count
        if reco_stats["successes"] > 0:
            reco_stats["avg_run_time"] /= reco_stats["successes"]

    if verbosity > 0:
        if reco is None:
            print("event_count = {}".format(event_count))
        else:
            print("reco_stats:")
            for key, val in reco_stats.items():
                print("  {} = {}".format(key, val))

    if reco is None:
        return event_count
    return reco_stats



def main(description=__doc__):
    """Script interface to `count_events` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--root-dirs",
        required=True,
        nargs="+",
        help="""Directories in which to recursively search for events""",
    )
    parser.add_argument(
        "--reco",
        required=False,
        default=None,
        help="""Count successes, failures, and not-yet-run statistics for a
        particular reconstruction""",
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        required=False,
        default=1,
        action="count",
        help="""verbosity""",
    )

    kwargs = vars(parser.parse_args())
    kwargs["verbosity"] = 1
    count_events(**kwargs)


if __name__ == "__main__":
    main()
