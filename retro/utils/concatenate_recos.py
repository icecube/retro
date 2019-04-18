#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name


"""
Recursively search for and aggregate "slc*.{reco}.npy" reco files into a single
"{reco}.npy" file (one file per leaf directory)
"""


from __future__ import absolute_import, division, print_function

__all__ = ["concatenate_recos"]

from collections import OrderedDict
from glob import glob
from os import listdir, walk
from os.path import abspath, dirname, expanduser, expandvars, isdir, join, splitext
import sys

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import join_struct_arrays, nsort


def concatenate_recos(root_dirs, recos="all"):
    """Concatenate events, truth, and recos recursively found within `root_dirs`.

    Parameters
    ----------
    root_dir : str or iterable thereof
    recos : str or iterable thereof

    Returns
    -------
    events : numpy.array
    truth : numpy.array or None
    recos_d : OrderedDict
        Keys are names listed in `recos` and values are the numpy arrays
        containing reconstructions' information.

    """
    orig_root_dirs = root_dirs
    if isinstance(root_dirs, string_types):
        root_dirs = [root_dirs]

    all_dirs = []
    for root_dir in root_dirs:
        all_dirs.extend(nsort(glob(root_dir)))
    root_dirs = []
    for dpath in all_dirs:
        dpath = expanduser(expandvars(dpath))
        if not isdir(dpath):
            raise IOError('Directory does not exist: "{}"'.format(dpath))
        root_dirs.append(dpath)

    if not root_dirs:
        raise ValueError(
            "Found no directories; `root_dirs` = {}".format(orig_root_dirs)
        )

    if recos is None:
        recos = []
    if isinstance(recos, string_types) and recos != "all":
        recos = [recos]

    paths_concatenated = []
    all_events = None
    all_truths = None
    all_recos = OrderedDict()  # [(r, None) for r in recos])
    total_num_events = 0

    for root_dir in root_dirs:
        for dirpath, _, files in walk(root_dir, followlinks=True):
            if "events.npy" not in files:
                continue

            paths_concatenated.append(dirpath)

            events = np.load(join(dirpath, "events.npy"))
            total_num_events += len(events)
            if all_events is None:
                all_events = [events]
            else:
                all_events.append(events)

            if "truth.npy" in files:
                truth = np.load(join(dirpath, "truth.npy"))
                assert len(truth) == len(events)
                if all_truths is None:
                    all_truths = [truth]
                else:
                    all_truths.append(truth)
            elif all_truths is not None:
                raise IOError(
                    '"truth.npy" exists in other directories but not in "{}"'.format(
                        dirpath
                    )
                )

            if recos == "all":
                reco_fnames = sorted(listdir(join(dirpath, "recos")))
                recos = [splitext(reco_fname)[0] for reco_fname in reco_fnames]

            for reco_name in recos:
                reco = np.load(join(dirpath, "recos", reco_name + ".npy"))
                if len(reco) != len(events):
                    raise ValueError(
                        'reco "{}" len = {}, events len = {} in dir "{}"'.format(
                            reco_name, len(reco), len(events), dirpath
                        )
                    )
                if reco_name in all_recos:
                    all_recos[reco_name].append(reco)
                else:
                    all_recos[reco_name] = [reco]

    if total_num_events == 0:
        raise ValueError(
            "Found no events to concatenate at path(s) {}".format(root_dirs)
        )

    # Create a single array with all extracted info; first-level dtype names
    # are all dtype names from `events` plus "truth" and the names of each reco

    events = np.concatenate(all_events)
    if all_truths is not None:
        truth = np.concatenate(all_truths)
        # Re-cast such that dtype is ["truth"][key_n] instead of just [key_n]
        truth = truth.view(dtype=np.dtype([("truth", truth.dtype)]))
    else:
        truth = None

    recos = []
    for rname, rvals in all_recos.items():
        rvals = np.concatenate(rvals)
        # Re-cast such that dtype is ["truth"][key_n] instead of just [key_n]
        rvals = rvals.view(dtype=np.dtype([(rname, rvals.dtype)]))
        recos.append(rvals)

    to_join = [events]
    if truth is not None:
        to_join.append(truth)
    to_join.extend(recos)

    joined_concacatenated_array = join_struct_arrays(to_join)

    return joined_concacatenated_array
