#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position


"""
Recursively search for and aggregate "slc*.{reco}.npy" reco files into a single
"{reco}.npy" file (one file per leaf directory)
"""


from __future__ import absolute_import, division, print_function

__all__ = [
    "EVENT_ID_FIELDS",
    "get_common_mask",
    "get_common_events",
    "concatenate_recos",
]

from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from os import listdir, walk
from os.path import (
    abspath,
    dirname,
    expanduser,
    expandvars,
    isdir,
    isfile,
    join,
    splitext,
)
import sys

import numpy as np
from six import string_types
import numba

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import join_struct_arrays, nsort_key_func


EVENT_ID_FIELDS = [
    "run_id",
    "sub_run_id",
    "event_id",
    "sub_event_id",
    "start_time",
    "end_time",
]
"""Note that "start_time" and "end_time" are only necessary because poeple are
inconsiderate assholes, overwriting the *_id fields and making them no longer
unique (at least in MC)"""


@numba.jit(
    nopython=True,
    nogil=True,
    parallel=True,
    fastmath=True,
    error_model="numpy",
    cache=True,
)
def get_common_mask(event_ids, common_ids):
    """Create a mask for events IDs that are found in common set of event IDs.

    Parameters
    ----------
    event_ids, common_ids : numpy.arrays
        Arrays dtypes must contain `EVENT_ID_FIELDS` (and with sub-dtypes as
        found in the equality comparison in the below code).

    Returns
    -------
    common_mask : numpy.array of dtype numpy.bool8

    """
    common_mask = np.empty(shape=event_ids.shape, dtype=np.bool8)
    for idx in numba.prange(len(event_ids)):  # pylint: disable=not-an-iterable
        event_id = event_ids[idx]
        event_id_is_common = False
        for common_id in common_ids:
            if (
                event_id["run_id"] == common_id["run_id"]
                and event_id["sub_run_id"] == common_id["sub_run_id"]
                and event_id["event_id"] == common_id["event_id"]
                and event_id["sub_event_id"] == common_id["sub_event_id"]
                and event_id["start_time"]["utc_year"]
                == common_id["start_time"]["utc_year"]
                and event_id["start_time"]["utc_daq_time"]
                == common_id["start_time"]["utc_daq_time"]
                and event_id["end_time"]["utc_year"]
                == common_id["end_time"]["utc_year"]
                and event_id["end_time"]["utc_daq_time"]
                == common_id["end_time"]["utc_daq_time"]
            ):
                event_id_is_common = True
                break
        common_mask[idx] = event_id_is_common
    return common_mask


def get_common_events(arrays):
    """Get only those events from each array that are common to all arrays.

    Parameters
    ----------
    arrays : sequence of numpy.array

    Returns
    -------
    common_arrays : list of numpy.array

    """
    all_ids = []
    ids_dtype = None
    for array in arrays:
        ids = array[EVENT_ID_FIELDS]
        all_ids.append(ids)
        ids_dtype = ids.dtype

    all_ids_sets = [set(ids.tolist()) for ids in all_ids]

    # Enforce that IDs must be unique
    for array_num, (array, ids_set) in enumerate(zip(arrays, all_ids_sets)):
        if len(ids_set) != len(array):
            raise ValueError(
                "array {}: IDs are not unique! (len(ids) = {}) != (len(array) = {})".format(
                    array_num, len(ids_set), len(array)
                )
            )

    common_ids = np.array(sorted(set.intersection(*all_ids_sets)), dtype=ids_dtype)
    print("Number of events common to all arrays:", len(common_ids))

    common = []
    for array_num, (array, ids) in enumerate(zip(arrays, all_ids)):
        common_mask = get_common_mask(event_ids=ids, common_ids=common_ids)
        selected = array[common_mask]
        if len(selected) != len(common_ids):
            raise ValueError(
                "array {}: (len(selected) = {}) != (len(common_ids) = {})".format(
                    array_num, len(selected), len(common_ids)
                )
            )
        common.append(np.sort(selected, order=EVENT_ID_FIELDS))

    return common


def concatenate_recos(root_dirs, recos="all", allow_missing_recos=True):
    """Concatenate events, truth, and recos recursively found within `root_dirs`.

    Parameters
    ----------
    root_dirs : str or iterable thereof
    recos : str or iterable thereof
        Specify "all" to load all recos found in first "events/recos" dir
        encountered. Specify None to load no recos.
    allow_missing_recos : bool

    Returns
    -------
    joined_concacatenated_array : numpy.array

    """
    orig_root_dirs = root_dirs
    if isinstance(root_dirs, string_types):
        root_dirs = [root_dirs]

    all_dirs = []
    for root_dir in root_dirs:
        all_dirs.extend(sorted(glob(root_dir), key=nsort_key_func))
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

    sys.stdout.write("`recos` to extract: {}\n".format(recos))

    paths_concatenated = []
    all_events = None
    all_truths = None
    all_recos = OrderedDict()
    total_num_events = 0
    events_dtype = None

    for root_dir in root_dirs:
        for dirpath, dirs, files in walk(root_dir, followlinks=True):
            # Force directory walking to be ordered in most logical way for our
            # files (we only need consistency, but nice to sort in a sensible
            # way). Note this must be an in-place sort. See ref.
            #   https://stackoverflow.com/a/6670926
            dirs.sort(key=nsort_key_func)

            if "events.npy" not in files:
                continue

            paths_concatenated.append(dirpath)

            events = np.load(join(dirpath, "events.npy"))
            if all_events is None:
                all_events = [events]
                events_dtype = events.dtype
            else:
                if events.dtype != events_dtype:
                    raise TypeError(
                        "events.dtype should be\n{}but dtype found is\n{}".format(
                            events_dtype, events.dtype
                        )
                    )
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
                sys.stdout.write("`recos` specified as 'all'; found {}\n".format(recos))

            for reco_name in recos:
                fpath = join(dirpath, "recos", reco_name + ".npy")
                if not isfile(fpath):
                    if not allow_missing_recos:
                        raise IOError(
                            'Missing reco "{}" file at path "{}"'.format(
                                reco_name, fpath
                            )
                        )
                    if total_num_events > 0 and reco_name in all_recos:
                        raise IOError(
                            'Reco "{}" missing from dir "{}"'.format(reco_name, dirpath)
                        )
                    continue
                reco = np.load(fpath)
                if len(reco) != len(events):
                    raise ValueError(
                        'reco "{}" len = {}, events len = {} in dir "{}"'.format(
                            reco_name, len(reco), len(events), dirpath
                        )
                    )
                if reco_name in all_recos:
                    all_recos[reco_name].append(reco)
                else:
                    if total_num_events > 0:
                        raise IOError(
                            'Reco "{}" found in current dir "{}" but was'
                            " missing in another events directory".format(
                                reco_name, dirpath
                            )
                        )
                    all_recos[reco_name] = [reco]

            total_num_events += len(events)

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


def concatenate_recos_and_save(outfile, **kwargs):
    """Concatenate recos and save to a file.

    Parameters
    ----------
    outfile : str
    **kwargs
        Arguments passed to `concatenate_recos`

    """
    out_array = concatenate_recos(**kwargs)
    np.save(outfile, out_array)


def main():
    """Script interface to `concatenate_recos_and_save` function"""
    parser = ArgumentParser(
        description="""Concatenate events, truth, and reco(s) from one or more
        directories (searching recursively) into a single numpy array"""
    )
    parser.add_argument(
        "-d",
        "--root-dirs",
        required=True,
        nargs="+",
        help="""Directories in which to search for events, truth, and
        associated recos""",
    )
    parser.add_argument(
        "--recos",
        default=None,
        help="""Specify reco names to retrieve, or specify "all" to retrieve
        all recos present (at least those found in the first events/recos
        directory encountered); """,
    )
    parser.add_argument(
        "--allow-missing-recos",
        action="store_true",
        help="""Allow recos explicitly specified by
            --recos <reco_name0> <reco_name1> ..
        to be absent in all found directories (if present in some but absent in
        others, an error will still result).""",
    )
    parser.add_argument(
        "--outfile",
        required=True,
        help="""Results will be written to this ".npy" file.""",
    )
    kwargs = vars(parser.parse_args())
    concatenate_recos_and_save(**kwargs)


if __name__ == "__main__":
    main()
