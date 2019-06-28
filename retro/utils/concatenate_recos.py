#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position


"""
Recursively search for events, truth, and recos .npy files, and aggregate into
a single numpy array.
"""


from __future__ import absolute_import, division, print_function

__all__ = [
    "EVENT_ID_FIELDS",
    "get_common_mask",
    "get_common_events",
    "concatenate_recos",
]

from argparse import ArgumentParser
from collections import Iterable, OrderedDict
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
from retro.utils.misc import expand, join_struct_arrays, mkdir, nsort_key_func


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
        e_id = event_ids[idx]
        event_id_is_common = False
        for c_id in common_ids:
            if (
                e_id["run_id"] == c_id["run_id"]
                and e_id["sub_run_id"] == c_id["sub_run_id"]
                and e_id["event_id"] == c_id["event_id"]
                and e_id["sub_event_id"] == c_id["sub_event_id"]
                and e_id["start_time"]["utc_year"] == c_id["start_time"]["utc_year"]
                and e_id["start_time"]["utc_daq_time"] == c_id["start_time"]["utc_daq_time"]
                and e_id["end_time"]["utc_year"] == c_id["end_time"]["utc_year"]
                and e_id["end_time"]["utc_daq_time"] == c_id["end_time"]["utc_daq_time"]
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
    recos : str or iterable thereof, or None
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
        expected_reco_names = []
    elif isinstance(recos, string_types):
        if recos == "all":
            expected_reco_names = None
        else:
            expected_reco_names = [recos]
    else:
        assert isinstance(recos, Iterable)
        expected_reco_names = list(recos)

    if expected_reco_names:
        for reco_name in expected_reco_names:
            assert isinstance(reco_name, string_types), str(reco_name)

    sys.stdout.write("`recos` to extract: {}\n".format(expected_reco_names))

    paths_with_empty_events_file = []
    paths_with_nonempty_events_file = []
    all_events = []
    all_truths = []
    all_recos = []
    total_num_events = 0
    events_dtype = None
    has_truth = False
    all_found_reco_names = set()

    for root_dir in root_dirs:
        for dirpath, dirs, files in walk(root_dir, followlinks=True):
            # Force directory walking to be ordered in most logical way for our
            # files (we only need consistency, but nice to sort in a sensible
            # way). Note this must be an in-place sort. See ref.
            #   https://stackoverflow.com/a/6670926
            dirs.sort(key=nsort_key_func)

            if "events.npy" not in files:
                continue

            this_events = np.load(join(dirpath, "events.npy"))
            if len(this_events) == 0:
                paths_with_empty_events_file.append(dirpath)
                continue

            paths_with_nonempty_events_file.append(dirpath)

            if events_dtype is None:
                events_dtype = this_events.dtype
            else:
                if this_events.dtype != events_dtype:
                    raise TypeError(
                        '"{}": events.dtype should be\n{}\nbut dtype found'
                        "is\n{}".format(
                            dirpath, events_dtype, this_events.dtype
                        )
                    )

            this_truths = None
            if "truth.npy" in files:
                this_truths = np.load(join(dirpath, "truth.npy"))
                has_truth = True
                if len(this_truths) != len(this_events):
                    raise ValueError(
                        "len(this_truths) = {} != len(this_events) = {}, dir {}"
                        .format(len(this_truths), len(this_events), dirpath)
                    )
            elif has_truth:
                raise IOError(
                    '"truth.npy" exists in other directories but not in "{}"'.format(
                        dirpath
                    )
                )

            if expected_reco_names is None:
                reco_fnames = sorted(listdir(join(dirpath, "recos")))
                this_expected_reco_names = [splitext(reco_fname)[0] for reco_fname in reco_fnames]
            else:
                this_expected_reco_names = expected_reco_names
            all_found_reco_names.update(this_expected_reco_names)

            missing_recos = []
            if this_expected_reco_names:
                this_recos = OrderedDict()
                for reco_name in this_expected_reco_names:
                    fpath = join(dirpath, "recos", reco_name + ".npy")
                    if not isfile(fpath):
                        if not allow_missing_recos:
                            raise ValueError(
                                'Missing reco "{}" in dir "{}"'.format(
                                    reco_name, dirpath
                                )
                            )
                        missing_recos.append(reco_name)
                        break
                    reco = np.load(fpath)
                    if len(reco) != len(this_events):
                        raise ValueError(
                            'reco "{}" len = {}, events len = {} in dir "{}"'.format(
                                reco_name, len(reco), len(this_events), dirpath
                            )
                        )
                    this_recos[reco_name] = reco
            else:
                this_recos = None

            if missing_recos:
                print(
                    'Missing recos {} in dir "{}", skipping'.format(
                        missing_recos, dirpath
                    )
                )

            all_events.append(this_events)
            all_truths.append(this_truths)
            all_recos.append(this_recos)
            total_num_events += len(this_events)

    # -- Loop through, validating the above -- #

    all_dirs_concatenated = []
    all_events_to_process = []
    all_truths_to_process = []
    all_recos_to_process = OrderedDict()

    for dirpath, this_events, this_truths, this_recos in zip(
        paths_with_nonempty_events_file,
        all_events,
        all_truths,
        all_recos,
    ):
        if len(all_found_reco_names) > 0:
            if this_recos is None or set(this_recos.keys()) != all_found_reco_names:
                if allow_missing_recos:
                    continue
                else:
                    raise ValueError('"{}"'.format(dirpath))

            for reco_name, reco_vals in this_recos.items():
                if reco_name not in all_recos_to_process:
                    all_recos_to_process[reco_name] = []
                all_recos_to_process[reco_name].append(reco_vals)

        if has_truth:
            if this_truths is None:
                raise ValueError('missing truth in dir "{}"'.format(dirpath))
            else:
                all_truths_to_process.append(this_truths)

        all_events_to_process.append(this_events)
        all_dirs_concatenated.append(dirpath)

    if len(all_events_to_process) == 0:
        raise ValueError(
            "Found no events to concatenate recursively from path(s) {}".format(
                root_dirs
            )
        )

    # Create a single array with all extracted info; first-level dtype names
    # are all dtype names from `events` plus "truth" and the names of each reco

    events = np.concatenate(all_events_to_process)
    if has_truth:
        truth = np.concatenate(all_truths_to_process)
        # Re-cast such that dtype is ["truth"][key_n] instead of just [key_n]
        truth = truth.view(dtype=np.dtype([("truth", truth.dtype)]))
    else:
        truth = None

    if len(all_found_reco_names) == 0:
        recos = None
    else:
        recos = []
        for reco_name, reco_vals in all_recos_to_process.items():
            reco_vals = np.concatenate(reco_vals)
            # Re-cast such that dtype is ["truth"][key_n] instead of just [key_n]
            reco_vals = reco_vals.view(dtype=np.dtype([(reco_name, reco_vals.dtype)]))
            recos.append(reco_vals)

    to_join = [events]
    if truth is not None:
        to_join.append(truth)
    if recos is not None:
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
    outfile = expand(outfile)
    out_array = concatenate_recos(**kwargs)
    outdir = dirname(outfile)
    if not isdir(outdir):
        mkdir(outdir)
    np.save(outfile, out_array)
    sys.stdout.write('Saved concatenated array to "{}"\n'.format(outfile))


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
        nargs="+",
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
