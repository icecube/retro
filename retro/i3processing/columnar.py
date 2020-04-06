#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order, import-outside-toplevel


"""
Extract information on events into columnar storage (npy arrays)
"""


from __future__ import absolute_import, division, print_function

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2020 Justin L. Lanfranchi

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
    "RUN_DIR_RE",
    "IC_SEASON_DIR_RE",
    "CATEGORY_INDEX_POSTFIX",
    "LEGAL_ARRAY_NAMES",
    "extract_run",
    "combine_season_runs",
    "find_array_paths",
    "construct_arrays",
    "index_and_concatenate_arrays",
    "load_contained_paths",
    "run_icetray_converter",
    "ConvertI3ToNumpy",
]

from argparse import ArgumentParser
from collections import OrderedDict

try:
    from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
except ImportError:
    from collections import Mapping, MutableMapping, MutableSequence, Sequence
from glob import glob
from multiprocessing import Pool, cpu_count
from os import listdir, walk
from os.path import abspath, basename, dirname, isdir, isfile, join, splitext
import re
from shutil import rmtree
import sys
from tempfile import mkdtemp
import time

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import retro_types as rt
from retro.utils.misc import expand, mkdir, nsort_key_func
from retro.i3processing.extract_common import (
    DATA_GCD_FNAME_RE,
    OSCNEXT_I3_FNAME_RE,
    dict2struct,
    maptype2np,
)

# TODO: optional npz compression of key dirs


RUN_DIR_RE = re.compile(r"(?P<pfx>Run)?(?P<run>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC "run" dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""

IC_SEASON_DIR_RE = re.compile(
    r"((IC)?(?P<detector>[0-9]+)\.)?(?P<year>[0-9]+)", flags=re.IGNORECASE
)
"""Matches data season dirs, e.g. 'IC86.11' or 'IC86.2011'"""

CATEGORY_INDEX_POSTFIX = "__scalar_index.npy"
"""Category scalar index files are named {category}{CATEGORY_INDEX_POSTFIX}"""

LEGAL_ARRAY_NAMES = ("data", "index", "valid")
"""Array names produced / read as data containers within "items" (the items
extracted from an I3 file)"""


def compress(path, keep=False, keys=None):
    path = expand(path)
    assert isdir(path)
    if isinstance(keys, string_types):
        keys = [keys]
    print(path, keep, keys)

    array_fnames = {n: "{}.npy".format(n) for n in LEGAL_ARRAY_NAMES}

    for dirpath, dirs, files in walk(path):
        dirs.sort(key=nsort_key_func)
        if "data.npy" not in files or dirs or set(files).difference(array_fnames.values()):
            continue

        if keys is not None and basename(dirpath) not in keys:
            continue

        print("compressing {}".format(dirpath))
        array_d = OrderedDict()
        for array_name, array_fname in array_fnames.items():
            if array_fname in files:
                array_d[array_name] = np.load(join(dirpath, array_fname))
        np.savez_compressed(dirpath + ".npz", **array_d)
        del dirs[:]
        del files[:]
        if not keep:
            print("removing", dirpath)
            rmtree(dirpath)


def extract_run(
    path,
    outdir=None,
    tempdir=None,
    gcd=None,
    keys=None,
    overwrite=False,
    mmap=True,
    keep_tempfiles_on_fail=False,
    procs=cpu_count(),
):
    """E.g. .. ::

        data/level7_v01.04/IC86.14/Run00125177
        genie/level7_v01.04/140000

    Note that what can be considered "subruns" for both data and MC are
    represented as files in both, at least for this version of oscNext.

    Parameters
    ----------
    path : str
    outdir : str
    tempdir : str or None, optional
        Intermediate arrays will be written to this directory.
    gcd : str or None, optional
    keys : str, iterable thereof, or None; optional
    overwrite : bool, optional
    mmap : bool, optional
    keep_tempfiles_on_fail : bool, optional

    """
    path = expand(path)
    assert isdir(path), path

    outdir = expand(outdir)
    if tempdir is None:
        tempdir = "/tmp"
    tempdir = expand(tempdir)

    if isinstance(keys, string_types):
        keys = [keys]

    parent_dirs = [path]
    for _ in range(3):
        parent_dirs.insert(0, dirname(parent_dirs[0]))
    rel_dirs = [basename(d) for d in parent_dirs]

    match = RUN_DIR_RE.match(basename(path))
    assert match, 'path not a run directory? "{}"'.format(basename(path))
    groupdict = match.groupdict()

    is_data = groupdict["pfx"] is not None
    is_mc = not is_data
    run_str = groupdict["run"]
    run_int = int(groupdict["run"].lstrip("0"))

    if is_mc:
        print("is_mc")
        assert isinstance(gcd, string_types) and isfile(expand(gcd)), str(gcd)
        gcd = expand(gcd)
        rel_subdir = join(*rel_dirs[-3:])
    else:
        print("is_data")
        if gcd is None:
            gcd = path
        assert isinstance(gcd, string_types)
        gcd = expand(gcd)
        if not isfile(gcd):
            assert isdir(gcd)
            # TODO: use DATA_GCD_FNAME_RE
            gcd = glob(join(gcd, "*Run{}*GCD*.i3*".format(run_str)))
            assert len(gcd) == 1, gcd
            gcd = expand(gcd[0])
        rel_subdir = join(*rel_dirs[-4:])

    full_outdir = join(outdir, rel_subdir)
    full_tempdir = join(tempdir, rel_subdir)

    print("full_outdir:", full_outdir)
    mkdir(full_outdir)
    print("full_tempdir:", full_tempdir)
    mkdir(full_tempdir)

    if keys is not None:
        if not overwrite:
            existing_keys = set((x.split(".npz")[0] for x in listdir(full_outdir)))
            redundant_keys = existing_keys.intersection(keys)
            if redundant_keys:
                print("will not extract existing keys:", sorted(redundant_keys))
                keys = [k for k in keys if k not in redundant_keys]
        if is_data:
            mc_only_keys = set(["I3MCWeightDict", "I3MCTree", "I3GENIEResultDict"])
            invalid_keys = mc_only_keys.intersection(keys)
            if invalid_keys:
                print(
                    "MC-only keys {} were specified but this is data, so these will be skipped.".format(
                        sorted(invalid_keys)
                    )
                )
            keys = [k for k in keys if k not in mc_only_keys]
        print("keys remaining to extract:", keys)

        if len(keys) == 0:
            print("nothing to do!")
            return

    subrun_filepaths = []
    for basepath in sorted(listdir(path), key=nsort_key_func):
        match = OSCNEXT_I3_FNAME_RE.match(basepath)
        if not match:
            continue
        groupdict = match.groupdict()
        assert int(groupdict["run"]) == run_int
        subrun_int = int(groupdict["subrun"])
        subrun_filepaths.append((subrun_int, join(path, basepath)))
    # Ensure sorting by subrun
    subrun_filepaths.sort()

    t0 = time.time()
    if is_mc:
        arrays = OrderedDict()
        requests = OrderedDict()


        subrun_tempdirs = []
        pool = None
        if procs > 1:
            pool = Pool(procs)
        try:
            for subrun, fpath in subrun_filepaths:
                print(fpath)
                paths = [gcd, fpath]

                subrun_tempdir = join(full_tempdir, "subrun{}".format(subrun))
                mkdir(subrun_tempdir)
                subrun_tempdirs.append(subrun_tempdir)

                kw = dict(paths=paths, outdir=subrun_tempdir, keys=keys)
                if procs == 1:
                    arrays[subrun] = run_icetray_converter(**kw)
                else:
                    requests[subrun] = pool.apply_async(
                        run_icetray_converter,  # func
                        tuple(),  # args
                        kw,  # kwds
                        # callback
                        # error_callback
                    )
            if procs > 1:
                for key, result in requests.items():
                    arrays[key] = result.get()

            index_and_concatenate_arrays(
                arrays, category_name="subrun", outdir=full_outdir, mmap=mmap,
            )

        except Exception:
            if not keep_tempfiles_on_fail:
                for subrun_tempdir in subrun_tempdirs:
                    try:
                        rmtree(subrun_tempdir)
                    except Exception as err:
                        print(err)

        else:
            for subrun_tempdir in subrun_tempdirs:
                try:
                    rmtree(subrun_tempdir)
                except Exception as err:
                    print(err)

        finally:
            if pool is not None:
                pool.close()
                pool.join()

    else:  # is_data
        paths = [gcd] + [fpath for _, fpath in subrun_filepaths]
        run_icetray_converter(paths=paths, outdir=full_outdir, keys=keys)

    print("{} s to extract run path {}".format(time.time() - t0, path))


def combine_season_runs(path, outdir, keys=None, mmap=True):
    """
    Parameters
    ----------
    path : str
        IC86.XX season directory that contains already-extracted arrays

    outdir : str
        Store concatenated arrays to this directory

    keys : str, iterable thereof, or None; optional
        Only preserver these keys. If None, preserve all keys found in all
        subpaths

    mmap : bool
        Note that if `mmap` is True, ``load_contained_paths`` will be called
        with `inplace=False` or else too many open files will result

    """
    path = expand(path)
    assert isdir(path), str(path)
    outdir = expand(outdir)

    path_base = basename(path)
    match = IC_SEASON_DIR_RE.match(path_base)
    if not match:
        raise ValueError('path "{}" does not look like a season dir'.format(path))

    run_dirs = []
    for subname in sorted(listdir(path), key=nsort_key_func):
        subpath = join(path, subname)
        if not isdir(subpath):
            continue
        match = RUN_DIR_RE.match(subname)
        if not match:
            continue
        groupdict = match.groupdict()
        run_str = groupdict["run"]
        run_int = int(run_str.lstrip("0"))
        run_dirs.append((run_int, subpath))
    # Ensure sorting by numerical run number
    run_dirs.sort()

    print("{} run dirs found".format(len(run_dirs)))

    arrays = OrderedDict()
    existing_category_indexes = OrderedDict()

    for run_int, run_dir in run_dirs:
        arrays[run_int], csi = find_array_paths(run_dir)
        if csi:
            existing_category_indexes[run_int] = csi

    mkdir(outdir)

    index_and_concatenate_arrays(
        category_array_map=arrays,
        existing_category_indexes=existing_category_indexes,
        category_name="run",
        category_dtype=np.uint32,  # see retro_types.I3EVENTHEADER_T
        outdir=outdir,
        mmap=mmap,
    )


def find_array_paths(path):
    """
    Parameters
    ----------
    path : str
        Path to directory containing columnar array "keys" (directories like
        "I3EventHeader" or numpy zip archives like "I3EventHeader.npz") and
        possibly category scalar indices ("<category>__scalar_index.npy" files)

    Returns
    -------
    arrays
    category_indexes

    """
    path = expand(path)
    assert isdir(path), str(path)

    arrays = OrderedDict()
    category_indexes = OrderedDict()

    unrecognized = []

    for name in sorted(listdir(path), key=nsort_key_func):
        subpath = join(path, name)

        if isfile(subpath):
            if name.endswith(CATEGORY_INDEX_POSTFIX):
                category = name[: -len(CATEGORY_INDEX_POSTFIX)]
                category_indexes[category] = subpath
                continue

            if name.endswith(".npz"):
                array_d = OrderedDict()
                npz = np.load(subpath)
                try:
                    contents = set(npz.keys())
                    for array_name in LEGAL_ARRAY_NAMES:
                        if array_name in contents:
                            array_d[array_name] = npz[array_name]
                        contents.remove(array_name)
                finally:
                    npz.close()

                for array_name in contents:
                    unrecognized.append(subpath + "/" + array_name)

                if array_d:
                    arrays[name] = array_d
                else:
                    unrecognized.append(subpath)

                continue

        if not isdir(subpath):
            unrecognized.append(subpath)
            continue

        array_d = OrderedDict()
        contents = set(listdir(subpath))
        for array_name in LEGAL_ARRAY_NAMES:
            fname = array_name + ".npy"
            if fname in contents:
                array_d[array_name] = join(subpath, fname)
                contents.remove(fname)

        for subname in sorted(contents):
            unrecognized.append(join(subpath, subname))

        if array_d:
            arrays[name] = array_d

    if unrecognized:
        print("WARNING: Unrecognized paths ignored: {}".format(unrecognized))

    return arrays, category_indexes


def construct_arrays(data, delete_while_filling=False, outdir=None):
    """Construct arrays to collect same-key scalars / vectors across frames

    Parameters
    ----------
    data : dict or sequence thereof
    delete_while_filling : bool
    outdir : str

    Returns
    -------
    arrays : dict

    """
    if isinstance(data, Mapping):
        data = [data]

    if isinstance(outdir, string_types):
        outdir = expand(outdir)
        mkdir(outdir)

    # Get type and size info

    scalar_dtypes = {}
    vector_dtypes = {}

    num_frames = len(data)
    for frame_d in data:
        # Must get all vector values for all frames to get both dtype and
        # total length, but only need to get a scalar value once to get its
        # dtype
        for key in set(frame_d.keys()).difference(scalar_dtypes.keys()):
            val = frame_d[key]
            # if val is None:
            #    continue
            dtype = val.dtype

            if np.isscalar(val):
                scalar_dtypes[key] = dtype
            else:
                if key not in vector_dtypes:
                    vector_dtypes[key] = [0, dtype]  # length, type
                vector_dtypes[key][0] += len(val)

    # Construct empty arrays

    scalar_arrays = {}
    vector_arrays = {}

    scalar_arrays_paths = {}
    vector_arrays_paths = {}

    for key, dtype in scalar_dtypes.items():
        # Until we know we need one (i.e., when an event is missing this
        # `key`), the "valid" mask array is omitted
        if outdir is not None:
            dpath = join(outdir, key)
            mkdir(dpath)
            data_array_path = join(dpath, "data.npy")
            scalar_arrays_paths[key] = dict(data=data_array_path)
            data_array = np.lib.format.open_memmap(
                data_array_path, mode="w+", shape=(num_frames,), dtype=dtype
            )
        else:
            data_array = np.empty(shape=(num_frames,), dtype=dtype)
        scalar_arrays[key] = dict(data=data_array)

    # `vector_arrays` contains "data" and "index" arrays.
    # `index` has the same number of entries as the scalar arrays,
    # and each entry points into the corresponding `data` array to
    # determine which vector data correspond to this scalar datum

    for key, (length, dtype) in vector_dtypes.items():
        if outdir is not None:
            dpath = join(outdir, key)
            mkdir(dpath)
            data_array_path = join(dpath, "data.npy")
            index_array_path = join(dpath, "index.npy")
            vector_arrays_paths[key] = dict(
                data=data_array_path, index=index_array_path
            )
            data_array = np.lib.format.open_memmap(
                data_array_path, mode="w+", shape=(length,), dtype=dtype
            )
            index_array = np.lib.format.open_memmap(
                index_array_path, mode="w+", shape=(num_frames,), dtype=rt.START_STOP_T,
            )
        else:
            data_array = np.empty(shape=(length,), dtype=dtype)
            index_array = np.empty(shape=(num_frames,), dtype=rt.START_STOP_T)
        vector_arrays[key] = dict(data=data_array, index=index_array)

    # Fill the arrays

    for frame_idx, frame_d in enumerate(data):
        for key, array_d in scalar_arrays.items():
            val = frame_d.get(key, None)
            if val is None:
                if "valid" not in array_d:
                    if outdir is not None:
                        dpath = join(outdir, key)
                        valid_array_path = join(dpath, "valid.npy")
                        scalar_arrays_paths[key]["valid"] = valid_array_path
                        valid_array = np.lib.format.open_memmap(
                            valid_array_path,
                            mode="w+",
                            shape=(num_frames,),
                            dtype=np.bool8,
                        )
                        valid_array[:] = True
                    else:
                        valid_array = np.ones(shape=(num_frames,), dtype=np.bool8)
                    array_d["valid"] = valid_array
                array_d["valid"][frame_idx] = False
            else:
                array_d["data"][frame_idx] = val
                if delete_while_filling:
                    del frame_d[key]

        for key, array_d in vector_arrays.items():
            index = array_d["index"]
            if frame_idx == 0:
                prev_stop = 0
            else:
                prev_stop = index[frame_idx - 1]["stop"]

            start = int(prev_stop)

            val = frame_d.get(key, None)
            if val is None:
                index[frame_idx] = (start, start)
            else:
                length = len(val)
                stop = start + length
                index[frame_idx] = (start, stop)
                array_d["data"][start:stop] = val[:]
                if delete_while_filling:
                    del index, frame_d[key]

    arrays = scalar_arrays
    arrays.update(vector_arrays)

    arrays_paths = scalar_arrays_paths
    arrays_paths.update(vector_arrays_paths)

    if outdir is None:
        return arrays

    del arrays, scalar_arrays, vector_arrays

    return arrays_paths


def index_and_concatenate_arrays(
    category_array_map,
    existing_category_indexes=None,
    category_name=None,
    category_dtype=None,
    outdir=None,
    mmap=True,
):
    """A given scalar array might or might not be present in each tup


    Parameters
    ----------
    category_array_map : OrderedDict
        Keys are the categories (e.g., run number or subrun number), values
        are array dicts and/or paths to Numpy .npz files containing these.
        An array dict must contain the key "data"; if vector data, it has to
        have key "index"; and, optionally, it has key "valid". Values are either Numpy
        arrays, or a string path to the array on disk (to be loaded via np.load).

    existing_category_indexes : mapping of mappings or None, optional

    category_name : str, optional
        Name of the category being indexed, i.e., keys in `category_array_map`.
        This is used both to formulate the structured dtype used for the index
        and to formulate the file (if `outdir` is specified) in which to save
        the category index.

    category_dtype : numpy.dtype or None, optional
        If None, use the type Numpy infers, found via .. ::

            category_dtype = np.array(list(category_array_map.keys())).dtype

    outdir : str or None, optional
        If specified, category index and arrays are written to disk within this
        directory


    Returns
    -------
    category_indexes : dict
        Minimally contains key `category_name` with value a
        shape-(num_categories,) numpy ndarray of custom dtype .. ::

            [(category_name, category_dtype), ("index", retro_types.START_STOP_T)]

    arrays : dict of dicts containing arrays

    """
    if existing_category_indexes:
        raise NotImplementedError(
            "concatenating existing_category_indexes not implemented"
        )

    if category_name is None:
        category_name = "category"

    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)

    load_contained_paths_kw = dict(mmap=mmap, inplace=not mmap)

    # Datatype of each data array (same key must have same dtype regardless
    # of which category)
    key_dtypes = OrderedDict()

    # All scalar data arrays and vector index arrays in one category must have
    # same length as one another; record this length for each category
    category_scalar_array_lengths = OrderedDict()

    # scalar data, scalar valid, and vector index arrays will have this length
    total_scalar_length = 0

    # vector data has different total length for each item
    total_vector_lengths = OrderedDict()

    # Record any keys that, for any category, already have a valid array
    # created, as these keys will require valid arrays to be created and
    # filled
    keys_with_valid_arrays = set()

    # Record keys that contain vector data
    vector_keys = set()

    # Get and validate metadata about arrays

    for n, (category, array_dicts) in enumerate(category_array_map.items()):
        scalar_array_length = None
        for key, array_d in array_dicts.items():
            array_d = load_contained_paths(array_d, **load_contained_paths_kw)

            data = array_d["data"]
            valid = array_d.get("valid", None)
            index = array_d.get("index", None)

            is_scalar = index is None

            if scalar_array_length is None:
                if is_scalar:
                    scalar_array_length = len(data)
                else:
                    scalar_array_length = len(index)
            elif is_scalar and len(data) != scalar_array_length:
                raise ValueError(
                    "category={}, key={}, ref len={}, this len={}".format(
                        category, key, scalar_array_length, len(data)
                    )
                )

            if valid is not None:
                keys_with_valid_arrays.add(key)

                if len(valid) != scalar_array_length:
                    raise ValueError(
                        "category={}, key={}, ref len={}, this len={}".format(
                            category, key, scalar_array_length, len(valid)
                        )
                    )

            if index is not None:
                vector_keys.add(key)
                if key not in total_vector_lengths:
                    total_vector_lengths[key] = 0
                total_vector_lengths[key] += len(data)

                if len(index) != scalar_array_length:
                    raise ValueError(
                        "category={}, key={}, ref len={}, this len={}".format(
                            category, key, scalar_array_length, len(index)
                        )
                    )

            dtype = data.dtype
            existing_dtype = key_dtypes.get(key, None)
            if existing_dtype is None:
                key_dtypes[key] = dtype
            elif dtype != existing_dtype:
                raise TypeError(
                    "category={}, key={}, dtype={}, existing_dtype={}".format(
                        category, key, dtype, existing_dtype
                    )
                )

        if scalar_array_length is None:
            scalar_array_length = 0

        category_scalar_array_lengths[category] = scalar_array_length
        total_scalar_length += scalar_array_length
        print(
            'category {}, {}={}: scalar array len={}, total scalar array len={}'.format(
                n, category_name, category, scalar_array_length, total_scalar_length
            )
        )

    # Create the index

    # Use simple numpy array for now for ease of working with dtypes, ease and
    # consistency in saving to disk; this can be expaned to a dict for easy
    # key/value acces or numba.typed.Dict for direct use in Numba

    categories = np.array(list(category_array_map.keys()), dtype=category_dtype)
    category_dtype = categories.dtype
    category_index_dtype = np.dtype(
        [(category_name, category_dtype), ("index", rt.START_STOP_T)]
    )
    if outdir is not None:
        category_index = np.lib.format.open_memmap(
            join(outdir, category_name + CATEGORY_INDEX_POSTFIX),
            mode="w+",
            shape=(len(categories),),
            dtype=category_index_dtype,
        )
    else:
        category_index = np.empty(shape=(len(categories),), dtype=category_index_dtype)

    # Populate the category index

    start = 0
    for i, (category, array_length) in enumerate(
        zip(categories, category_scalar_array_lengths.values())
    ):
        stop = start + array_length
        value = np.array([(start, stop)], dtype=rt.START_STOP_T)[0]
        category_index[i] = (category, value)
        start = stop

    # Record keys that are missing in one or more categories

    all_keys = set(key_dtypes.keys())
    keys_with_missing_data = set()
    for category, array_dicts in category_array_map.items():
        keys_with_missing_data.update(all_keys.difference(array_dicts.keys()))

    # Create and populate `data` arrays and any necessary `valid` arrays

    # N.b. missing vector arrays DO require valid array so that the resulting
    # "index" array (which spans all categories) has the same number of
    # elements as scalar arrays

    keys_requiring_valid_array = set.union(
        keys_with_missing_data, keys_with_valid_arrays
    )

    concatenated_arrays = OrderedDict()
    for key, dtype in key_dtypes.items():
        if key in vector_keys:
            data_length = total_vector_lengths[key]
        else:
            data_length = total_scalar_length

        # Create big data array

        if outdir is not None:
            dpath = join(outdir, key)
            mkdir(dpath)
            data = np.lib.format.open_memmap(
                join(dpath, "data.npy"), mode="w+", shape=(data_length,), dtype=dtype
            )
        else:
            data = np.empty(shape=(data_length,), dtype=dtype)

        # Create big valid array if needed

        valid = None
        if key in keys_requiring_valid_array:
            if outdir is not None:
                dpath = join(outdir, key)
                mkdir(dpath)
                valid = np.lib.format.open_memmap(
                    join(dpath, "valid.npy"),
                    mode="w+",
                    shape=(total_scalar_length,),
                    dtype=np.bool8,
                )
            else:
                valid = np.empty(shape=(total_scalar_length,), dtype=np.bool8)

        # Create big index array if vector data

        index = None
        if key in vector_keys:
            if outdir is not None:
                dpath = join(outdir, key)
                mkdir(dpath)
                index = np.lib.format.open_memmap(
                    join(dpath, "index.npy"),
                    mode="w+",
                    shape=(total_scalar_length,),
                    dtype=rt.START_STOP_T,
                )
            else:
                index = np.empty(shape=(total_scalar_length,), dtype=np.bool8)

        # Fill chunks of the big arrays from each category

        vector_start = vector_stop = 0
        for category, array_dicts in category_array_map.items():
            scalar_start, scalar_stop = category_index[
                category_index[category_name] == category
            ][0]["index"]

            key_arrays = array_dicts.get(key, None)
            if key_arrays is None:
                valid[scalar_start:scalar_stop] = False
                continue

            key_arrays = load_contained_paths(key_arrays, **load_contained_paths_kw)
            data_ = key_arrays["data"]
            if key not in vector_keys:  # scalar data
                data[scalar_start:scalar_stop] = data_
            else:  # vector data
                # N.b.: copy values to a new array in memory, necessary if
                # mmaped file AND necessary because we don't want to modify
                # original array
                index_ = np.copy(key_arrays["index"])

                vector_stop = vector_start + len(data_)

                if vector_start != 0:
                    index_["start"] += vector_start
                    index_["stop"] += vector_start
                index[scalar_start:scalar_stop] = index_

                data[vector_start:vector_stop] = data_

                vector_start = vector_stop

            valid_ = key_arrays.get("valid", None)
            if valid_ is not None:
                valid[scalar_start:scalar_stop] = valid_
            elif valid is not None:
                valid[scalar_start:scalar_stop] = True

        concatenated_arrays[key] = OrderedDict()
        concatenated_arrays[key]["data"] = data
        if index is not None:
            concatenated_arrays[key]["index"] = index
        if valid is not None:
            concatenated_arrays[key]["valid"] = valid

    category_indexes = OrderedDict()
    category_indexes[category_name] = category_index
    # TODO: put concatenated existing_category_indexes here, too

    return category_indexes, concatenated_arrays


def load_contained_paths(obj, inplace=False, mmap=False):
    """If `obj` or any sub-element of `obj` is a (string) path to a file, load
    the file and replace that element with the object loaded from the file.

    Unhandled containers or objects (whether that is `obj` itself or child
    objects within `obj`) are simply returned, unmodified.

    Parameters
    ----------
    obj

    inplace: bool, optional
        Only valid if all objects and sub-objects that are containers are
        mutable

    mmap : bool, optional
        Load numpy ".npy" files memory mapped. Only applies to ".npy" for now,
        `mmap` is ignored for all other files.

    Returns
    -------
    obj
        If the input `obj` is a container type, the `obj` returned is the same
        object (if inplace=True) or a new object (if inplace=False) but with
        string paths replaced by the contents of the files they refer to. Note
        that all mappings (and .npz file paths) are converted to OrderedDict,
        where a conversion is necessary.

    """
    # Record kwargs beyond `obj` for recursively calling
    my_kwargs = dict(inplace=inplace, mmap=mmap)

    if isinstance(obj, string_types):  # numpy strings evaluate False
        if isfile(obj):
            _, ext = splitext(obj)
            if ext == ".npy":
                obj = np.load(obj, mmap_mode="r" if mmap else None)
            elif ext == ".npz":
                npz = np.load(obj)
                try:
                    obj = OrderedDict(npz.items())
                finally:
                    npz.close()
            # TODO : other file types?

    elif isinstance(obj, Mapping):
        if inplace:
            assert isinstance(obj, MutableMapping)
            out_d = obj
        else:
            out_d = OrderedDict()
        for key in obj.keys():
            out_d[key] = load_contained_paths(obj[key], **my_kwargs)
        obj = out_d

    elif isinstance(obj, Sequence):  # numpy ndarrays evaluate False
        if inplace:
            assert isinstance(obj, MutableSequence)
            for i, val in enumerate(obj):
                obj[i] = load_contained_paths(val, **my_kwargs)
        else:
            obj = type(obj)(load_contained_paths(val, **my_kwargs) for val in obj)

    return obj


def run_icetray_converter(paths, outdir, keys):
    """Function to be called by subprocesses (i.e., run in parallel)"""
    from I3Tray import I3Tray

    converter = ConvertI3ToNumpy()

    tray = I3Tray()
    tray.AddModule(_type="I3Reader", _name="reader", FilenameList=paths)
    tray.Add(_type=converter, _name="ConvertI3ToNumpy", keys=keys)
    tray.Execute()
    tray.Finish()

    arrays = converter.finalize_icetray(outdir=outdir)

    del tray, I3Tray

    return arrays


class ConvertI3ToNumpy(object):
    """
    Convert icecube objects to Numpy typed objects
    """

    __slots__ = [
        "icetray",
        "dataio",
        "dataclasses",
        "i3_scalars",
        "custom_funcs",
        "getters",
        "mapping_str_simple_scalar",
        "mapping_str_structured_scalar",
        "mapping_str_attrs",
        "attrs",
        "unhandled_types",
        "frame",
        "failed_keys",
        "frame_data",
    ]

    def __init__(self):
        # pylint: disable=unused-variable, unused-import
        from icecube import icetray, dataio, dataclasses, recclasses, simclasses

        try:
            from icecube import millipede
        except ImportError:
            millipede = None

        try:
            from icecube import santa
        except ImportError:
            santa = None

        try:
            from icecube import genie_icetray
        except ImportError:
            genie_icetray = None

        try:
            from icecube import tpx
        except ImportError:
            tpx = None

        self.icetray = icetray
        self.dataio = dataio
        self.dataclasses = dataclasses

        self.i3_scalars = {
            icetray.I3Bool: np.bool8,
            icetray.I3Int: np.int32,
            dataclasses.I3Double: np.float64,
            dataclasses.I3String: np.string0,
        }

        self.custom_funcs = {
            dataclasses.I3MCTree: self.extract_flat_mctree,
            dataclasses.I3RecoPulseSeries: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMap: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapMask: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapUnion: self.extract_flat_pulse_series,
            dataclasses.I3SuperDSTTriggerSeries: self.extract_seq_of_same_type,
            dataclasses.I3TriggerHierarchy: self.extract_flat_trigger_hierarchy,
            dataclasses.I3VectorI3Particle: self.extract_singleton_seq_to_scalar,
            dataclasses.I3DOMCalibration: self.extract_i3domcalibration,
        }

        self.getters = {recclasses.I3PortiaEvent: (rt.I3PORTIAEVENT_T, "Get{}")}

        self.mapping_str_simple_scalar = {
            dataclasses.I3MapStringDouble: np.float64,
            dataclasses.I3MapStringInt: np.int32,
            dataclasses.I3MapStringBool: np.bool8,
        }

        self.mapping_str_structured_scalar = {}
        if genie_icetray:
            self.mapping_str_structured_scalar[
                genie_icetray.I3GENIEResultDict
            ] = rt.I3GENIERESULTDICT_SCALARS_T

        self.mapping_str_attrs = {dataclasses.I3FilterResultMap: rt.I3FILTERRESULT_T}

        self.attrs = {
            icetray.I3RUsage: rt.I3RUSAGE_T,
            icetray.OMKey: rt.OMKEY_T,
            dataclasses.TauParam: rt.TAUPARAM_T,
            dataclasses.LinearFit: rt.LINEARFIT_T,
            dataclasses.SPEChargeDistribution: rt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3Direction: rt.I3DIRECTION_T,
            dataclasses.I3EventHeader: rt.I3EVENTHEADER_T,
            dataclasses.I3FilterResult: rt.I3FILTERRESULT_T,
            dataclasses.I3Position: rt.I3POSITION_T,
            dataclasses.I3Particle: rt.I3PARTICLE_T,
            dataclasses.I3ParticleID: rt.I3PARTICLEID_T,
            dataclasses.I3VEMCalibration: rt.I3VEMCALIBRATION_T,
            dataclasses.SPEChargeDistribution: rt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3SuperDSTTrigger: rt.I3SUPERDSTTRIGGER_T,
            dataclasses.I3Time: rt.I3TIME_T,
            dataclasses.I3TimeWindow: rt.I3TIMEWINDOW_T,
            recclasses.I3DipoleFitParams: rt.I3DIPOLEFITPARAMS_T,
            recclasses.I3LineFitParams: rt.I3LINEFITPARAMS_T,
            recclasses.I3FillRatioInfo: rt.I3FILLRATIOINFO_T,
            recclasses.I3FiniteCuts: rt.I3FINITECUTS_T,
            recclasses.I3DirectHitsValues: rt.I3DIRECTHITSVALUES_T,
            recclasses.I3HitStatisticsValues: rt.I3HITSTATISTICSVALUES_T,
            recclasses.I3HitMultiplicityValues: rt.I3HITMULTIPLICITYVALUES_T,
            recclasses.I3TensorOfInertiaFitParams: rt.I3TENSOROFINERTIAFITPARAMS_T,
            recclasses.I3Veto: rt.I3VETO_T,
            recclasses.I3CLastFitParams: rt.I3CLASTFITPARAMS_T,
            recclasses.I3CscdLlhFitParams: rt.I3CSCDLLHFITPARAMS_T,
            recclasses.I3DST16: rt.I3DST16_T,
            recclasses.DSTPosition: rt.DSTPOSITION_T,
            recclasses.I3StartStopParams: rt.I3STARTSTOPPARAMS_T,
            recclasses.I3TrackCharacteristicsValues: rt.I3TRACKCHARACTERISTICSVALUES_T,
            recclasses.I3TimeCharacteristicsValues: rt.I3TIMECHARACTERISTICSVALUES_T,
            recclasses.CramerRaoParams: rt.CRAMERRAOPARAMS_T,
        }
        if millipede:
            self.attrs[
                millipede.gulliver.I3LogLikelihoodFitParams
            ] = rt.I3LOGLIKELIHOODFITPARAMS_T
        if santa:
            self.attrs[santa.I3SantaFitParams] = rt.I3SANTAFITPARAMS_T

        # Define types we know we don't handle; these will be expanded as new
        # types are encountered to avoid repeatedly failing on the same types

        self.unhandled_types = set(
            [
                dataclasses.I3Geometry,
                dataclasses.I3Calibration,
                dataclasses.I3DetectorStatus,
                dataclasses.I3DOMLaunchSeriesMap,
                dataclasses.I3MapKeyVectorDouble,
                dataclasses.I3RecoPulseSeriesMapApplySPECorrection,
                dataclasses.I3SuperDST,
                dataclasses.I3TimeWindowSeriesMap,
                dataclasses.I3VectorDouble,
                dataclasses.I3VectorOMKey,
                dataclasses.I3VectorTankKey,
                dataclasses.I3MapKeyDouble,
                recclasses.I3DSTHeader16,
            ]
        )
        if tpx:
            self.unhandled_types.add(tpx.I3TopPulseInfoSeriesMap)

        self.frame = None
        self.failed_keys = set()
        self.frame_data = []

    def __call__(self, frame, keys=None):
        """Allows calling the instantiated class directly, which is the
        mechanism IceTray uses (including requiring `frame` as the first
        argument)

        Parameters
        ----------
        frame : icetray.I3Frame
        keys : str, iterable thereof, or None, optional
            Extract only these keys

        Returns
        -------
        False
            This disallows frames from being pushed to subsequent modules. I
            don't know why I picked this value. Probably not the "correct"
            value, so modify if this is an issue or there is a better way.

        """
        frame_data = self.extract_frame(frame, keys=keys)
        self.frame_data.append(frame_data)
        return False

    def finalize_icetray(self, outdir=None):
        """Construct arrays and cleanup data saved when running via icetray
        (i.e., the __call__ method)

        Parameters
        ----------
        outdir : str or None, optional
            If string, interpret as path to a directory in which to save the
            arrays (they are written to memory-mapped files to avoid excess
            memory usage). If None, exclusively construct the arrays in memory
            (do not save to disk).

        Returns
        -------
        arrays
            See `construct_arrays` for format of `arrays`

        """
        arrays = construct_arrays(self.frame_data, outdir=outdir)
        del self.frame_data[:]
        return arrays

    def extract_files(self, paths, keys=None):
        """Extract info from one or more i3 file(s)

        Parameters
        ----------
        paths : str or iterable thereof
        keys : str, iterable thereof, or None; optional

        Returns
        -------
        arrays : OrderedDict

        """
        raise NotImplementedError(
            """I3FrameSequence doesn't allow reading multiple files reliably
            while preserving GCD information for current frame. Until bug is
            fixed, this is disabled as unreliable, and use as icetray module
            instead."""
        )
        if isinstance(paths, str):
            paths = [paths]
        paths = [expand(path) for path in paths]
        i3file_iterator = self.dataio.I3FrameSequence()
        try:
            extracted_data = []
            for path in paths:
                i3file_iterator.add_file(path)
                while i3file_iterator.more():
                    frame = i3file_iterator.pop_frame()
                    if frame.Stop != self.icetray.I3Frame.Physics:
                        continue
                    data = self.extract_frame(frame=frame, keys=keys)
                    extracted_data.append(data)
                i3file_iterator.close_last_file()
        finally:
            i3file_iterator.close()

        return construct_arrays(extracted_data)

    def extract_file(self, path, keys=None):
        """Extract info from one or more i3 file(s)

        Parameters
        ----------
        path : str
        keys : str, iterable thereof, or None; optional

        Returns
        -------
        arrays : OrderedDict

        """
        path = expand(path)

        extracted_data = []

        i3file = self.dataio.I3File(path)
        try:
            while i3file.more():
                frame = i3file.pop_frame()
                if frame.Stop != self.icetray.I3Frame.Physics:
                    continue
                data = self.extract_frame(frame=frame, keys=keys)
                extracted_data.append(data)
        finally:
            i3file.close()

        return construct_arrays(extracted_data)

    def extract_frame(self, frame, keys=None):
        """Extract icetray frame objects to numpy typed objects

        Parameters
        ----------
        frame : icetray.I3Frame
        keys : str, iterable thereof, or None; optional

        """
        self.frame = frame

        auto_mode = False
        if keys is None:
            auto_mode = True
            keys = frame.keys()
        elif isinstance(keys, str):
            keys = [keys]
        keys = sorted(set(keys).difference(self.failed_keys))

        extracted_data = {}

        for key in keys:
            try:
                value = frame[key]
            except Exception:
                if auto_mode:
                    self.failed_keys.add(key)
                # else:
                #    extracted_data[key] = None
                continue

            try:
                np_value = self.extract_object(value)
            except Exception:
                print("failed on key {}".format(key))
                raise

            # if auto_mode and np_value is None:
            if np_value is None:
                continue

            extracted_data[key] = np_value

        return extracted_data

    def extract_object(self, obj, to_numpy=True):
        """Convert an object from a frame to a Numpy typed object.

        Note that e.g. extracting I3RecoPulseSeriesMap{Mask,Union} requires
        that `self.frame` be assigned the current frame to work.

        Parameters
        ----------
        obj : frame object
        to_numpy : bool, optional

        Returns
        -------
        np_obj : numpy-typed object or None

        """
        obj_t = type(obj)

        if obj_t in self.unhandled_types:
            return None

        dtype = self.i3_scalars.get(obj_t, None)
        if dtype:
            val = dtype(obj.value)
            if to_numpy:
                return val
            return val, dtype

        dtype_fmt = self.getters.get(obj_t, None)
        if dtype_fmt:
            return self.extract_getters(obj, *dtype_fmt, to_numpy=to_numpy)

        dtype = self.mapping_str_simple_scalar.get(obj_t, None)
        if dtype:
            return dict2struct(obj, set_explicit_dtype_func=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_structured_scalar.get(obj_t, None)
        if dtype:
            return maptype2np(obj, dtype=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_attrs.get(obj_t, None)
        if dtype:
            return self.extract_mapscalarattrs(obj, to_numpy=to_numpy)

        dtype = self.attrs.get(obj_t, None)
        if dtype:
            return self.extract_attrs(obj, dtype, to_numpy=to_numpy)

        func = self.custom_funcs.get(obj_t, None)
        if func:
            return func(obj, to_numpy=to_numpy)

        # New unhandled type found
        self.unhandled_types.add(obj_t)

        return None

    @staticmethod
    def extract_flat_trigger_hierarchy(obj, to_numpy=True):
        """Flatten a trigger hierarchy into a linear sequence of triggers,
        labeled such that the original hiercarchy can be recreated

        Parameters
        ----------
        obj : I3TriggerHierarchy
        to_numpy : bool, optional

        Returns
        -------
        flat_triggers : shape-(N-trigers,) numpy.ndarray of dtype FLAT_TRIGGER_T

        """
        iterattr = obj.items if hasattr(obj, "items") else obj.iteritems

        level_tups = []
        flat_triggers = []

        for level_tup, trigger in iterattr():
            level_tups.append(level_tup)
            level = len(level_tup) - 1
            if level == 0:
                parent_idx = -1
            else:
                parent_idx = level_tups.index(level_tup[:-1])
            # info_tup, _ = self.extract_attrs(trigger, TRIGGER_T, to_numpy=False)
            key = trigger.key
            flat_triggers.append(
                (
                    level,
                    parent_idx,
                    (
                        trigger.time,
                        trigger.length,
                        trigger.fired,
                        (key.source, key.type, key.subtype, key.config_id or 0),
                    ),
                )
            )

        if to_numpy:
            return np.array(flat_triggers, dtype=rt.FLAT_TRIGGER_T)

        return flat_triggers, rt.FLAT_TRIGGER_T

    def extract_flat_mctree(
        self,
        mctree,
        parent=None,
        parent_idx=-1,
        level=0,
        max_level=-1,
        flat_particles=None,
        to_numpy=True,
    ):
        """Flatten an I3MCTree into a sequence of particles with additional
        metadata "level" and "parent" for easily reconstructing / navigating the
        tree structure if need be.

        Parameters
        ----------
        mctree : icecube.dataclasses.I3MCTree
            Tree to flatten into a numpy array

        parent : icecube.dataclasses.I3Particle, optional

        parent_idx : int, optional

        level : int, optional

        max_level : int, optional
            Recurse to but not beyond `max_level` depth within the tree. Primaries
            are level 0, secondaries level 1, tertiaries level 2, etc. Set to
            negative value to capture all levels.

        flat_particles : appendable sequence or None, optional

        to_numpy : bool, optional


        Returns
        -------
        flat_particles : list of tuples or ndarray of dtype `FLAT_PARTICLE_T`


        Examples
        --------
        This is a recursive function, with defaults defined for calling simply for
        the typical use case of flattening an entire I3MCTree and producing a
        numpy.ndarray with the results. .. ::

            flat_particles = extract_flat_mctree(frame["I3MCTree"])

        """
        if flat_particles is None:
            flat_particles = []

        if max_level < 0 or level <= max_level:
            if parent:
                daughters = mctree.get_daughters(parent)
            else:
                level = 0
                parent_idx = -1
                daughters = mctree.get_primaries()

            if daughters:
                # Record index before we started appending
                idx0 = len(flat_particles)

                # First append all daughters found
                for daughter in daughters:
                    info_tup, _ = self.extract_attrs(
                        daughter, rt.I3PARTICLE_T, to_numpy=False
                    )
                    flat_particles.append((level, parent_idx, info_tup))

                # Now recurse, appending any granddaughters (daughters to these
                # daughters) at the end
                for daughter_idx, daughter in enumerate(daughters, start=idx0):
                    self.extract_flat_mctree(
                        mctree=mctree,
                        parent=daughter,
                        parent_idx=daughter_idx,
                        level=level + 1,
                        max_level=max_level,
                        flat_particles=flat_particles,
                        to_numpy=False,
                    )

        if to_numpy:
            return np.array(flat_particles, dtype=rt.FLAT_PARTICLE_T)

        return flat_particles, rt.FLAT_PARTICLE_T

    def extract_flat_pulse_series(self, obj, frame=None, to_numpy=True):
        """Flatten a pulse series into a 1D array of ((<OMKEY_T>), <PULSE_T>)

        Parameters
        ----------
        obj : dataclasses.I3RecoPUlseSeries{,Map,MapMask,MapUnion}
        frame : iectray.I3Frame, required if obj is {...Mask, ...Union}
        to_numpy : bool, optional

        Returns
        -------
        flat_pulses : shape-(N-pulses) numpy.ndarray of dtype FLAT_PULSE_T

        """
        if isinstance(
            obj,
            (
                self.dataclasses.I3RecoPulseSeriesMapMask,
                self.dataclasses.I3RecoPulseSeriesMapUnion,
            ),
        ):
            if frame is None:
                frame = self.frame
            obj = obj.apply(frame)

        flat_pulses = []
        for omkey, pulses in obj.items():
            omkey = (omkey.string, omkey.om, omkey.pmt)
            for pulse in pulses:
                info_tup, _ = self.extract_attrs(
                    pulse, dtype=rt.PULSE_T, to_numpy=False
                )
                flat_pulses.append((omkey, info_tup))

        if to_numpy:
            return np.array(flat_pulses, dtype=rt.FLAT_PULSE_T)

        return flat_pulses, rt.FLAT_PULSE_T

    def extract_singleton_seq_to_scalar(self, seq, to_numpy=True):
        """Extract a sole object from a sequence and treat it as a scalar.
        E.g., I3VectorI3Particle that, by construction, contains just one
        particle


        Parameters
        ----------
        seq : sequence
        to_numpy : bool, optional


        Returns
        -------
        obj

        """
        assert len(seq) == 1
        return self.extract_object(seq[0], to_numpy=to_numpy)

    def extract_attrs(self, obj, dtype, to_numpy=True):
        """Extract attributes of an object (and optionally, recursively, attributes
        of those attributes, etc.) into a numpy.ndarray based on the specification
        provided by `dtype`.


        Parameters
        ----------
        obj
        dtype : numpy.dtype
        to_numpy : bool, optional


        Returns
        -------
        vals : tuple or shape-(1,) numpy.ndarray of dtype `dtype`

        """
        vals = []
        if isinstance(dtype, np.dtype):
            descr = dtype.descr
        elif isinstance(dtype, Sequence):
            descr = dtype
        else:
            raise TypeError("{}".format(dtype))

        for name, subdtype in descr:
            val = getattr(obj, name)
            if isinstance(subdtype, (str, np.dtype)):
                vals.append(val)
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    out = self.extract_attrs(val, subdtype, to_numpy=False)
                assert out is not None, "{}: {} {}".format(name, subdtype, val)
                info_tup, _ = out
                vals.append(info_tup)
            else:
                raise TypeError("{}".format(subdtype))

        # Numpy converts tuples correctly; lists are interpreted differently
        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_mapscalarattrs(self, mapping, subdtype=None, to_numpy=True):
        """Convert a mapping (containing string keys and scalar-typed values)
        to a single-element Numpy array from the values of `mapping`, using
        keys defined by `subdtype.names`.

        Use this function if you already know the `subdtype` you want to end up
        with. Use `retro.utils.misc.dict2struct` directly if you do not know
        the dtype(s) of the mapping's values ahead of time.


        Parameters
        ----------
        mapping : mapping from strings to scalars

        dtype : numpy.dtype
            If scalar dtype, convert via `utils.dict2struct`. If structured
            dtype, convert keys specified by the struct field names and values
            are converted according to the corresponding type.


        Returns
        -------
        array : shape-(1,) numpy.ndarray of dtype `dtype`


        See Also
        --------
        dict2struct
            Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
            as you go (i.e., this is not known a priori)

        """
        keys = mapping.keys()
        if not isinstance(mapping, OrderedDict):
            keys.sort()

        out_vals = []
        out_dtype = []

        if subdtype is None:  # infer subdtype from values in mapping
            for key in keys:
                val = mapping[key]
                info_tup, subdtype = self.extract_object(val, to_numpy=False)
                out_vals.append(info_tup)
                out_dtype.append((key, subdtype))
        else:  # scalar subdtype
            for key in keys:
                out_vals.append(mapping[key])
                out_dtype.append((key, subdtype))

        out_vals = tuple(out_vals)

        if to_numpy:
            return np.array([out_vals], dtype=out_dtype)[0]

        return out_vals, out_dtype

    def extract_getters(self, obj, dtype, fmt="Get{}", to_numpy=True):
        """Convert an object whose data has to be extracted via methods that
        behave like getters (e.g., .`xyz = get_xyz()`).


        Parameters
        ----------
        obj
        dtype
        fmt : str
        to_numpy : bool, optional


        Examples
        --------
        To get all of the values of an I3PortiaEvent: .. ::

            extract_getters(frame["PoleEHESummaryPulseInfo"], dtype=rt.I3PORTIAEVENT_T, fmt="Get{}")

        """
        vals = []
        for name, subdtype in dtype.descr:
            getter_attr_name = fmt.format(name)
            getter_func = getattr(obj, getter_attr_name)
            val = getter_func()
            if not isinstance(subdtype, str) and isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "Failed to convert name {} val {} type {}".format(
                            name, val, type(val)
                        )
                    )
                val, _ = out
            # if isinstance(val, self.icetray.OMKey):
            #    val = self.extract_attrs(val, dtype=rt.OMKEY_T, to_numpy=False)
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_seq_of_same_type(self, seq, to_numpy=True):
        """Convert a sequence of objects, all of the same type, to a numpy array of
        that type.

        Parameters
        ----------
        seq : seq of N objects all of same type
        to_numpy : bool, optional

        Returns
        -------
        out_seq : list of N tuples or shape-(N,) numpy.ndarray of `dtype`

        """
        assert len(seq) > 0

        # Convert first object in sequence to get dtype
        val0 = seq[0]
        val0_tup, val0_dtype = self.extract_object(val0, to_numpy=False)
        data_tups = [val0_tup]

        # Convert any remaining objects
        for obj in seq[1:]:
            data_tups.append(self.extract_object(obj, to_numpy=False)[0])

        if to_numpy:
            return np.array(data_tups, dtype=val0_dtype)

        return data_tups, val0_dtype

    def extract_i3domcalibration(self, obj, to_numpy=True):
        """Extract the information from an I3DOMCalibration frame object"""
        vals = []
        for name, subdtype in rt.I3DOMCALIBRATION_T.descr:
            val = getattr(obj, name)
            if name == "dom_cal_version":
                if val == "unknown":
                    val = (-1, -1, -1)
                else:
                    val = tuple(int(x) for x in val.split("."))
            elif isinstance(subdtype, (str, np.dtype)):
                pass
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "{} {} {} {}".format(name, subdtype, val, type(val))
                    )
                val, _ = out
            else:
                raise TypeError(str(subdtype))
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=rt.I3DOMCALIBRATION_T)[0]

        return vals, rt.I3DOMCALIBRATION_T


def main():
    """Command line interface"""
    # pylint: disable=line-too-long

    from processing.samples.oscNext.verification.general_mc_data_harvest_and_plot import (
        ALL_OSCNEXT_VARIABLES,
    )

    mykeys = """L5_SPEFit11 LineFit_DC I3TriggerHierarchy SRTTWOfflinePulsesDC
    SRTTWOfflinePulsesDCTimeRange SplitInIcePulses SplitInIcePulsesTimeRange
    L5_oscNext_bool I3EventHeader I3MCWeightDict I3TriggerHierarchy
    I3GENIEResultDict I3MCTree""".split()
    keys = sorted(set([k.split(".")[0] for k in ALL_OSCNEXT_VARIABLES.keys()] + mykeys))
    # keys = mykeys

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_extract = subparsers.add_parser("extract")
    parser_extract.add_argument("--path", required=True)
    parser_extract.add_argument("--gcd", default=None)
    parser_extract.add_argument("--outdir", required=True)
    parser_extract.add_argument("--tempdir", default="/tmp")
    parser_extract.add_argument("--keys", nargs="+", default=keys)
    parser_extract.add_argument("--overwrite", action="store_true")
    parser_extract.add_argument("--no-mmap", action="store_true")
    parser_extract.add_argument("--keep-tempfiles-on-fail", action="store_true")
    parser_extract.add_argument("--procs", type=int, default=cpu_count())
    parser_extract.set_defaults(func=extract_run)

    parser_extract = subparsers.add_parser("combine_season_runs")
    parser_extract.add_argument("--path", required=True)
    parser_extract.add_argument("--outdir", required=True)
    parser_extract.add_argument("--keys", nargs="+", default=keys)
    parser_extract.add_argument("--no-mmap", action="store_true")
    parser_extract.set_defaults(func=combine_season_runs)

    parser_extract = subparsers.add_parser("compress")
    parser_extract.add_argument("--path", required=True)
    parser_extract.add_argument("--keep", action="store_true")
    parser_extract.add_argument("--keys", default=None)
    parser_extract.set_defaults(func=compress)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    if "no_mmap" in kwargs:
        kwargs["mmap"] = not kwargs.pop("no_mmap")
    func(**kwargs)


if __name__ == "__main__":
    main()
