#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order


from __future__ import absolute_import, division, print_function

__all__ = [
    "DATA_GCD_FNAME_RE",
    "OSCNEXT_I3_FNAME_RE",
    "find_gcds_in_dirs",
    "set_explicit_dtype",
    "test_OSCNEXT_I3_FNAME_RE",
]

from collections import OrderedDict
from numbers import Integral, Number
from os import walk
from os.path import abspath, dirname, join
import re
import sys

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, nsort_key_func


DATA_GCD_FNAME_RE = re.compile(
    r"""
    Level(?P<level>[0-9]+)
    (pass(?P<pass>[0-9]+))?
    _IC86\.(?P<year>[0-9]+)
    _data
    _Run(?P<run>[0-9]+)
    .*GCD.*
    \.i3
    (\..*)?
    """,
    flags=re.IGNORECASE | re.VERBOSE,

)


OSCNEXT_I3_FNAME_RE = re.compile(
    r"""
    (?P<basename>oscNext_(?P<kind>\S+?)
        (_IC86\.(?P<season>[0-9]+))?       #  only present for data
        _level(?P<level>[0-9]+)
        .*?                                #  other infixes, e.g. "addvars"
        _v(?P<levelver>[0-9.]+)
        _pass(?P<pass>[0-9]+)
        (_Run|\.)(?P<run>[0-9]+)           # data run pfxd by "_Run", MC by "."
        ((_Subrun|\.)(?P<subrun>[0-9]+))?  # data subrun pfxd by "_Subrun", MC by "."
    )
    \.i3
    (?P<compr_exts>(\..*)*)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def find_gcds_in_dirs(dirs, gcd_fname_re, recurse=True):
    """Find data run GCD files in directories.

    Parameters
    ----------
    dirs : str or iterable thereof
    recurse : bool

    Returns
    -------
    data_run_gcds : dict
        Keys are <tuple>(<str>2-digit season, <str>run number) and values are
        <str> path to corresponding GCD file

    """
    if isinstance(dirs, str):
        dirs = [dirs]
    dirs = [expand(rootdir) for rootdir in dirs]

    data_run_gcds = {}
    for rootdir in dirs:
        for dirpath, subdirs, files in walk(rootdir):
            if recurse:
                subdirs.sort(key=nsort_key_func)
            else:
                del subdirs[:]
            files.sort(key=nsort_key_func)

            for fname in files:
                gcd_match = gcd_fname_re.match(fname)
                if gcd_match:
                    gcd_groupdict = gcd_match.groupdict()
                    # get 2 digit year
                    year = "{:02d}".format(int(gcd_groupdict["year"]) % 2000)
                    key = (year, gcd_groupdict["run"])
                    # prefer "levelXpassY_* gcd files
                    if key in data_run_gcds and gcd_groupdict["pass"] is None:
                        continue
                    data_run_gcds[key] = join(dirpath, fname)

    return data_run_gcds


def set_explicit_dtype(x):
    """Force `x` to have a numpy type if it doesn't already have one.

    Parameters
    ----------
    x : numpy-typed object, bool, integer, float
        If not numpy-typed, type is attempted to be inferred. Currently only
        bool, int, and float are supported, where bool is converted to
        np.bool8, integer is converted to np.int64, and float is converted to
        np.float64. This ensures that full precision for all but the most
        extreme cases is maintained for inferred types.

    Returns
    -------
    x : numpy-typed object

    Raises
    ------
    TypeError
        In case the type of `x` is not already set or is not a valid inferred
        type. As type inference can yield different results for different
        inputs, rather than deal with everything, explicitly failing helps to
        avoid inferring the different instances of the same object differently
        (which will cause a failure later on when trying to concatenate the
        types in a larger array).

    """
    if hasattr(x, "dtype"):
        return x

    # "value" attribute is found in basic icecube.{dataclasses,icetray} dtypes
    # such as I3Bool, I3Double, I3Int, and I3String
    if hasattr(x, "value"):
        x = x.value

    # bools are numbers.Integral, so test for bool first
    if isinstance(x, bool):
        return np.bool8(x)

    if isinstance(x, Integral):
        x_new = np.int64(x)
        assert x_new == x
        return x_new

    if isinstance(x, Number):
        x_new = np.float64(x)
        assert x_new == x
        return x_new

    if isinstance(x, string_types):
        x_new = np.string0(x)
        assert x_new == x
        return x_new

    raise TypeError("Type of argument ({}) is invalid: {}".format(x, type(x)))


def dict2struct(
    mapping,
    set_explicit_dtype_func=set_explicit_dtype,
    only_keys=None,
    to_numpy=True,
):
    """Convert a dict with string keys and numpy-typed values into a numpy
    array with struct dtype.


    Parameters
    ----------
    mapping : Mapping
        The dict's keys are the names of the fields (strings) and the dict's
        values are numpy-typed objects. If `mapping` is an OrderedMapping,
        produce struct with fields in that order; otherwise, sort the keys for
        producing the dict.

    set_explicit_dtype_func : callable with one positional argument, optional
        Provide a function for setting the numpy dtype of the value. Useful,
        e.g., for icecube/icetray usage where special software must be present
        (not required by this module) to do the work. If no specified,
        the `set_explicit_dtype` function defined in this module is used.

    only_keys : str, sequence thereof, or None; optional
        Only extract one or more keys; pass None to extract all keys (default)

    to_numpy : bool, optional


    Returns
    -------
    array : numpy.array of struct dtype

    """
    if only_keys and isinstance(only_keys, str):
        only_keys = [only_keys]

    out_vals = []
    out_dtype = []

    keys = mapping.keys()
    if not isinstance(mapping, OrderedDict):
        keys.sort()

    for key in keys:
        if only_keys and key not in only_keys:
            continue
        val = set_explicit_dtype_func(mapping[key])
        out_vals.append(val)
        out_dtype.append((key, val.dtype))

    out_vals = tuple(out_vals)

    if to_numpy:
        return np.array([out_vals], dtype=out_dtype)[0]

    return out_vals, out_dtype


def maptype2np(mapping, dtype, to_numpy=True):
    """Convert a mapping (containing string keys and scalar-typed values) to a
    single-element Numpy array from the values of `mapping`, using keys
    defined by `dtype.names`.

    Use this function if you already know the `dtype` you want to end up with.
    Use `retro.utils.misc.dict2struct` directly if you do not know the dtype(s)
    of the mapping's values ahead of time.


    Parameters
    ----------
    mapping : mapping from strings to scalars

    dtype : numpy.dtype
        If scalar dtype, convert via `utils.dict2struct`. If structured dtype,
        convert keys specified by the struct field names and values are
        converted according to the corresponding type.


    Returns
    -------
    array : shape-(1,) numpy.ndarray of dtype `dtype`


    See Also
    --------
    dict2struct
        Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
        as you go (i.e., this is not known a priori)

    mapscalarattrs2np

    """
    out_vals = tuple(mapping[name] for name in dtype.names)
    if to_numpy:
        return np.array([out_vals], dtype=dtype)[0]
    return out_vals, dtype


def test_OSCNEXT_I3_FNAME_RE():
    """Unit tests for OSCNEXT_I3_FNAME_RE."""
    # pylint: disable=line-too-long
    test_cases = [
        (
            "oscNext_data_IC86.12_level5_v01.04_pass2_Run00120028_Subrun00000000.i3.zst",
            {
                'basename': 'oscNext_data_IC86.12_level5_v01.04_pass2_Run00120028_Subrun00000000',
                'compr_exts': '.zst',
                'kind': 'data',
                'level': '5',
                'pass': '2',
                'levelver': '01.04',
                #'misc': '',
                'run': '00120028',
                'season': '12',
                'subrun': '00000000',
            },
        ),
        (
            "oscNext_data_IC86.18_level7_addvars_v01.04_pass2_Run00132761.i3.zst",
            {
                'basename': 'oscNext_data_IC86.18_level7_addvars_v01.04_pass2_Run00132761',
                'compr_exts': '.zst',
                'kind': 'data',
                'level': '7',
                'pass': '2',
                'levelver': '01.04',
                #'misc': 'addvars',
                'run': '00132761',
                'season': '18',
                'subrun': None,
            },
        ),
        (
            "oscNext_genie_level5_v01.01_pass2.120000.000216.i3.zst",
            {
                'basename': 'oscNext_genie_level5_v01.01_pass2.120000.000216',
                'compr_exts': '.zst',
                'kind': 'genie',
                'level': '5',
                'pass': '2',
                'levelver': '01.01',
                #'misc': '',
                'run': '120000',
                'season': None,
                'subrun': '000216',
            },
        ),
        (
            "oscNext_noise_level7_v01.03_pass2.888003.000000.i3.zst",
            {
                'basename': 'oscNext_noise_level7_v01.03_pass2.888003.000000',
                'compr_exts': '.zst',
                'kind': 'noise',
                'level': '7',
                'pass': '2',
                'levelver': '01.03',
                #'misc': '',
                'run': '888003',
                'season': None,
                'subrun': '000000',
            },
        ),
        (
            "oscNext_muongun_level5_v01.04_pass2.139011.000000.i3.zst",
            {
                'basename': 'oscNext_muongun_level5_v01.04_pass2.139011.000000',
                'compr_exts': '.zst',
                'kind': 'muongun',
                'level': '5',
                'pass': '2',
                'levelver': '01.04',
                #'misc': '',
                'run': '139011',
                'season': None,
                'subrun': '000000',
            },
        ),
        (
            "oscNext_corsika_level5_v01.03_pass2.20788.000000.i3.zst",
            {
                'basename': 'oscNext_corsika_level5_v01.03_pass2.20788.000000',
                'compr_exts': '.zst',
                'kind': 'corsika',
                'level': '5',
                'pass': '2',
                'levelver': '01.03',
                #'misc': '',
                'run': '20788',
                'season': None,
                'subrun': '000000',
            }
        ),
    ]

    for test_input, expected_output in test_cases:
        try:
            match = OSCNEXT_I3_FNAME_RE.match(test_input)
            groupdict = match.groupdict()

            ref_keys = set(expected_output.keys())
            actual_keys = set(groupdict.keys())
            if actual_keys != ref_keys:
                excess = actual_keys.difference(ref_keys)
                missing = ref_keys.difference(actual_keys)
                err_msg = []
                if excess:
                    err_msg.append("excess keys: " + str(sorted(excess)))
                if missing:
                    err_msg.append("missing keys: " + str(sorted(missing)))
                if err_msg:
                    raise ValueError("; ".join(err_msg))

            err_msg = []
            for key, ref_val in expected_output.items():
                actual_val = groupdict[key]
                if actual_val != ref_val:
                    err_msg.append(
                        '"{key}": actual_val = "{actual_val}"'
                        ' but ref_val = "{ref_val}"'.format(
                            key=key, actual_val=actual_val, ref_val=ref_val
                        )
                    )
            if err_msg:
                raise ValueError("; ".join(err_msg))
        except Exception:
            sys.stderr.write('Failure on test input = "{}"\n'.format(test_input))
            raise


if __name__ == "__main__":
    test_OSCNEXT_I3_FNAME_RE()
