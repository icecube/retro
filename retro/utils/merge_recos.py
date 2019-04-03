#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Merge recos from one or more directories into a target directory
"""

from __future__ import absolute_import, division, print_function

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2019 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = ["merge_recos", "main"]

from argparse import ArgumentParser
from collections import OrderedDict
from os import listdir, makedirs, walk
from os.path import (
    abspath,
    basename,
    dirname,
    expanduser,
    expandvars,
    join,
    isdir,
    isfile,
    relpath,
    splitext,
)
import sys

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import FitStatus


def merge_recos(sourcedirs, destdir, recos, overwrite=False):
    """Merge recos found in `sourcedirs` and write results to `destdir`.

    Parameters
    ----------
    sourcedirs : str or iterable thereof
    destdir : str
    recos : str or iterable thereof
    overwrite : bool, optional

    """
    if isinstance(sourcedirs, string_types):
        sourcedirs = [sourcedirs]
    sourcedirs = [abspath(expanduser(expandvars(d))) for d in sourcedirs]
    destdir = abspath(expanduser(expandvars(destdir)))
    if isinstance(recos, string_types):
        recos = [recos]

    ref_sourcedir = sourcedirs[0]

    reco_filenames = ["{}.npy".format(reco) for reco in recos]

    for dirpath, _, filenames in walk(ref_sourcedir):
        is_recodir = False
        if basename(dirpath) == "recos":
            for filename in filenames:
                if filename.endswith(".npy"):
                    is_recodir = True
        if not is_recodir:
            continue

        recodir_relpath = relpath(dirpath, ref_sourcedir)

        to_merge = OrderedDict()
        for sourcedir in sourcedirs:
            check_dirpath = join(sourcedir, recodir_relpath)
            if not isdir(check_dirpath):
                continue
            for filename in sorted(listdir(check_dirpath)):
                if filename not in reco_filenames:
                    continue
                if filename not in to_merge:
                    to_merge[filename] = []
                array = np.load(join(check_dirpath, filename))
                fit_ok_mask = array["fit_status"] == FitStatus.OK
                to_merge[filename].append(
                    (sourcedir, check_dirpath, array, fit_ok_mask)
                )

        # Perform the merge
        for filename, info in to_merge.items():
            out_dirname = join(destdir, recodir_relpath)
            if not isdir(out_dirname):
                makedirs(out_dirname, mode=0o750)
            out_filepath = join(out_dirname, filename)
            if not overwrite and isfile(out_filepath):
                raise IOError('"{}" already exists'.format(out_filepath))

            merged_array = None
            merged_fit_ok_mask = None
            for sourcedir, check_dirpath, array, fit_ok_mask in info:
                if merged_array is None:
                    merged_array = array
                    merged_fit_ok_mask = fit_ok_mask
                    continue
                new_fits_mask = fit_ok_mask & np.logical_not(merged_fit_ok_mask)
                dup_fits_mask = fit_ok_mask & merged_fit_ok_mask
                num_dupes = np.count_nonzero(dup_fits_mask)
                if num_dupes > 0:
                    print(
                        '{} duplicated "{}" recos found in dir "{}"'.format(
                            num_dupes, splitext(filename)[0], check_dirpath
                        )
                    )
                merged_array[new_fits_mask] = array[new_fits_mask]
                merged_fit_ok_mask |= fit_ok_mask

            np.save(out_filepath, merged_array)


def main(description=__doc__):
    """Script interface to `merge_recos` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--sourcedirs", nargs="+")
    parser.add_argument("--destdir")
    parser.add_argument("--recos", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    kwargs = vars(parser.parse_args())
    merge_recos(**kwargs)


if __name__ == "__main__":
    main()
