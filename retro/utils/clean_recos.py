#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Clean recos: find retro_*.npy files and if reconstructions say they were OK
(fit_status == FitStatus.OK) but reco values are not finite (i.e., either
infinite or nan), set fit_status to a more appropriate value (e.g.,
FitStatus.NotSet).
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

__all__ = ["DEFAULT_INVALID_STATUS", "clean_recos", "main"]

from argparse import ArgumentParser
from os import walk
from os.path import abspath, dirname, expanduser, expandvars, join, splitext
import sys

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import FitStatus


DEFAULT_INVALID_STATUS = FitStatus.NotSet


def clean_recos(dirs, recos=None, invalid_status=DEFAULT_INVALID_STATUS):
    """Cleanup recos with nonsensical values but "fit_status" of FitStatus.OK.

    Find retro_{reco_name}.npy files and if reconstructions say they were OK
    (fit_status == FitStatus.OK) but reco values are not finite (i.e., either
    infinite or nan), set fit_status either to FitStatus as provided by user.

    Parameters
    ----------
    dirs : str or iterable thereof
        Directory path(s) in which to recursively look for reconstructions files
        ("reco_*.npy")

    recos : str, iterable thereof, or None; optional
        Specify only those reco names to clean. If None (default), all retro
        recos will be cleaned in `dir`.

    invalid_status : FitStatus or int convertible thereto, optional
        Invalid recos (with nan or infinite reco param values) will have their
        "fit_status" field set to this value. Default is FitStatus.NotSet.

    """
    if isinstance(dirs, string_types):
        dirs = [dirs]
    dirs = [abspath(expanduser(expandvars(d))) for d in dirs]

    if isinstance(recos, string_types):
        recos = [recos]

    if recos is not None:
        new_recos = []
        for reco in recos:
            if reco.startswith("retro_"):
                new_recos.append(reco)
            else:
                sys.stderr.write(
                    "WARNING: Adding 'retro_' prefix to reco '{}'\n".format(reco)
                )
                new_recos.append("retro_{}".format(reco))
        recos = new_recos

    invalid_status = FitStatus(invalid_status)

    to_clean = []
    for rootdir in dirs:
        for dirpath, _, filenames in walk(rootdir):
            for filename in filenames:
                root, ext = splitext(filename)
                if ext != ".npy":
                    continue
                if recos is None:
                    if root.startswith("retro_"):
                        to_clean.append(join(dirpath, filename))
                elif root in recos:
                    to_clean.append(join(dirpath, filename))

    print("\n".join(to_clean))
    for fpath in to_clean:
        reco_vals = np.load(fpath, mmap_mode="w+")


def main(description=__doc__):
    """Script interface to `clean_recos` function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument(
        "--recos",
        nargs="+",
        default=None,
        help="""If not specified, all retro_*.npy files will be cleaned""",
    )
    fit_status_vals = [fs.value for fs in FitStatus if fs != FitStatus.OK]
    fit_status_str = ", ".join(
        "{:2d}={}".format(fs.value, fs.name) for fs in FitStatus if fs != FitStatus.OK
    )
    parser.add_argument(
        "--invalid-status",
        type=int,
        default=FitStatus.NotSet,
        choices=fit_status_vals,
        help=""""fit_status" to set in a reco if it is determined to be
        invalid. Integer values are interpreted as: {}. If not specified,
        default is {}={}.""".format(
            fit_status_str, DEFAULT_INVALID_STATUS.value, DEFAULT_INVALID_STATUS.name
        ),
    )
    kwargs = vars(parser.parse_args())
    clean_recos(**kwargs)


if __name__ == "__main__":
    main()
