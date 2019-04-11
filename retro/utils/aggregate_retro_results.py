#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name


"""
Recursively search for and aggregate "slc*.{reco}.npy" reco files into a single
"{reco}.npy" file (one file per leaf directory)
"""


from __future__ import absolute_import, division, print_function

__all__ = ["SLC_RE", "get_fname_info", "aggregate_retro_results", "main"]

from argparse import ArgumentParser
from os import remove, walk
from os.path import expanduser, expandvars, isfile, join
import re
import time

import numpy as np


SLC_RE = re.compile(
    r"""
    .*
    slc
    (?P<start>[0-9]*):(?P<stop>[0-9]*):(?P<step>[0-9]*)\.
    (?P<reco_name>.+)\.
    estimate\.
    (?P<ext>npy|pkl)$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def get_fname_info(fname):
    """Get metadata from filename

    Parameters
    ----------
    fname : str

    Returns
    -------
    info : dict or None

    """
    info = None
    slc_match = SLC_RE.match(fname)
    if slc_match:
        match_d = slc_match.groupdict()
        reco_name = match_d["reco_name"]
        start = match_d["start"]
        stop = match_d["stop"]
        step = match_d["step"]
        ext = match_d["ext"]
        start = None if start == "" else int(start)
        stop = None if stop == "" else int(stop)
        step = None if step == "" else int(step)

        info = dict(start=start, stop=stop, step=step, reco_name=reco_name, ext=ext)

    return info


def aggregate_retro_results(
    dir, overwrite=False, remove_sourcefiles=False
):  # pylint: disable=redefined-builtin
    """Combine all retro recos prefixed by "slc" into single file(s).

    Parameters
    ----------
    dir : string
    remove_sourcefiles : bool
    overwrite : bool

    """
    t0 = time.time()

    dir = expanduser(expandvars(dir))
    num_aggregated = 0

    for dirpath, _, files in walk(dir, followlinks=True):
        is_leafdir = False
        for fname in files:
            if SLC_RE.match(fname):
                is_leafdir = True
                break
        if not is_leafdir:
            continue

        infos = {}
        skip_recos = []

        for fname in files:
            fname_info = get_fname_info(fname)
            if fname_info is None:
                continue
            start = fname_info["start"]
            stop = fname_info["stop"]
            step = fname_info["step"]
            reco_name = fname_info["reco_name"]
            ext = fname_info["ext"]

            if reco_name in skip_recos:
                continue

            if reco_name not in infos:
                outfname = "{}.npy".format(reco_name)
                outfpath = join(dirpath, outfname)
                if not overwrite and isfile(outfpath):
                    print(
                        'Aggregate reco file exists at "{}", skipping reco "{}"'.format(
                            outfpath, reco_name
                        )
                    )
                    skip_recos.append(reco_name)
                    continue
                infos[reco_name] = dict(
                    outfpath=outfpath,
                    reco_subsets=[],
                    indices=[],
                    length=0,
                    to_remove=[],
                )

            info = infos[reco_name]

            fpath = join(dirpath, fname)
            if ext == "npy":
                reco_subset = np.load(fpath)
            else:
                raise NotImplementedError(
                    'Extension not handled: "{}"'.format(join(dirpath, fpath))
                )

            if remove_sourcefiles:
                info["to_remove"].append(fpath)

            stop = start + step * len(reco_subset)
            indices = np.array(range(start, stop, step))
            num_aggregated += len(reco_subset)

            info["reco_subsets"].append(reco_subset)
            info["indices"].append(indices)
            if indices[-1] + 1 > info["length"]:
                info["length"] = indices[-1] + 1

        for reco_name, info in infos.items():
            dtype = info["reco_subsets"][0].dtype
            reco_array = np.full(shape=info["length"], fill_value=np.nan, dtype=dtype)
            for indices, reco_subset in zip(info["indices"], info["reco_subsets"]):
                try:
                    reco_array[indices] = reco_subset
                except:
                    print(reco_array.dtype)
                    print(reco_subset.dtype)
                    print(reco_array.dtype == reco_subset.dtype)
                    raise

            # Save to disk
            np.save(info["outfpath"], reco_array)
            print('"{}" recos saved to file "{}"'.format(reco_name, info["outfpath"]))

            if remove_sourcefiles:
                for fpath in info["to_remove"]:
                    remove(fpath)

    dt = time.time() - t0
    print("\nTook {:.3f} s to aggregate {} recos".format(dt, num_aggregated))


def main(description=__doc__):
    """Script interface to `aggregate_retro_results`: Parse command line
    arguments then call `aggregate_retro_results`"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        help="""Directory in which to recursively search for reco files""",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""Overwrite existing aggregated reco file""",
    )
    parser.add_argument(
        "--remove-sourcefiles",
        action="store_true",
        help="""Whether to remove source files after the aggregate file is
        successfully written""",
    )

    kwargs = vars(parser.parse_args())
    aggregate_retro_results(**kwargs)


if __name__ == "__main__":
    main()
