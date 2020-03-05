#!/usr/bin/env python
# pylint: disable=wrong-import-position


"""
Find missing, bad, or old extracted pulse series and print the paths of the
corresponding events directories.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["find_problematic_pulses", "main"]

from argparse import ArgumentParser
from os import walk
from os.path import abspath, dirname, isdir, isfile, join
import re
import sys

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.utils.misc import expand, nsort_key_func


OSCNEXT_FNAME_RE = re.compile(r"(?P<basename>oscNext.*)\.i3(\..*)*")


def find_problematic_pulses(indir, pulse_series):
    """Find missing, bad, or old extracted pulse series and print the paths of
    the corresponding events directories.

    Parameters
    ----------
    indir : str
    pulse_series : str or iterable thereof

    """
    if isinstance(pulse_series, str):
        pulse_series = [pulse_series]
    indir = expand(indir)

    for dirpath, dirs_, files in walk(indir, followlinks=True):
        if "events.npy" in files:
            dirs_.clear()
        else:
            dirs_.sort(key=nsort_key_func)
            files.sort(key=nsort_key_func)

            for fname in files:
                match = OSCNEXT_FNAME_RE.match(fname)
                if not match:
                    continue

                i3f_dname = join(dirpath, match.groupdict()["basename"])
                if isdir(i3f_dname):
                    if not isfile(join(i3f_dname, "events.npy")):
                        print(i3f_dname)
                else:
                    print(i3f_dname)

            continue

        sys.stderr.write(".")
        sys.stderr.flush()

        # If any one of the named pulse series are missing or bad, record
        # the path and move on without checking the other pulse series
        for ps_name in pulse_series:
            pulses_fpath = join(dirpath, "pulses", ps_name + ".pkl")
            if not isfile(pulses_fpath):
                print(dirpath)
                break
            try:
                pulses = load_pickle(pulses_fpath)
                if len(pulses) > 0 and "flags" not in pulses[0][0][1].dtype.names:
                    print(dirpath)
                    break
            except Exception:
                print(dirpath)
                break


def main(description=__doc__):
    """Script interface to produce_arrays function"""
    parser = ArgumentParser(description=description)
    parser.add_argument("--indir")
    parser.add_argument(
        "--pulse-series", nargs="+", help="E.g., specify SRTTWOfflinePulsesDC"
    )
    args = parser.parse_args()
    kwargs = vars(args)
    find_problematic_pulses(**kwargs)


if __name__ == "__main__":
    main()
