#!/usr/bin/env python
# pylint: disable=wrong-import-position


"""
Find and extract oscNext events to retro (native python/numpy-friendly) format.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["find_files_to_extract", "main"]

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from inspect import getargspec
from itertools import chain
from multiprocessing import Pool, cpu_count
from os import walk
from os.path import abspath, basename, dirname, isdir, isfile, join
from socket import gethostname
import sys

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3processing.extract_events import wrapped_extract_events
from retro.i3processing.extract_common import (
    OSCNEXT_I3_FNAME_RE, DATA_GCD_FNAME_RE, find_gcds_in_dirs
)
from retro.utils.misc import expand, nsort_key_func


def find_files_to_extract(
    roots, overwrite, find_gcd_in_dir=False, data_run_gcds=None
):
    """Find missing, bad, or old extracted pulse series and print the paths of
    the corresponding events directories.

    Parameters
    ----------
    roots : str

    overwrite : bool

    find_gcd_in_dir : bool or str
        If True, search for a data run's GCD file in the same directory as each
        i3 data file that is found. If False, do not search for a data run's
        GCD file.

        If `find_gcd_in_dir` is a string, interpret as a directory path; search
        for a data run's GCD file in that directory.

        Note that Monte Carlo i3 files will not return a `gcd_fpath`, as it is
        difficult to logically ascertain which GCD was used for a MC run and
        where it exists.

    data_run_gcds : dict or None, optional
        Keys must be <tuple>(<str>2-digit IC86 season, <str>Run number). Each
        value is a string full path to the corresponding GCD file.

    Yields
    ------
    fpath : str
        Full path to data/MC i3 file

    gcd_fpath : str
        Full path to GCD file corresponding to the data i3 file at fpath

    fname_groupdict : dict
        As returned by OSCNEXT_I3_FNAME_RE.match(...).groupdict()

    """
    if isinstance(roots, str):
        roots = [roots]
    roots = [expand(root) for root in roots]

    # If `find_gcd_in_dir` is a string, interpret as a directory and search for
    # GCD's in that directory (recursively)
    found_data_run_gcds = None
    if isinstance(find_gcd_in_dir, str):
        find_gcd_in_dir = expand(find_gcd_in_dir)
        assert isdir(find_gcd_in_dir), str(find_gcd_in_dir)
        found_data_run_gcds = find_gcds_in_dirs(
            find_gcd_in_dir, gcd_fname_re=DATA_GCD_FNAME_RE, recurse=True
        )


    def get_i3_events_file_info(dirpath, fname):
        """Closure to find only i3 events file names and, in that case, grab a
        relevant GCD file (if file contains data events and such a GCD can be
        found in `dirpath`), and return the info extracted from the file name.

        Parameters
        ----------
        dirpath : str
            Fully qualified path to file's directory

        fname : str
            (basename) of the file (i.e., excluding any directories)

        Returns
        -------
        retval : None or 3-tuple
            Returns `None` if the file is determined to not be an i3 events
            file (based on filename alone). Otherwise, returns .. ::

                fpath : str
                    fully qualified (including directories) path to the i3
                    events file

                gcd_fpath : str or None
                    fully qualified (including directories) path to a relevant
                    GCD file found in the same dir, or None if none is found

                fname_groupdict : mapping
                    Filename info as returned by regex

        """
        fname_match = OSCNEXT_I3_FNAME_RE.match(fname)
        if not fname_match:
            return None

        fname_groupdict = fname_match.groupdict()

        i3_retro_dir = join(dirpath, fname_groupdict["basename"])
        if (
            not overwrite
            and isdir(i3_retro_dir)
            and isfile(join(i3_retro_dir, "events.npy"))
        ):
            return None

        fpath = join(dirpath, fname)

        gcd_fpath = None
        if fname_groupdict["kind"] == "data":
            key = (fname_groupdict["season"], fname_groupdict["run"])

            if data_run_gcds:
                gcd_fpath = data_run_gcds.get(key, None)

            if gcd_fpath is None and found_data_run_gcds:
                gcd_fpath = found_data_run_gcds.get(key, None)

            if gcd_fpath is None and thisdir_data_run_gcds:
                gcd_fpath = thisdir_data_run_gcds.get(key, None)

        return fpath, gcd_fpath, fname_groupdict


    for root in roots:
        if isfile(root):
            retval = get_i3_events_file_info(dirpath=dirname(root), fname=basename(root))
            if retval is not None:
                yield retval
            continue

        for dirpath, dirs, files in walk(root, followlinks=True):
            if "events.npy" in files:
                # No need to recurse into an existing retro events directory,
                # so clear out remaining directories
                del dirs[:]
                continue

            dirs.sort(key=nsort_key_func)
            files.sort(key=nsort_key_func)

            # If `find_gcd_in_dir` is True (i.e., not a string and not False),
            # look in current directory for all data-run GCD files
            thisdir_data_run_gcds = None
            if find_gcd_in_dir is True:
                thisdir_data_run_gcds = find_gcds_in_dirs(
                    dirpath, gcd_fname_re=DATA_GCD_FNAME_RE, recurse=False
                )

            for fname in files:
                retval = get_i3_events_file_info(dirpath=dirpath, fname=fname)
                if retval is not None:
                    yield retval


def main(description=__doc__):
    """Script interface to `extract_events` function: Parse command line args
    and call function."""

    hostname = gethostname()
    dflt = {}
    if hostname in ["schwyz", "luzern", "uri", "unterwalden"]:
        sim_gcd_dir = "/data/icecube/gcd"
        dflt["retro_gcd_dir"] = "/data/icecube/retro_gcd"
        dflt["data_gcd_dir"] = "/data/icecube/gcd"
        dflt["procs"] = cpu_count()
    elif hostname.endswith(".aci.ics.psu.edu"):
        sim_gcd_dir = "/gpfs/group/dfc13/default/gcd/mc"
        dflt["retro_gcd_dir"] = "/gpfs/group/dfc13/default/retro_gcd"
        dflt["data_gcd_dir"] = None
        dflt["procs"] = 1
    else:  # wisconsin?
        sim_gcd_dir = "/data/sim/DeepCore/2018/pass2/gcd"
        dflt["retro_gcd_dir"] = "~/retro_gcd"
        dflt["data_gcd_dir"] = None
        dflt["procs"] = 1
        raise ValueError("Unknown host: {}".format(hostname))

    dflt["sim_gcd"] = join(
        expand(sim_gcd_dir),
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
    )

    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="""i3 file(s) and/or directories to search for i3 files to extract""",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""If event was already extracted, overwrite existing object(s)
        (existing files will not be deleted, so excess objects not specified
        here will not be removed)""",
    )
    parser.add_argument(
        "--retro-gcd-dir",
        required=False,
        default=dflt["retro_gcd_dir"], #dflt.get("retro_gcd_dir", None),
        help="""Directory into which to store any extracted GCD info""",
    )
    parser.add_argument(
        "--sim-gcd",
        required=False,
        default=dflt.get("sim_gcd", None),
        help="""Specify an external GCD file or md5sum (as returned by
        `retro.i3processing.extract_gcd_frames`, i.e., the md5sum of an
        uncompressed i3 file containing _only_ the G, C, and D frames). It is
        not required to specify --gcd if the G, C, and D frames are embedded in
        all files specified by --i3-files. Any GCD frames within said files
        will also take precedent if --gcd _is_ specified.""",
    )
    parser.add_argument(
        "--data-gcd-dir",
        required=False,
        default=dflt.get("data_gcd_dir", None),
        help="""If data GCDs all live in one directory, specify it here.""",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        help="""Directory into which to store the extracted directories and
        files. If not specified, the directory where each i3 file is stored is
        used (a leaf directory is created with the same name as each i3 file
        but with .i3 and any compression extensions removed)."""
    )
    #parser.add_argument(
    #    "--photons",
    #    nargs="+",
    #    default=[],
    #    help="""Photon series names to extract from each event""",
    #)
    parser.add_argument(
        "--pulses",
        required=False,
        nargs="+",
        default=["SRTTWOfflinePulsesDC", "SplitInIcePulses"],
        help="""Pulse series names to extract from each event""",
    )
    parser.add_argument(
        "--recos",
        required=False,
        nargs="+",
        help="""Reco names to extract from each event. If not specified,
        "L5_SPEFit11", "LineFit_DC", and "retro_crs_prefit" (if the file name
        matches L6 processing or above) are extracted.""",
    )
    parser.add_argument(
        "--triggers",
        nargs="+",
        default=["I3TriggerHierarchy"],
        help="""Trigger hierarchy names to extract from each event""",
    )
    parser.add_argument(
        "--no-truth",
        action="store_true",
        help="""Do not extract truth information from Monte Carlo events""",
    )
    parser.add_argument(
        "--additional-keys",
        default=None,
        nargs="+",
        help="""Additional keys to extract from event I3 frame""",
    )
    parser.add_argument(
        "--procs",
        default=dflt["procs"],
        type=int,
        help="""Number of (sub)processes to use for converting files""",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    find_func_kwargs = {
        k: kwargs.pop(k)
        for k in getargspec(find_files_to_extract).args if k in kwargs
    }

    no_truth = kwargs.pop("no_truth")
    data_gcd_dir = kwargs.pop("data_gcd_dir", None)
    sim_gcd = kwargs.pop("sim_gcd", None)
    procs = kwargs.pop("procs", None)

    if data_gcd_dir:
        data_run_gcds = find_gcds_in_dirs(
            data_gcd_dir, gcd_fname_re=DATA_GCD_FNAME_RE
        )
    else:
        data_run_gcds = None

    kwargs["additional_keys"] = kwargs.pop("additional_keys", None)
    if not kwargs["additional_keys"]:
        from processing.samples.oscNext.verification.general_mc_data_harvest_and_plot import (
            L5_VARS, L6_VARS, L7_VARS
        )

    pool = Pool(procs)
    requests = []
    for fpath, gcd_fpath, fname_groupdict in find_files_to_extract(
        find_gcd_in_dir=True, data_run_gcds=data_run_gcds, **find_func_kwargs
    ):
        print(fpath)
        extract_events_kwargs = deepcopy(kwargs)
        extract_events_kwargs["i3_files"] = [fpath]

        is_data = fname_groupdict["kind"].lower() == "data"
        if is_data:
            assert gcd_fpath is not None
            extract_events_kwargs["gcd"] = gcd_fpath
        else:
            extract_events_kwargs["truth"] = not is_data and not no_truth
            extract_events_kwargs["gcd"] = sim_gcd

        level = int(fname_groupdict["level"])

        if "recos" not in extract_events_kwargs or not extract_events_kwargs["recos"]:
            recos = []
            if level >= 5:
                recos.extend(["LineFit_DC", "L5_SPEFit11"])
            if level >= 6:
                recos.append("retro_crs_prefit")
            extract_events_kwargs["recos"] = recos

        if not extract_events_kwargs["additional_keys"]:
            additional_keys = []
            if level >= 5:
                additional_keys.extend(L5_VARS.keys())
            if level >= 6:
                additional_keys.extend(L6_VARS.keys())
            if level >= 7:
                additional_keys.extend(L7_VARS.keys())
            extract_events_kwargs["additional_keys"] = sorted(additional_keys)

        requests.append(
            (
                extract_events_kwargs,
                pool.apply_async(
                    wrapped_extract_events, tuple(), extract_events_kwargs
                ),
            )
        )

    failed_i3_files = []
    for extract_events_kwargs, async_result in requests:
        retval = async_result.get()
        if not retval:
            failed_i3_files.append(extract_events_kwargs["i3_files"])

    pool.close()
    pool.join()

    if failed_i3_files:
        for failure in chain(*failed_i3_files):
            print('"{}"'.format(failure))

    print(
        "\n{} failures out of {} i3 files found that needed to be extracted".format(
            len(failed_i3_files), len(requests)
        )
    )


if __name__ == "__main__":
    main()
