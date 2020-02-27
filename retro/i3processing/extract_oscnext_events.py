#!/usr/bin/env python
# pylint: disable=wrong-import-position


"""
Find and extract oscNext events to retro (native python/numpy-friendly) format.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "OSCNEXT_I3_FNAME_RE",
    "test_OSCNEXT_I3_FNAME_RE",
    "DATA_GCD_FNAME_RE",
    "find_files_to_extract",
    "main",
]

from argparse import ArgumentParser
from copy import deepcopy
from inspect import getargspec
from itertools import chain
from multiprocessing import Pool
from os import walk
from os.path import abspath, dirname, isdir, isfile, join
import re
from socket import gethostname
import sys

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3processing.extract_events import wrapped_extract_events
from retro.utils.misc import expand, nsort_key_func


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


DATA_GCD_FNAME_RE = re.compile(
    r".*_IC86\.20(?P<season>[0-9]+).*_data_Run(?P<run>[0-9]+).*GCD.*\.i3(\..*)*",
    flags=re.IGNORECASE,
)

HOST = gethostname()


def test_OSCNEXT_I3_FNAME_RE():
    """Unit tests for OSCNEXT_I3_FNAME_RE."""
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
                        '"{key}": actual_val = "{actual_val}" but ref_val = "{ref_val}"'.format(
                            key=key, actual_val=actual_val, ref_val=ref_val
                        )
                    )
            if err_msg:
                raise ValueError("; ".join(err_msg))
        except Exception:
            sys.stderr.write('Failure on test input = "{}"\n'.format(test_input))
            raise


def find_data_gcds_in_dirs(rootdirs, recurse=True):
    """Find data run GCD files in directories.

    Parameters
    ----------
    rootdirs : str or iterable thereof
    recurse : bool

    Returns
    -------
    data_run_gcds : dict
        Keys are <tuple>(<str>2-digit season, <str>run number) and values are
        <str> path to corresponding GCD file

    """
    if isinstance(rootdirs, str):
        rootdirs = [rootdirs]
    rootdirs = [expand(rootdir) for rootdir in rootdirs]

    data_run_gcds = {}
    for rootdir in rootdirs:
        for dirpath, dirs, files in walk(rootdir):
            if recurse:
                dirs.sort(key=nsort_key_func)
            else:
                del dirs[:]
            files.sort(key=nsort_key_func)

            for fname in files:
                gcd_match = DATA_GCD_FNAME_RE.match(fname)
                if gcd_match:
                    gcd_groupdict = gcd_match.groupdict()
                    data_run_gcds[
                        (gcd_groupdict["season"], gcd_groupdict["run"])
                    ] = join(dirpath, fname)

    return data_run_gcds


def find_files_to_extract(
    rootdirs, overwrite, find_gcd_in_dir=False, data_run_gcds=None
):
    """Find missing, bad, or old extracted pulse series and print the paths of
    the corresponding events directories.

    Parameters
    ----------
    rootdirs : str

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

    gcd_dict : dict or None, optional
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
    if isinstance(rootdirs, str):
        rootdirs = [rootdirs]
    rootdirs = [expand(rootdir) for rootdir in rootdirs]

    # If `find_gcd_in_dir` is a string, interpret as a directory and search for
    # GCD's in that directory (recursively)
    found_data_run_gcds = None
    if isinstance(find_gcd_in_dir, str):
        find_gcd_in_dir = expand(find_gcd_in_dir)
        assert isdir(find_gcd_in_dir), str(find_gcd_in_dir)
        found_data_run_gcds = find_data_gcds_in_dirs(find_gcd_in_dir, recurse=True)

    for rootdir in rootdirs:
        for dirpath, dirs, files in walk(rootdir, followlinks=True):
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
                thisdir_data_run_gcds = find_data_gcds_in_dirs(dirpath, recurse=False)

            for fname in files:
                fname_match = OSCNEXT_I3_FNAME_RE.match(fname)

                if not fname_match:
                    continue

                fname_groupdict = fname_match.groupdict()

                i3_retro_dir = join(dirpath, fname_groupdict["basename"])
                if (
                    not overwrite
                    and isdir(i3_retro_dir)
                    and isfile(join(i3_retro_dir, "events.npy"))
                ):
                    continue

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

                yield fpath, gcd_fpath, fname_groupdict


def main(description=__doc__):
    """Script interface to `extract_events` function: Parse command line args
    and call function."""

    dflt = {}
    if HOST in ["schwyz", "luzern", "uri", "unterwalden"]:
        sim_gcd_dir = "/data/icecube/gcd"
        dflt["retro_gcd_dir"] = "/data/icecube/retro_gcd"
        dflt["data_gcd_dir"] = None
    elif HOST.endswith(".aci.ics.psu.edu"):
        sim_gcd_dir = "/gpfs/group/dfc13/default/gcd/mc"
        dflt["retro_gcd_dir"] = "/gpfs/group/dfc13/default/retro_gcd"
        dflt["data_gcd_dir"] = None
    else:  # wisconsin
        sim_gcd_dir = "/data/sim/DeepCore/2018/pass2/gcd"
        dflt["retro_gcd_dir"] = "~/retro_gcd"
        dflt["data_gcd_dir"] = None

    dflt["sim_gcd"] = join(
        expand(sim_gcd_dir),
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz",
    )

    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--rootdirs",
        nargs="+",
        required=True,
        help="""Directories to search for i3 files to extract""",
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
        default=dflt.get("retro_gcd_dir", None),
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
        default="L5_oscNext_bool",
        nargs="+",
        help="""Additional keys to extract from event I3 frame""",
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

    if data_gcd_dir:
        data_run_gcds = find_data_gcds_in_dirs(data_gcd_dir)
    else:
        data_run_gcds = None

    pool = Pool()

    requests = []
    for fpath, gcd_fpath, fname_groupdict in find_files_to_extract(
        find_gcd_in_dir=True, data_run_gcds=data_run_gcds, **find_func_kwargs
    ):
        extract_events_kwargs = deepcopy(kwargs)
        extract_events_kwargs["i3_files"] = [fpath]

        is_data = fname_groupdict["kind"].lower() == "data"
        if is_data:
            assert gcd_fpath is not None
            extract_events_kwargs["gcd"] = gcd_fpath
        else:
            extract_events_kwargs["truth"] = not is_data and not no_truth
            extract_events_kwargs["gcd"] = sim_gcd

        if "recos" not in extract_events_kwargs or not extract_events_kwargs["recos"]:
            level = int(fname_groupdict["level"])
            recos = []
            if level >= 5:
                recos.extend(["LineFit_DC", "L5_SPEFit11"])
            if level >= 6:
                recos.append("retro")
            extract_events_kwargs["recos"] = recos

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

    if failed_i3_files:
        for failure in chain(*failed_i3_files):
            print('"{}"'.format(failure))

    print("\n{} failures out of {} i3 files found that needed to be extracted".format(
        len(failed_i3_files), len(requests))
    )


if __name__ == "__main__":
    test_OSCNEXT_I3_FNAME_RE()
    main()
