#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name


from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import os
from os.path import abspath, basename, dirname, getsize, isdir, isfile, islink
import shutil
import sys

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, mkdir, nsort_key_func
from retro.i3processing.extract_events import GENERIC_I3_FNAME_RE


GCD_DIR = "/data/icecube/gcd"


def find_unique_gcds():
    """Find unique GCD files in data"""
    with open(expand("~/all_data_gcd_files.txt"), "r") as f:
        fpaths = [expand(l.strip()) for l in f.readlines()]

    original_num_files = len(fpaths)
    root_infos = {}
    original_size = 0
    final_size = 0
    for fpath in fpaths:
        base = basename(fpath)
        size = getsize(fpath)
        fname_info = GENERIC_I3_FNAME_RE.match(base).groupdict()
        root = fname_info["base"]
        compext = fname_info.get("compext", None)
        original_size += size
        if root not in root_infos:
            root_infos[root] = []
            final_size += size

        root_infos[root].append(
            dict(
                fpath=fpath,
                base=base,
                root=root,
                compext=compext,
                size=size,
            )
        )

    root_infos = OrderedDict(
        [(rn, root_infos[rn]) for rn in sorted(root_infos.keys(), key=nsort_key_func)]
    )
    final_num_files = len(root_infos)

    #unequal_sizes = False
    #for rn, finfos in root_infos.items():
    #    file_paths = []
    #    file_sizes = []
    #    for fpath, fsize in finfos:
    #        file_paths.append(fsize)
    #        file_sizes.append(fsize)
    #    file_sizes = np.array(file_sizes)
    #    if not np.all(file_sizes == file_sizes[0]):
    #        unequal_sizes = True
    #        for file_path, file_size in finfos:
    #            print("{:14d} b : {}".format(file_size, file_path))

    print(
        "original number of files = {}, final = {}".format(
            original_num_files, final_num_files
        )
    )

    print(
        "original size = {:.0f} GiB, final size = {:.0f} GiB".format(
            original_size/(1024**3), final_size/(1024**3)
        )
    )

    return root_infos


#def recompress_file(info):
#    if info["compext"] != ":


def centralize_gcds(root_infos, gcd_dir=GCD_DIR):
    """Move GCD files to a single directory, if they don't already exist there.

    Compression extensions should be ignored, so only one version of each GCD
    exists.

    Parameters
    ----------
    root_infos : mapping
    gcd_dir : str, optional

    """
    gcd_dir = expand(gcd_dir)
    mkdir(gcd_dir)

    existing_fnames = os.listdir(gcd_dir)
    existing_roots = set()
    for fname in existing_fnames:
        match = GENERIC_I3_FNAME_RE.match(fname)
        if not match:
            continue
        groupdict = match.groupdict()
        existing_roots.add(groupdict["base"])

    for root, infos in root_infos.items():
        for info in infos:
            is_link = islink(info["fpath"])
            is_file = isfile(info["fpath"])

            if is_link:
                if is_file:  # link to an existing file
                    if root not in existing_roots:
                        shutil.copy2(info["fpath"], gcd_dir, follow_symlinks=True)
                        existing_roots.add(root)
                else:  # bad link (to nothing, or to a directory)
                    if not isdir(info["fpath"]):
                        print(f'os.remove({info["fpath"]})')
                        os.remove(info["fpath"])
            else:
                if root in existing_roots:
                    if is_file:
                        print(f'os.remove({info["fpath"]})')
                        os.remove(info["fpath"])
                else:
                    print(f'shutil.move({info["fpath"]}, {gcd_dir})')
                    shutil.move(info["fpath"], gcd_dir)
                    existing_roots.add(root)


if __name__ == "__main__":
    find_unique_gcds()
