#!/usr/bin/env python


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

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, join_struct_arrays, mkdir, nsort_key_func

def get_qtot_by_nchan(root_dir, pulse_series_name):
    for dirpath, dirs, files in walk(root_dir, followlinks=True):
        dirs.sort(key=nsort_key_func)

        if "events.npy" not in files:
            continue

        this_pulses = pickle.load(
            file(join(dirpath, "pulses", "{}.pkl".format(pulse_series_name)), "r")
        )

        


def qtot_by_nchan(mc_dirs, data_dirs):
    if isinstance(mc_dirs, string_types):
        mc_dirs = [mc_dirs]
    if isinstance(data_dirs, string_types):
        data_dirs = [data_dirs]

    data_vals = []
    for data_dir in data_dirs:
        data_a = np.load(data_f, mmap_mode="r")
        
