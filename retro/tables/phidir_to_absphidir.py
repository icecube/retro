#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Convert deltaphidir to absdeltaphidir
"""

from __future__ import absolute_import, division, print_function

__all__ = ["deltaphidir_to_absdeltaphidir", "main"]

__author__ = "J.L. Lanfranchi"

__license__ = """Copyright 2019 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from argparse import ArgumentParser
from os.path import abspath, dirname, isdir, join
import sys

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, mkdir


def deltaphidir_to_absdeltaphidir(input_file, output_file):
    """
    Parameters
    ----------
    input_file : str
        Path to input file (table)

    output_file : str
        Path to output file (table)

    """
    dim_name = "deltaphidir"

    input_file = expand(input_file)
    output_file = expand(output_file)

    input_dir = dirname(input_file)
    output_dir = dirname(output_file)

    if abspath(output_dir) == abspath(input_dir):
        raise ValueError("Will not allow output dir to be same as input dir")

    if not isdir(output_dir):
        mkdir(output_dir)

    input_table = np.load(input_file, mmap_mode="r")
    input_binning = np.load(join(input_dir, "binning.npy"))

    dim_num = list(input_binning.dtype.names).index(dim_name)

    output_dtype_spec = []
    output_bin_edges = []
    for dim_descr in input_binning.dtype.descr:
        dname, dt, shape = dim_descr
        orig_dim_be = input_binning[dname]
        if dname == dim_name:
            be_in_pi = ((orig_dim_be + np.pi) % (2*np.pi)) - np.pi
            closest_be_to_zero = np.min(np.abs(be_in_pi))

            if np.isclose(closest_be_to_zero, 0):
                raise NotImplementedError()
            else:
                output_dim_shape = (int((shape[0] - 1) / 2 + 1),)
                output_dim_be = (
                    np.abs(orig_dim_be[orig_dim_be < 0])[::-1]
                    - np.mean(np.diff(orig_dim_be)) / 2
                )
            output_dim_be -= output_dim_be[0]
            output_dim_be /= output_dim_be[-1] / np.pi
            output_bin_edges.append(tuple(output_dim_be.tolist()))
            output_dtype_spec.append((dname, dt, output_dim_shape))
        else:
            output_dtype_spec.append(dim_descr)
            output_bin_edges.append(orig_dim_be.tolist())

    output_binning = np.array(tuple(output_bin_edges), dtype=output_dtype_spec)

    output_shape = tuple(dim_spec[2][0] - 1 for dim_spec in output_binning.dtype.descr)
    output_table = np.zeros(shape=output_shape, dtype=np.float64)

    mapping = []

    for input_bin_idx, (input_le, input_ue) in enumerate(
        zip(input_binning[dim_name][:-1], input_binning[dim_name][1:])
    ):
        input_wid = input_ue - input_le

        for output_bin_idx, (output_le, output_ue) in enumerate(
            zip(output_binning[dim_name][:-1], output_binning[dim_name][1:])
        ):
            overlap_fract = 0.

            for sign in [-1, +1]:
                if sign > 0:
                    actual_output_le = output_le
                else:
                    actual_output_le = -output_ue

                # Compute input bin edges relative to the lower output bin edge
                input_rel_le = ((input_le - actual_output_le) + np.pi) % (2*np.pi) - np.pi
                input_rel_ue = ((input_ue - actual_output_le) + np.pi) % (2*np.pi) - np.pi

                output_wid = abs(output_ue - output_le)

                input_clipped_rel_edges = np.clip(
                    [input_rel_le, input_rel_ue],
                    a_min=0,
                    a_max=output_wid,
                )

                overlap_fract = np.diff(input_clipped_rel_edges)[0] / input_wid
                if overlap_fract > 0:
                    dupe_idx = None
                    for idx, (obi, ibi, ofr) in enumerate(mapping):
                        if obi == output_bin_idx and ibi == input_bin_idx:
                            dupe_idx = idx
                            overlap_fract += ofr
                    entry = (output_bin_idx, input_bin_idx, overlap_fract)
                    if dupe_idx is None:
                        mapping.append(entry)
                    else:
                        mapping[dupe_idx] = entry

    output_slicer = [slice(None) for _ in output_binning.dtype.names]
    input_slicer = [slice(None) for _ in input_binning.dtype.names]

    for output_bin_idx, input_bin_idx, overlap_fract in mapping:
        output_slicer[dim_num] = output_bin_idx
        input_slicer[dim_num] = input_bin_idx
        output_table[output_slicer] += input_table[input_slicer]

    # Save the binning to the output directory
    np.save(join(output_dir, "binning.npy"), output_binning)

    # Legacy way of storing bin edges: store each dim individually
    for d_name in output_binning.dtype.names:
        bin_edges_fpath = join(output_dir, "{}_bin_edges.npy".format(d_name))
        np.save(bin_edges_fpath, output_binning[d_name])

    # Save the table
    np.save(output_file, output_table)


def main(description=__doc__):
    """Command-line interface to `remove_dimension` function."""
    parser = ArgumentParser(description=description)
    parser.add_argument("--input-file", required=True, help="Input table")
    parser.add_argument("--output-file", required=True, help="Output file")
    args = parser.parse_args()
    kwargs = vars(args)
    deltaphidir_to_absdeltaphidir(**kwargs)


if __name__ == "__main__":
    main()
