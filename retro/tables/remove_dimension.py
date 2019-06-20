#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Remove a dimension from a table.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["remove_dimension", "main"]

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
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
import sys

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, mkdir


def remove_dimension(input_file, output_file, dim_name):
    """
    Parameters
    ----------
    input_file : str
        Path to input file (table)

    output_file : str
        Path to output file (table)

    dim_name : str
        Dimension to remove from the intput table

    """
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

    dim_num = [i for i, n in enumerate(input_binning.dtype.names) if n == dim_name][0]
    output_binning = input_binning[
        [n for n in input_binning.dtype.names if n != dim_name]
    ]

    # Save the binning to the output directory
    np.save(join(output_dir, "binning.npy"), output_binning)

    # Legacy way of storing bin edges: store each dim individually
    for d_name in output_binning.dtype.names:
        bin_edges_fpath = join(output_dir, "{}_bin_edges.npy".format(d_name))
        np.save(bin_edges_fpath, output_binning[d_name])

    # If we find the removed dimension's bin edges in output dir, remove that file
    bin_edges_fpath = join(output_dir, "{}_bin_edges.npy".format(dim_name))
    if isfile(bin_edges_fpath):
        remove(bin_edges_fpath)

    output_shape = tuple(n for i, n in enumerate(input_table.shape) if i != dim_num)
    output_table = np.empty(shape=output_shape, dtype=input_table.dtype)
    #output_table = np.memmap(
    #    output_file, dtype=input_table.dtype, mode="w+", shape=output_shape
    #)

    # Perform the summation over the dimension to be removed

    # Note that setting dtype to float64 causes accumulator to be double
    # precision, even if output table is not
    input_table.sum(axis=dim_num, dtype=np.float64, out=output_table)

    np.save(output_file, output_table)


def main(description=__doc__):
    """Command-line interface to `remove_dimension` function."""
    parser = ArgumentParser(description=description)
    parser.add_argument("--input-file", required=True, help="Input table")
    parser.add_argument("--output-file", required=True, help="Output file")
    parser.add_argument(
        "--dim-name", required=True, help="Dimension to be removed from the input file"
    )
    args = parser.parse_args()
    kwargs = vars(args)
    remove_dimension(**kwargs)


if __name__ == "__main__":
    main()
