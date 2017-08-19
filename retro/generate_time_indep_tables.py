#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Create time-independent "whole-detector" table.

First define a cartesian grid covering all of the IceCube fiducial volume and
sum for each voxel info from the tables placed at the locations of each DOM.

The new table is in (x, y, z), independent of time, for purposes of matching
with a hypothesis (which will be defined in a small subset of the voxels) to
determine the total charge expected for that hypothesis.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import os
from os.path import abspath, dirname
import time

import numpy as np
import pyfits

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DETECTOR_GEOM_FILE
from retro import edges_to_binspec, expand, extract_photon_info


IC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_IC_DOM{dom:d}_r_cz_t_angles.fits'
)

DC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_DC_DOM{dom:d}_r_cz_t_angles.fits'
)


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--nx', type=int,
        help='''Number of x bins'''
    )
    parser.add_argument(
        '--ny', type=int,
        help='''Number of y bins'''
    )
    parser.add_argument(
        '--nz', type=int,
        help='''Number of z bins'''
    )
    parser.add_argument(
        '--xlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-700, 700),
        help='''limits on binning in x dimension'''
    )
    parser.add_argument(
        '--ylims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on binning in y dimension'''
    )
    parser.add_argument(
        '--zlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on binning in z dimension'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/data/icecube/retro_tables/full1000',
        help='''Directory containing retro tables''',
    )
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''NPY file containing DOM locations as (string, dom, x, y, z)
        entries'''
    )
    args = parser.parse_args()
    return args


def main(tables_dir, geom_file, xlims, ylims, zlims, nx, ny, nz):
    """Perform scans and minimization for events.

    Parameters
    ----------
    tables_dir : string
        Path to directory containing the retro tables

    geom_file : string
        File containing detector geometry

    xlims, ylims, zlims : sequence of two floats
        Lower and upper limits for volume that will be divided.

    nx, ny, nz : int
        Number of voxels in each dimension. Note that there will be e.g. ``nx +
        1`` bin edges in the x-dimension.

    """
    shape = (nx, ny, nz)
    x_extent = max(xlims) - min(xlims)
    x_bin_num_factor = x_extent / nx

    y_extent = max(ylims) - min(ylims)
    y_bin_num_factor = y_extent / ny

    z_extent = max(zlims) - min(zlims)
    z_bin_num_factor = z_extent / nz

    # Create double precision arrays for accumulation
    counts = np.zeros(shape=shape, dtype=np.float64)
    avg_photon_x = np.zeros(shape=shape, dtype=np.float64)
    avg_photon_y = np.zeros(shape=shape, dtype=np.float64)
    avg_photon_z = np.zeros(shape=shape, dtype=np.float64)

    # Load detector geometry
    geom = np.load(geom_file)

    for table_kind in ['ic', 'dc']:
        if table_kind == 'ic':
            table_fpath_proto = IC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            string_sl = slice(None, 79)
        elif table_kind == 'dc':
            table_fpath_proto = DC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            string_sl = slice(79, 86)

        for dom_depth_idx in dom_depth_indices:
            # Get the (x, y, z) coords for all DOMs of this type and at this
            # "depth index"
            doms_xyz = geom[string_sl, ...]

            # Load the 4D table
            fpath = table_fpath_proto.format(
                tables_dir=tables_dir, dom=dom_depth_idx
            )
            photon_info, bin_edges = extract_photon_info(
                fpath=expand(fpath), dom_depth_index=dom_depth_idx
            )

            bin_min, bin_max, num_bins = edges_to_binspec(edges=bin_edges)

            #num_bins_upsamp = BinningCoords(
            #    t=num_bins.t,
            #    r=(num_bins.r if num_bins.r > 100 else
            #       int(num_bins.r * np.ceil(100 / num_bins.r))),
            #    theta=(num_bins.theta if num_bins.theta > 100 else
            #           int(num_bins.theta * np.ceil(100 / num_bins.theta))),
            #    phi=100
            #)

            for r_bin_num in range(num_bins.r):
                for theta_bin_num in range(num_bins.theta):
                    pass



if __name__ == '__main__':
    main(**vars(parse_args()))
