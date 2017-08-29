#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
3D visualization of a time- and DOM-independent table
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from copy import deepcopy
import os
from os.path import abspath, dirname, isdir, isfile, join

import numpy as np
import pyfits
import yt

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DETECTOR_GEOM_FILE


DOM_RADIUS_M = 0.3302
TEST_DOM_COORD = [46.29000092,  -34.88000107, -350.05999756]



def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''Path to NPY file containing DOM locations as
        (string, dom, x, y, z) entries'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/data/icecube/retro_tables/full1000',
        help='''Directory containing retro tables''',
    )
    parser.add_argument(
        '--tables-basename', metavar='DIR', type=str, required=True,
        help='''Basename of the tables, e.g.
        `qdeficit_cart_table_20x20x20_os_r1_zen1_test`''',
    )
    parser.add_argument(
        '--plot-slices', action='store_true',
        help='''Plot slices in each plane''',
    )
    parser.add_argument(
        '--plot-projections', action='store_true',
        help='''Plot projections in each plane''',
    )
    parser.add_argument(
        '--plot-3d', action='store_true',
        help='''Plot 3D density''',
    )
    args = parser.parse_args()
    return args


def main(tables_dir, tables_basename, geom_file, plot_slices=True,
         plot_projections=True, plot_3d=True):
    geom = np.load(geom_file)
    data = {}
    #for dim in ['x', 'y', 'z']:
    #    fname = '%s_avg_photon_%s.fits' % (tables_basename, dim)
    #    fpath = join(tables_dir, fname)
    #    with pyfits.open(fpath) as fits_file:
    #        data['particle_velocity_' + dim] = (fits_file[0].data, 'm/s')

    fname = '%s_survival_prob.fits' % tables_basename
    fpath = join(tables_dir, fname)
    with pyfits.open(fpath) as fits_file:
        d = fits_file[0].data
        mi = d.min()
        ma = d.max()
        print('mi: %e, ma: %e' % (mi, ma))
        data['density'] = (d, 'kg/m**3')

    bbox = np.array([
        (-700, 700), # x
        (-650, 650), # y
        (-650, 650)  # z
    ])
    nx, ny, nz = [100]*3
    ds = yt.load_uniform_grid(data, domain_dimensions=(nx, ny, nz), bbox=bbox,
                              nprocs=4)

    savefig_kw = dict(name=join(tables_dir, tables_basename), suffix='png',
                      mpl_kwargs=dict(dpi=300))
    plots = []

    sphere_kwargs = dict(
        radius=(5*DOM_RADIUS_M, 'cm'),
        coord_system='data',
        circle_args=dict(color=(0, 0.8, 0), linewidth=1, alpha=0.3, facecolor=(0,0.8,0))
    )

    if plot_projections:
        for normal in ['x', 'y', 'z']:
            prj = yt.ProjectionPlot(ds, normal, 'density')
            prj.set_log('density', False)
            prj.set_cmap('density', 'inferno')

            if 'test' in tables_basename:
                kw = deepcopy(sphere_kwargs)
                kw['radius'] = (20*DOM_RADIUS_M, 'cm')
                kw['circle_args']['alpha'] = 1
                prj.annotate_sphere(TEST_DOM_COORD, **kw)
            for depth_xyz in geom:
                for xyz in depth_xyz:
                    prj.annotate_sphere(xyz, **sphere_kwargs)

            prj.save(**savefig_kw)
            plots.append(prj)

    if plot_slices:
        skw = deepcopy(sphere_kwargs)
        skw['circle_args']['color'] = 'black'
        skw['circle_args']['facecolor'] = 'black'
        for normal in ['x', 'y', 'z']:
            slc = yt.SlicePlot(ds, normal=normal, fields='density', center=TEST_DOM_COORD)
            slc.set_cmap('density', 'octarine')
            #slc.set_log('density', False)
            if 'test' in tables_basename:
                kw = deepcopy(skw)
                kw['radius'] = (20*DOM_RADIUS_M, 'cm')
                kw['circle_args']['alpha'] = 1
                slc.annotate_sphere(TEST_DOM_COORD, **kw)
            for depth_xyz in geom:
                for xyz in depth_xyz:
                    slc.annotate_sphere(xyz, **skw)

            #slc.annotate_grids(cmap=None)
            slc.save(**savefig_kw)
            plots.append(slc)

    if plot_3d:
        # Choose a vector representing the viewing direction.
        L = [-0.5, -0.5, -0.5]

        # Define the center of the camera to be the domain center
        c = ds.domain_center[0]
        #c = (1400*100, 1300*100, 1300*100)

        # Define the width of the image
        W = 1.0*ds.domain_width[0]

        # Define the number of pixels to render
        Npixels = 2048

        sc = yt.create_scene(ds, 'density')
        source = sc[0]
        source.log_field = False

        tf = yt.ColorTransferFunction((0, ma), grey_opacity=True)
        tf.map_to_colormap(0, ma, scale=1.0, colormap='inferno')

        source.set_transfer_function(tf)

        sc.add_source(source)

        cam = sc.add_camera()
        cam.width = W
        cam.center = c
        cam.normal_vector = L
        cam.north_vector = [0, 0, 1]
        cam.position = (1400, 1300, 1300)

        #sc.show(sigma_clip=4)

        sc.save(savefig_kw['name'])

        plots.append(sc)

    return ds, plots


if __name__ == '__main__':
    ds, plots = main(**vars(parse_args()))
