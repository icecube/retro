#!/usr/bin/env python
# pylint: disable=wrong-import-position, no-member

"""
2D and 3D visualizations of a time- and DOM-independent (TDI) table
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from copy import deepcopy
import os
from os.path import abspath, basename, dirname, join
import re

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pyfits
import yt

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DETECTOR_GEOM_FILE, Cart3DCoord


DOM_RADIUS_M = 0.3302


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''Path to NPY file containing DOM locations as
        (string, dom, x, y, z) entries'''
    )
    #parser.add_argument(
    #    '--tables-dir', metavar='DIR', type=str,
    #    default='/data/icecube/retro_tables/full1000',
    #    help='''Directory containing retro tables''',
    #)
    parser.add_argument(
        '--table-path', metavar='DIR', type=str, required=True,
        help='''Path to one of the tables, e.g.
        `qdeficit_cart_table_20x20x20_os_r1_zen1_test`''',
    )
    parser.add_argument(
        '--slices', action='store_true',
        help='''Plot slices in each plane''',
    )
    parser.add_argument(
        '--projections', action='store_true',
        help='''Plot projections in each plane''',
    )
    parser.add_argument(
        '--plot-3d', action='store_true',
        help='''Plot 3D density''',
    )
    args = parser.parse_args()
    return args


def visualize_tables(table_path, geom_file, slices=True, projections=True,
                     plot_3d=True):
    tables_dir = dirname(table_path)
    table_fname = basename(table_path)
    tables_basename = table_fname
    for name in ['survival_prob', 'avg_photon_x', 'avg_photon_y', 'avg_photon_z']:
        tables_basename = tables_basename.replace('_' + name + '.fits', '')
    if tables_basename[-1] == '_':
        tables_basename = tables_basename[:-1]

    det_string_depth_xyz = np.load(geom_file)

    num_doms_in_detector = np.prod(det_string_depth_xyz.shape[:2])

    data = {}
    fname = '%s_survival_prob.fits' % tables_basename
    fpath = join(tables_dir, fname)
    with pyfits.open(fpath) as fits_file:
        survival_prob = fits_file[0].data
        ma = survival_prob.max()
        print('Max survival probability         :', ma)
        mi = survival_prob.min()
        print('Min survival probability         :', mi)
        mi_nonzero = survival_prob[survival_prob != 0].min()
        print('Min non-zero survival probability:', mi_nonzero)
        data['density'] = (survival_prob, 'kg/m**3')
        xyz_shape = fits_file[1].data
        lims = fits_file[2].data
        doms_used = fits_file[3].data

        # If 3D, dims represent: (string numbers, depth indices, (x, y, z))
        if len(doms_used.shape) == 3:
            doms_used = np.stack((doms_used[:, :, 0].flatten(),
                                  doms_used[:, :, 1].flatten(),
                                  doms_used[:, :, 2].flatten())).T

        nx, ny, nz = xyz_shape
        xlims = lims[0, :]
        ylims = lims[1, :]
        zlims = lims[2, :]
        print('x lims:', xlims)
        print('y lims:', ylims)
        print('z lims:', zlims)
        print('(nx, ny, nz):', xyz_shape)
        num_doms_used = doms_used.shape[0]
        print('num doms used:', num_doms_used)
        print('doms used:', doms_used.shape)

    if slices:
        mask = survival_prob > 0 #(ma / 10000000)
        avg_photon_info = {}
        for dim in ['x', 'y', 'z']:
            fname = '%s_avg_photon_%s.fits' % (tables_basename, dim)
            fpath = join(tables_dir, fname)
            with pyfits.open(fpath) as fits_file:
                d = np.zeros_like(survival_prob)
                d[mask] = fits_file[0].data[mask]
                #d = -fits_file[0].data
                avg_photon_info[dim] = d
                data['velocity_' + dim] = (d, 'm/s')
        avg_photon_info = Cart3DCoord(**avg_photon_info)
        del mask

    bbox = lims
    ds = yt.load_uniform_grid(data, domain_dimensions=(nx, ny, nz), bbox=bbox,
                              nprocs=4)

    savefig_kw = dict(name=join(tables_dir, tables_basename), suffix='png',
                      mpl_kwargs=dict(dpi=300))
    plots = []

    sphere_kwargs = dict(
        radius=(5*DOM_RADIUS_M, 'cm'),
        coord_system='data',
        circle_args=dict(color=(0, 0.8, 0), linewidth=1, alpha=0.3)
    )

    if projections:
        for normal in ['x', 'y', 'z']:
            prj = yt.ProjectionPlot(ds, normal, 'density')
            prj.set_log('density', False)
            prj.set_cmap('density', 'inferno')

            # Display all doms in the detector
            for depth_xyz in det_string_depth_xyz:
                for xyz in depth_xyz:
                    prj.annotate_sphere(xyz, **sphere_kwargs)

            # Display only doms used (if subset of the detector)
            if num_doms_used != num_doms_in_detector:
                kw = deepcopy(sphere_kwargs)
                kw['radius'] = (15*DOM_RADIUS_M, 'cm')
                kw['circle_args']['alpha'] = 1
                for depth_xyz in doms_used:
                    prj.annotate_sphere(depth_xyz, **kw)

            prj.save(**savefig_kw)
            plots.append(prj)

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

    if slices:
        skw = deepcopy(sphere_kwargs)
        skw['circle_args']['color'] = (0.8, 0, 0)
        if num_doms_used != num_doms_in_detector:
            center = np.mean(doms_used, axis=0)
        else:
            center = (0, 0, 0)
            #center = det_string_depth_xyz[35, 47]
            #cut_plane_strings = [1 - s for s in [6, 12, 27, 36, 45, 54, 62, 69, 75]]
            #normal =
            #north_vector = (0, 0, 1)

        for normal in ['x', 'y', 'z']:
            #if normal == 'x':
            #    plt.
            slc = yt.SlicePlot(ds, normal=normal, fields='density',
                               center=center)
            slc.set_cmap('density', 'octarine')
            #slc.set_log('density', False)

            for depth_xyz in det_string_depth_xyz:
                for xyz in depth_xyz:
                    slc.annotate_sphere(xyz, **skw)

            if num_doms_used != num_doms_in_detector:
                kw = deepcopy(skw)
                kw['radius'] = (15*DOM_RADIUS_M, 'cm')
                kw['circle_args']['alpha'] = 1
                for depth_xyz in doms_used:
                    slc.annotate_sphere(depth_xyz, **kw)

            nskip = 10
            kw = dict(factor=nskip, scale=1e3)
            if normal == 'x':
                slc.annotate_quiver('velocity_y', 'velocity_z', **kw)
            elif normal == 'y':
                slc.annotate_quiver('velocity_z', 'velocity_x', **kw)
            elif normal == 'z':
                slc.annotate_quiver('velocity_x', 'velocity_y', **kw)

            #slc.annotate_grids(cmap=None)
            slc.save(**savefig_kw)
            plots.append(slc)

    return ds, plots


if __name__ == '__main__':
    ds, plots = visualize_tables(**vars(parse_args())) # pylint: disable=invalid-name
