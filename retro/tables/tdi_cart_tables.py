# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for time- and DOM-independent table.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    TDI_TABLE_FNAME_PROTO
    TDI_TABLE_FNAME_RE
    TDICartTable
'''.split()

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from copy import deepcopy
from glob import glob
from os.path import abspath, basename, dirname, isdir, join
import re
import sys
from time import time

import numpy as np

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.tables.pexp_xyz import pexp_xyz
from retro.utils.misc import (
    expand, force_little_endian, generate_anisotropy_str
)


TDI_TABLE_FNAME_PROTO = [
    (
        'retro_tdi_table'
        '_{tdi_hash:s}'
        '_binmap_{binmap_hash:s}'
        '_geom_{geom_hash:s}'
        '_domtbl_{dom_tables_hash:s}'
        '_times_{times_str:s}'
        '_x{x_min:.3f}_{x_max:.3f}'
        '_y{y_min:.3f}_{y_max:.3f}'
        '_z{z_min:.3f}_{z_max:.3f}'
        '_bw{binwidth:.9f}'
        '_anisot_{anisotropy_str:s}'
        '_icqe{ic_dom_quant_eff:.5f}'
        '_dcqe{dc_dom_quant_eff:.5f}'
        '_icexp{ic_exponent:.5f}'
        '_dcexp{dc_exponent:.5f}'
        '_{table_name:s}'
        '.fits'
    )
]
"""Time- and DOM-independent (TDI) table file names follow this template"""

TDI_TABLE_FNAME_RE = re.compile(
    r'^retro_tdi_table'
    r'_(?P<tdi_hash>[^_]+)'
    r'_binmap_(?P<binmap_hash>[^_]+)'
    r'_geom_(?P<geom_hash>[^_]+)'
    r'_domtbl_(?P<dom_tables_hash>[^_]+)'
    r'_times_(?P<times_str>[^_]+)'
    r'_x(?P<x_min>[^_]+)_(?P<x_max>[^_]+)'
    r'_y(?P<y_min>[^_]+)_(?P<y_max>[^_]+)'
    r'_z(?P<z_min>[^_]+)_(?P<z_max>[^_]+)'
    r'_bw(?P<binwidth>[^_]+)'
    r'_anisot_(?P<anisotropy>.+?)'
    r'_icqe(?P<ic_dom_quant_eff>.+?)'
    r'_dcqe(?P<dc_dom_quant_eff>.+?)'
    r'_icexp(?P<ic_exponent>.+?)'
    r'_dcexp(?P<dc_exponent>.+?)'
    r'_(?P<table_name>(avg_photon_x|avg_photon_y|avg_photon_z|survival_prob))'
    r'\.fits$'
    , re.IGNORECASE
)
"""Time- and DOM-independent (TDI) table file names can be found / interpreted
using this regex"""


# TODO: convert to using exponent rather than scale (scale will be applied via
# dom_quant_eff when generating the TDI table in the first place; at this
# stage, we want to go either up or down with probabilities, so a single
# exponent should be appropriate, while a scale factor can exceed 1 for
# probabilities).
class TDICartTable(object):
    """Load and use information from a time- and DOM-independent Cartesian
    (x, y, z)-binned Retro table.

    The parameters used to generate the table are passed at instantiation of
    this class to determine which table(s) to load when a table is requested
    (multiple "tables" are loaded if a single table is generated from multiple
    smaller tiles meant to be stitched together).

    Parameters
    ----------
    tables_dir : string

    proto_tile_hash : string
        Hash value used to locate files in the `tables_dir` which contain tiles
        relevant to the table being loaded.

    scale : float from 0 to 1, optional
        Scale factor by which to multiply the detection probabilities in the
        table.

    subvol : None or sequence of 3 2-element sequences, optional
        Specify (min, max) values for the x-, y-, and z-dimensions to load only
        this portion of the large table. If None, load the entire table

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts. Note that if
        directionality is not to be used, the corresponding tables will not be
        loaded, resulting in ~1/4 the memory footprint.

    """
    def __init__(self, tables_dir, proto_tile_hash, subvol=None, scale=1,
                 use_directionality=True):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert isinstance(use_directionality, bool)
        assert isinstance(proto_tile_hash, basestring)
        assert scale > 0

        self.tables_dir = tables_dir
        self.use_directionality = use_directionality
        self.proto_tile_hash = proto_tile_hash
        self.scale = scale

        self.survival_prob = None
        self.avg_photon_x = None
        self.avg_photon_y = None
        self.avg_photon_z = None

        self.x_min, self.y_min, self.z_min = None, None, None
        self.x_max, self.y_max, self.z_max = None, None, None
        self.nx, self.ny, self.nz = None, None, None # pylint: disable=invalid-name
        self.nx_tiles, self.ny_tiles, self.nz_tiles = None, None, None
        self.nx_per_tile, self.ny_per_tile, self.nz_per_tile = None, None, None

        self.tables_meta = None

        proto_table_fpath = glob(join(
            expand(self.tables_dir),
            'retro_tdi_table_%s_*survival_prob.fits' % self.proto_tile_hash
        ))
        if not proto_table_fpath:
            raise ValueError('Could not find the prototypical table.')
        proto_table_fpath = proto_table_fpath[0]
        proto_meta = self.get_table_metadata(proto_table_fpath)
        if not proto_meta:
            raise ValueError('Could not figure out metadata from\n%s'
                             % proto_table_fpath)
        self.proto_meta = proto_meta

        # Some "universal" metadata can be gotten from the proto table
        self.binmap_hash = proto_meta['binmap_hash']
        self.geom_hash = proto_meta['geom_hash']
        self.dom_tables_hash = proto_meta['dom_tables_hash']
        self.time_indices = proto_meta['time_indices']
        self.times_str = proto_meta['times_str']
        self.x_tile_width = proto_meta['x_width']
        self.y_tile_width = proto_meta['y_width']
        self.z_tile_width = proto_meta['z_width']
        self.binwidth = proto_meta['binwidth']
        self.anisotropy = proto_meta['anisotropy']

        if subvol is not None:
            raise NotImplementedError()
            sv_x0, sv_x1 = subvol[0][0], subvol[0][1]
            sv_y0, sv_y1 = subvol[1][0], subvol[1][1]
            sv_z0, sv_z1 = subvol[2][0], subvol[2][1]
            assert sv_x1 - sv_x0 >= self.binwidth
            assert sv_y1 - sv_y0 >= self.binwidth
            assert sv_z1 - sv_z0 >= self.binwidth
            sv_x0_idx = (sv_x0 - proto_meta['x_min']) / self.binwidth
            sv_y0_idx = (sv_y0 - proto_meta['y_min']) / self.binwidth
            sv_z0_idx = (sv_z0 - proto_meta['z_min']) / self.binwidth
            assert abs(np.round(sv_x0_idx) - sv_x0_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_y0_idx) - sv_y0_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_z0_idx) - sv_z0_idx) * self.binwidth < 1e-6
            sv_x1_idx = (sv_x1 - proto_meta['x_min']) / self.binwidth
            sv_y1_idx = (sv_y1 - proto_meta['y_min']) / self.binwidth
            sv_z1_idx = (sv_z1 - proto_meta['z_min']) / self.binwidth
            assert abs(np.round(sv_x1_idx) - sv_x1_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_y1_idx) - sv_y1_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_z1_idx) - sv_z1_idx) * self.binwidth < 1e-6
        self.subvol = subvol

        self.tables_loaded = False
        self.load_tables()

    @staticmethod
    def get_table_metadata(fpath):
        """Interpret a Retro TDI table filename or path, returning the critical
        parameters used in generating that table.

        Parameters
        ----------
        fpath : string
            Path to the file or simply the filename

        Returns
        -------
        info : None or dict
            None is returned if the file name/path does not match the format
            for a TDI table.

        """
        fname = basename(fpath)
        match = TDI_TABLE_FNAME_RE[-1].match(fname)
        if match is None:
            return None
        meta = match.groupdict()
        for key, value in meta.items():
            if key.endswith('min') or key.endswith('max') or key == 'binwidth':
                meta[key] = float(meta[key])
            elif key == 'n_phibins':
                meta[key] = int(meta[key])
            elif key.endswith('scale'):
                meta[key] = float(meta[key])
            elif key.endswith('hash'):
                if value.lower() == 'none':
                    meta[key] = None
            elif key == 'times_str':
                if value == 'all':
                    meta['time_indices'] = slice(None)
                else:
                    meta['time_indices'] = hrlist2list(value)
            elif key == 'anisotropy':
                if value.lower() == 'none':
                    meta[key] = None
                # TODO: implement any other anisotropy spec xlation here
        meta['x_width'] = meta['x_max'] - meta['x_min']
        meta['y_width'] = meta['y_max'] - meta['y_min']
        meta['z_width'] = meta['z_max'] - meta['z_min']
        return meta

    def load_tables(self, force_reload=False):
        """Load all tables that match the `proto_tile_hash`; if multiple tables
        match, then stitch these together into one large TDI table."""
        if self.tables_loaded and not force_reload:
            return
        import pyfits

        t0 = time()

        x_ref = self.proto_meta['x_min']
        y_ref = self.proto_meta['y_min']
        z_ref = self.proto_meta['z_min']

        must_match = [
            'binmap_hash', 'geom_hash', 'dom_tables_hash', 'time_indices',
            'binwidth', 'anisotropy',
            # For simplicity, assume all tiles have equal widths. If there's a
            # compelling reason to use something more complicated, we could
            # implement it... but I see no reason to do so now
            'x_width', 'y_width', 'z_width'
        ]

        # Work with "survival_prob" table filepaths, which generalizes to all
        # table filepaths (so long as they exist)
        fpaths = glob(join(
            expand(self.tables_dir),
            'retro_tdi_table_*survival_prob.fits'
        ))

        lowermost_corner = np.array([np.inf]*3)
        uppermost_corner = np.array([-np.inf]*3)
        to_load_meta = {}
        for fpath in fpaths:
            meta = self.get_table_metadata(fpath)
            if meta is None:
                continue

            is_match = True
            for key in must_match:
                if meta[key] != self.proto_meta[key]:
                    is_match = False

            if not is_match:
                continue

            # Make sure that the corner falls on the reference grid (within
            # micrometer precision)
            x_float_idx = (meta['x_min'] - x_ref) / self.x_tile_width
            y_float_idx = (meta['y_min'] - y_ref) / self.y_tile_width
            z_float_idx = (meta['z_min'] - z_ref) / self.x_tile_width
            indices_widths = (
                [x_float_idx, y_float_idx, z_float_idx],
                [self.x_tile_width, self.y_tile_width, self.z_tile_width]
            )
            for float_idx, tile_width in zip(indices_widths):
                if abs(np.round(float_idx) - float_idx) * tile_width >= 1e-6:
                    continue

            # Extend the limits of the tiled volume to include this tile
            lower_corner = [meta['x_min'], meta['y_min'], meta['z_min']]
            upper_corner = [meta['x_max'], meta['y_max'], meta['z_max']]
            lowermost_corner = np.min([lowermost_corner, lower_corner], axis=0)
            uppermost_corner = np.max([uppermost_corner, upper_corner], axis=0)

            # Store the metadata by relative tile index
            rel_idx = tuple(int(np.round(i))
                            for i in (x_float_idx, y_float_idx, z_float_idx))
            to_load_meta[rel_idx] = meta

        x_min, y_min, z_min = lowermost_corner
        x_max, y_max, z_max = uppermost_corner

        # Figure out how many tiles we _should_ have
        nx_tiles = int(np.round((x_max - x_min) / self.x_tile_width))
        ny_tiles = int(np.round((y_max - y_min) / self.y_tile_width))
        nz_tiles = int(np.round((z_max - z_min) / self.z_tile_width))
        n_tiles = nx_tiles * ny_tiles * nz_tiles
        if len(to_load_meta) < n_tiles:
            raise ValueError('Not enough tiles found! Cannot fill the extents'
                             ' of the outermost extents of the volume defined'
                             ' by the tiles found.')
        elif len(to_load_meta) > n_tiles:
            print(self.proto_meta['tdi_hash'])
            print('x:', self.proto_meta['x_min'], self.proto_meta['x_max'],
                  self.proto_meta['x_width'])
            print('y:', self.proto_meta['y_min'], self.proto_meta['y_max'],
                  self.proto_meta['y_width'])
            print('z:', self.proto_meta['z_min'], self.proto_meta['z_max'],
                  self.proto_meta['z_width'])
            print('')
            for v in to_load_meta.values():
                print(v['tdi_hash'])
                print('x:', v['x_min'], v['x_max'], v['x_width'])
                print('y:', v['y_min'], v['y_max'], v['y_width'])
                print('z:', v['z_min'], v['z_max'], v['z_width'])
                print('')
            raise ValueError(
                'WTF? How did we get here? to_load_meta = %d, n_tiles = %d'
                % (len(to_load_meta), n_tiles)
            )

        # Figure out how many bins in each dimension fill the volume
        nx = int(np.round(nx_tiles * self.x_tile_width / self.binwidth))
        ny = int(np.round(ny_tiles * self.y_tile_width / self.binwidth))
        nz = int(np.round(nz_tiles * self.z_tile_width / self.binwidth))

        # Number of bins per dimension in the tile
        nx_per_tile = int(np.round(self.x_tile_width / self.binwidth))
        ny_per_tile = int(np.round(self.y_tile_width / self.binwidth))
        nz_per_tile = int(np.round(self.z_tile_width / self.binwidth))

        # Create empty arrays to fill
        survival_prob = np.empty((nx, ny, nz), dtype=np.float32)
        if self.use_directionality:
            avg_photon_x = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_y = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_z = np.empty((nx, ny, nz), dtype=np.float32)
        else:
            avg_photon_x, avg_photon_y, avg_photon_z = None, None, None

        anisotropy_str = generate_anisotropy_str(self.anisotropy)

        tables_meta = {} #[[[None]*nz_tiles]*ny_tiles]*nx_tiles
        for meta in to_load_meta.values():
            tile_x_idx = int(np.round(
                (meta['x_min'] - x_min) / self.x_tile_width
            ))
            tile_y_idx = int(np.round(
                (meta['y_min'] - y_min) / self.y_tile_width
            ))
            tile_z_idx = int(np.round(
                (meta['z_min'] - z_min) / self.z_tile_width
            ))

            x0_idx = int(np.round((meta['x_min'] - x_min) / self.binwidth))
            y0_idx = int(np.round((meta['y_min'] - y_min) / self.binwidth))
            z0_idx = int(np.round((meta['z_min'] - z_min) / self.binwidth))

            bin_idx_range = (slice(x0_idx, x0_idx + nx_per_tile),
                             slice(y0_idx, y0_idx + ny_per_tile),
                             slice(z0_idx, z0_idx + nz_per_tile))

            kwargs = deepcopy(meta)
            kwargs.pop('table_name')

            to_fill = [('survival_prob', survival_prob)]
            if self.use_directionality:
                to_fill.extend([
                    ('avg_photon_x', avg_photon_x),
                    ('avg_photon_y', avg_photon_y),
                    ('avg_photon_z', avg_photon_z)
                ])

            for table_name, table in to_fill:
                fpath = join(
                    self.tables_dir,
                    TDI_TABLE_FNAME_PROTO[-1].format(
                        table_name=table_name, anisotropy_str=anisotropy_str,
                        **kwargs
                    ).lower()
                )

                with pyfits.open(fpath) as fits_table:
                    data = force_little_endian(fits_table[0].data) # pylint: disable=no-member

                if self.scale != 1 and table_name == 'survival_prob':
                    data = 1 - (1 - data)**self.scale

                table[bin_idx_range] = data

            tables_meta[(tile_x_idx, tile_y_idx, tile_z_idx)] = meta

        # Since we have made it to the end successfully, it is now safe to
        # store the above-computed info to the object for later use
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nx_tiles = nx_tiles
        self.ny_tiles = ny_tiles
        self.nz_tiles = nz_tiles
        self.n_bins = self.nx * self.ny * self.nz
        self.n_tiles = self.nx_tiles * self.ny_tiles * self.nz_tiles
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max

        self.survival_prob = survival_prob
        self.avg_photon_x = avg_photon_x
        self.avg_photon_y = avg_photon_y
        self.avg_photon_z = avg_photon_z

        self.tables_meta = tables_meta
        self.tables_loaded = True

        if self.n_tiles == 1:
            tstr = 'tile'
        else:
            tstr = 'tiles'
        print('Loaded %d %s spanning'
              ' x ∈ [%.2f, %.2f) m,'
              ' y ∈ [%.2f, %.2f) m,'
              ' z ∈ [%.2f, %.2f) m;'
              ' bins are (%.3f m)³'
              % (self.n_tiles, tstr, self.x_min, self.x_max, self.y_min,
                 self.y_max, self.z_min, self.z_max, self.binwidth))
        print('Time to load: {} s'.format(np.round(time() - t0, 3)))

    def get_photon_expectation(self, sources):
        """Get the expectation for photon survival.

        Parameters
        ----------
        sources : shape (N,) numpy ndarray, dtype SRC_T

        Returns
        -------
        expected_photon_count : float
            Total expected surviving photon count

        """
        if not self.tables_loaded:
            raise Exception("Tables haven't been loaded")

        return pexp_xyz(
            sources=sources,
            x_min=self.x_min, y_min=self.y_min, z_min=self.z_min,
            nx=self.nx, ny=self.ny, nz=self.nz,
            binwidth=self.binwidth,
            survival_prob=self.survival_prob,
            avg_photon_x=self.avg_photon_x,
            avg_photon_y=self.avg_photon_y,
            avg_photon_z=self.avg_photon_z,
            use_directionality=self.use_directionality
        )

    #def plot_slices(self, x_slice=slice(None), y_slice=slice(None),
    #                z_slice=slice(None)):
    #    # Formulate a slice through the table to look at
    #    slx = slice(dom_x_idx - ncells,
    #                dom_x_idx + ncells,
    #                1)
    #    sly = slice(dom_y_idx - ncells,
    #                dom_y_idx + ncells,
    #                1)
    #    slz = dom_z_idx
    #    sl = (x_slice, y_slice, slz)

    #    # Slice the x and y directions
    #    pxsl = binned_px[sl]
    #    pysl = binned_py[sl]

    #    xmid = (xlims[0] + x_bw/2.0 + x_bw * np.arange(nx))[x_slice]
    #    ymid = (ylims[0] + y_bw/2.0 + y_bw * np.arange(ny))[y_slice]
    #    zmid = zlims[0] + z_bw/2.0 + z_bw * dom_z_idx

    #    x_inner_lim = (xmid.min() - x_bw/2.0, xmid.max() + x_bw/2.0)
    #    y_inner_lim = (ymid.min() - y_bw/2.0, ymid.max() + y_bw/2.0)
    #    X, Y = np.meshgrid(xmid, ymid, indexing='ij')

    #    fig = plt.figure(1, figsize=(10, 10), dpi=72)
    #    fig.clf()
    #    ax = fig.add_subplot(111)

    #    ax.plot(
    #        dom_x, dom_y,
    #        'ro', ms=8, lw=0.5,
    #        label='Actual DOM location'
    #    )
    #    ax.plot(
    #        xlims[0] + x_os_bw*dom_x_os_idx,
    #        ylims[0] + y_os_bw*dom_y_os_idx,
    #        'go', ms=8, lw=0.5,
    #        label='DOM location used for binning'
    #    )
    #    ax.quiver(
    #        X, Y, pxsl, pysl,
    #        label='Binned average photon direction'
    #    )

    #    ax.axis('image')
    #    ax.set_xlabel('x (m)')
    #    ax.set_ylabel('y (m)')

    #    ax.set_xticks(np.arange(xlims[0], xlims[1]+x_bw, x_bw), minor=False)
    #    ax.grid(which='major', b=True)
    #    if x_oversample > 1:
    #        ax.set_xticks(
    #            np.arange(x_inner_lim[0]+x_os_bw, x_inner_lim[1], x_os_bw),
    #            minor=True
    #        )
    #        ax.grid(which='minor', b=True, ls=':', alpha=0.6)

    #    if y_oversample > 1:
    #        ax.set_yticks(
    #            np.arange(y_inner_lim[0]+y_os_bw, y_inner_lim[1], y_os_bw),
    #            minor=True
    #        )
    #        ax.grid(which='minor', b=True, ls=':', alpha=0.6)

    #    ax.set_xlim(x_inner_lim)
    #    ax.set_ylim(y_inner_lim)
    #    ax.legend(loc='upper left', fancybox=True, framealpha=0.9)
    #    ax.set_title('Detail of table, XY-slice through center of DOM')
    #    fig.savefig('xyslice_detail.png', dpi=300)
    #    fig.savefig('xyslice_detail.pdf')

    #def plot_projections(self):
    #    pass
