# pylint: disable=wrong-import-position

"""
Classes for reading and getting info from Retro tables.
"""


from __future__ import absolute_import, division, print_function

from glob import glob
import math
import os
from os.path import abspath, dirname, isdir, join
import re

import numba
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (DC_TABLE_FNAME_PROTO, IC_TABLE_FNAME_PROTO,
                   SPEED_OF_LIGHT_M_PER_NS, MAX_POL_TABLE_SPACETIME_SEP,
                   POL_TABLE_DT, POL_TABLE_RPWR, POL_TABLE_DRPWR,
                   POL_TABLE_DCOSTHETA)
from retro import RetroPhotonInfo
from retro import expand, extract_photon_info
from retro.test_tindep import TDI_TABLE_FNAME_PROTO, TDI_TABLE_FNAME_RE


__all__ = ['pexp_t_r_theta', 'RetroDOMTimePolarTables', 'RetroTDICartTables']


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def pexp_t_r_theta(pinfo_array, dom_coord, table, use_directionality=False):
    """Compute total expected photons in a DOM based on the (t,r,theta)-binned
    Retro DOM tables applied to a the generated photon info `pinfo_gen`.

    Parameters
    ----------
    pinfo_array : shape (N, 8) numpy ndarray, dtype float32
    dom_coord : shape (3,) numpy ndarray, dtype float64
    table :
    use_directionality : bool

    Returns
    -------
    expected_photon_count : float

    """
    expected_photon_count = 0.0
    for (t, x, y, z, p_count, p_x, p_y, p_z) in pinfo_array:
        # An photon that starts in the past (before the DOM was hit) will show
        # up in the Retro DOM tables as a positive time relative to the DOM.
        # Therefore, invert the sign of the t coordinate.
        dt = -t
        dx = x - dom_coord[1]
        dy = y - dom_coord[2]
        dz = z - dom_coord[3]

        rho2 = dx**2 + dy**2
        r = math.sqrt(rho2 + dz**2)

        spacetime_sep = SPEED_OF_LIGHT_M_PER_NS*dt - r
        if spacetime_sep < 0 or spacetime_sep >= MAX_POL_TABLE_SPACETIME_SEP:
            continue

        tbin_idx = int(math.floor(dt / POL_TABLE_DT))
        rbin_idx = int(math.floor(r**(1/POL_TABLE_RPWR) / POL_TABLE_DRPWR))
        thetabin_idx = int(dz / (r * POL_TABLE_DCOSTHETA))
        photon_info = table[tbin_idx, rbin_idx, thetabin_idx]
        surviving_count = p_count * photon_info.survival_prob

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        # TODO: Incorporate photon direction info
        if use_directionality:
            pass

        expected_photon_count += surviving_count

    return expected_photon_count


class RetroDOMTimePolarTables(object):
    """Load and use information from individual-dom (time, r, theta)-binned
    Retro tables.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float64

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts.

    scale : float

    """
    def __init__(self, tables_dir, hash_val, geom, use_directionality,
                 scale=1):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        assert 0 < scale < np.inf

        self.tables_dir = tables_dir
        self.geom = geom
        self.use_directionality = use_directionality
        self.scale = scale
        self.tables = {'ic': {}, 'dc': {}}

    def load_table(self, string, depth_idx, force_reload=False):
        """Load a table from disk into memory.

        Parameters
        ----------
        string : int
        depth_idx : int
        force_reload : bool

        """
        if string < 79:
            subdet = 'ic'
            fname_proto = IC_TABLE_FNAME_PROTO
        else:
            subdet = 'dc'
            fname_proto = DC_TABLE_FNAME_PROTO

        if not force_reload and depth_idx in self.tables[subdet]:
            return

        fpath = join(self.tables_dir, fname_proto.format(depth_idx=depth_idx))

        photon_info, _ = extract_photon_info(
            fpath=fpath,
            depth_idx=depth_idx,
            scale=self.scale
        )

        self.tables[subdet][depth_idx] = RetroPhotonInfo(
            survival_prob=photon_info.survival_prob[depth_idx],
            theta=photon_info.theta[depth_idx],
            deltaphi=photon_info.deltaphi[depth_idx],
            length=(photon_info.length[depth_idx]
                    * np.cos(photon_info.deltaphi[depth_idx]))
        )

    def load_tables(self):
        """Load all tables"""
        for string in range(86):
            for depth_idx in range(60):
                self.load_table(string=string, depth_idx=depth_idx,
                                force_reload=False)

    def get_photon_expectation(self, pinfo_array, string, depth_idx,
                               use_directionality=None):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_array : shape (N, 8) numpy ndarray, dtype float32

        use_directionality : None or bool
            Whether to use photon directionality informatino in hypo / table to
            modify expected surviving photon counts. If specified, overrides
            argument passed at class instantiation time. Otherwise, that value
            for `use_directionality` is used.

        Returns
        -------
        expected_photon_count : float
            Total expected surviving photon count

        """
        if use_directionality is None:
            use_directionality = self.use_directionality

        dom_coord = self.geom[string, depth_idx]
        if string < 79:
            subdet = 'ic'
        else:
            subdet = 'dc'
        table = self.tables[subdet][depth_idx]
        return pexp_t_r_theta(pinfo_array=pinfo_array,
                              dom_coord=dom_coord,
                              table=table,
                              use_directionality=use_directionality)


class RetroTDICartTables(object):
    """Load and use information from time- and DOM-independent Cartesian
    (x, y, z)-binned Retro tables.

    The parameters used to generate the table are passed at instantiation of
    this class to determine which table(s) to load when a table is requested.

    Parameters
    ----------
    tables_dir : string

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts. Note that if
        directionality is not to be used, the corresponding tables will not be
        loaded, resulting in ~1/4 the memory footprint.

    x_bw, y_bw, z_bw
        Tables x-, y-, and z-bin widths

    x_oversample, y_oversample, z_oversample
        Tables x, y, and z oversampling

    antialias_factor
        Tables antialiasing factor

    scale : float

    """
    def __init__(self, tables_dir, use_directionality, proto_tile_hash):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert isinstance(use_directionality, bool)
        assert isinstance(proto_tile_hash, basestring)

        self.tables_dir = tables_dir
        self.use_directionality = use_directionality
        self.proto_tile_hash = proto_tile_hash

        self.survival_prob = None
        self.avg_photon_x = None
        self.avg_photon_y = None
        self.avg_photon_z = None

        self.x_min, self.y_min, self.z_min = None, None, None
        self.x_max, self.y_max, self.z_max = None, None, None
        self.nx, self.ny, self.nz = None, None, None
        self.nx_tiles, self.ny_tiles, self.nz_tiles = None, None, None
        self.nx_per_tile, self.ny_per_tile, self.nz_per_tile = None, None, None

        self.tables_meta = None

        self.proto_table_fpath = glob(join(
            self.tables_dir,
            'retro_tdi_table_%s_*survival_prob.fits' % self.proto_tile_hash
        ))
        if not self.proto_table_fpath:
            raise ValueError('Could not find the prototypical table.')
        self.proto_meta = self.get_table_metadata(proto_tables[0])

        # Some "universal" metadata can be gotten from the proto table
        self.binmap_hash = self.proto_meta['binmap_hash']
        self.geom_hash = self.proto_meta['geom_hash']
        self.dom_table_hash = self.proto_meta['dom_table_hash']
        self.times = self.proto_meta['times']
        self.times_str = self.proto_meta['times_str']
        self.x_tile_width = self.proto_meta['x_width']
        self.y_tile_width = self.proto_meta['y_width']
        self.z_tile_width = self.proto_meta['z_width']
        self.binwidth = self.proto_meta['binwidth']
        self.anisotropy = self.proto_meta['anisotropy']

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
        fname = basneame(fpath)
        match = TDI_TABLE_FNAME_RE.match(fname)
        if match is None:
            return None
        meta = match.groupdict()
        for key, value in meta.items():
            if key.endswith('min') or key.endswith('max') or key == 'binwidth':
                meta[key] = float(meta[key])
            elif key == 'n_phibins':
                meta[key] = int(meta[key])
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
        if self.tables_loaded and not force_reload:
            return

        x_ref = self.proto_meta['x_min']
        y_ref = self.proto_meta['y_min']
        z_ref = self.proto_meta['z_min']

        must_match = [
            'binmap_hash', 'geom_hash', 'dom_tables_hash', 'times', 'binwidth',
            'anisotropy',
            # For simplicity, assume all tiles have equal widths. If there's a
            # compelling reason to use something more complicated, we could
            # implement it... but I see no reason to do so now
            'x_width', 'y_width', 'z_width'
        ]

        # Work with "survival_prob" table filepaths, which generalizes to all
        # table filepaths (so long as they exist)
        fpaths = glob(join(
            self.tables_dir, 'retro_tdi_table_*survival_prob.fits'
        ))

        lowermost_corner = np.array([np.inf]*3)
        uppermost_corner = np.array([-np.inf]*3)
        to_load_meta = {}
        for fpath in fpaths:
            meta = self.get_table_metadata(fpath)
            if meta is None:
                continue

            for key in must_match:
                if meta[key] != self.proto_meta[key]:
                    continue

            # Make sure that the corner falls on the reference grid (within
            # micrometer precision)
            x_float_idx = (meta['x_min'] - x_ref) / x_tile_width
            y_float_idx = (meta['y_min'] - y_ref) / y_tile_width
            z_float_idx = (meta['z_min'] - z_ref) / x_tile_width
            for float_idx, tile_width in zip([x_float_idx, y_float_idx, z_float_idx],
                                             [x_tile_width, y_tile_width, z_tile_width]):
                if abs(round(float_idx) - float_idx) * tile_width >= 1e-6:
                    continue

            # Extend the limits of the tiled volume to include this tile
            lower_corner = [meta['x_min'], meta['y_min'], meta['z_min']]
            upper_corner = [meta['x_max'], meta['y_max'], meta['z_max']]
            lowermost_corner = np.min([lowermost_corner, lower_corner], axis=0)
            uppermost_corner = np.max([uppermost_corner, upper_corner], axis=0)

            # Store the metadata by relative tile index
            rel_idx = tuple(int(round(i)) for i in [x_float_idx, y_float_idx, z_float_idx])
            to_load_meta[rel_idx] = meta

        x_min, y_min, z_min = lowermost_corner
        x_max, y_max, z_max = uppermost_corner

        # Figure out how many tiles we _should_ have
        nx_tiles = int(round((x_max - x_min) / self.x_tile_width))
        ny_tiles = int(round((y_max - y_min) / self.y_tile_width))
        nz_tiles = int(round((z_max - z_min) / self.z_tile_width))
        n_tiles = nx_tiles * ny_tiles * nz_tiles
        if len(to_load_meta) < n_tiles:
            raise ValueError('Not enough tiles found! Cannot fill the extents'
                             ' of the outermost extents of the volume defined'
                             ' by the tiles found.')
        elif len(to_load_meta) > n_tiles:
            raise ValueError('WTF? How did we get here?')

        # Figure out how many bins in each dimension fill the volume
        nx = int(round(nx_tiles * self.x_tile_width / self.binwidth))
        ny = int(round(ny_tiles * self.y_tile_width / self.binwidth))
        nz = int(round(nz_tiles * self.z_tile_width / self.binwidth))

        # Number of bins per dimension in the tile
        nx_per_tile = int(round(self.x_tile_width / self.binwidth))
        ny_per_tile = int(round(self.y_tile_width / self.binwidth))
        nz_per_tile = int(round(self.z_tile_width / self.binwidth))

        # Create empty arrays to fill
        survival_prob = np.empty((nx, ny, nz), dtype=np.float32)
        if self.use_directionality:
            avg_photon_x = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_y = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_z = np.empty((nx, ny, nz), dtype=np.float32)
        else:
            avg_photon_x, avg_photon_y, avg_photon_z = None, None, None

        tables_meta = {} #[[[None]*nz_tiles]*ny_tiles]*nx_tiles
        for meta in to_load_meta:
            tile_x_idx = int(round((meta['x_min'] - x_min) / self.x_tile_width))
            tile_y_idx = int(round((meta['y_min'] - y_min) / self.y_tile_width))
            tile_z_idx = int(round((meta['z_min'] - z_min) / self.z_tile_width))

            x0_idx = int(round((meta['x_min'] - x_min) / self.binwidth))
            y0_idx = int(round((meta['y_min'] - y_min) / self.binwidth))
            z0_idx = int(round((meta['z_min'] - z_min) / self.binwidth))

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
                    TDI_TABLE_FNAME_PROTO.format(
                        table_name=table_name, **kwargs
                    )
                )
                with pyfits.open(fpath) as fits_table:
                    table[idx] = fits_table[0].data

            tables_meta[(tile_x_idx, tile_y_idx, tile_z_idx)] = meta

        # Since we have made it to the end successfully, it is now safe to
        # store the above-computed info to the object for later use
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nx_tiles, self.ny_tiles, zelf.nz_tiles = nx_tiles, ny_tiles, nz_tiles
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max

        self.survival_prob = survival_prob
        self.avg_photon_x = avg_photon_x
        self.avg_photon_y = avg_photon_y
        self.avg_photon_z = avg_photon_z

        self.tables_meta = tables_meta
        self.tables_loaded = True

    def get_photon_expectation(self, pinfo_array):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_array : shape (N, 8) numpy ndarray, dtype float32

        Returns
        -------
        expected_photon_count : float
            Total expected surviving photon count

        """
        if not self.tables_loaded:
            raise Exception("Tables haven't been loaded")

        kwargs = dict(
            pinfo_array=pinfo_array,
            x_min=self.x_min, y_min=self.y_min, z_min=self.z_min,
            survival_prob=self.survival_prob,
            avg_photon_x=self.avg_photon_x,
            avg_photon_y=self.avg_photon_y,
            avg_photon_z=self.avg_photon_z,
            use_directionality=self.use_directionality
        )
        photon_expectation = pexp_xyz(**kwargs)

        return photon_expectation
