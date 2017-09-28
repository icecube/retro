# pylint: disable=wrong-import-position
# -*- coding: utf-8 -*-

"""
Classes for reading and getting info from Retro tables.
"""
# TODO: should the QE numbers be simple multipliers?

from __future__ import absolute_import, division, print_function

from copy import deepcopy
from glob import glob
import math
import os
from os.path import abspath, basename, dirname, isdir, join
import time

import numba
import numpy as np
import pyfits

from pisa.utils.timing import timediffstamp

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (IC_QUANT_EFF, DC_QUANT_EFF, DC_TABLE_FNAME_PROTO,
                   IC_TABLE_FNAME_PROTO, SPEED_OF_LIGHT_M_PER_NS, POL_TABLE_DT,
                   POL_TABLE_RMAX, POL_TABLE_RPWR, POL_TABLE_DRPWR,
                   POL_TABLE_DCOSTHETA, POL_TABLE_NTBINS, POL_TABLE_NRBINS,
                   POL_TABLE_NTHETABINS)
from retro import RetroPhotonInfo
from retro import expand, extract_photon_info
from retro.test_tindep import (TDI_TABLE_FNAME_PROTO, TDI_TABLE_FNAME_RE,
                               get_anisotropy_str)

from pisa.utils.format import hrlist2list


__all__ = ['pexp_t_r_theta', 'pexp_xyz', 'DOMTimePolarTables', 'TDICartTables']


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def pexp_t_r_theta(pinfo_gen, hit_time, dom_coord, survival_prob,
                   avg_photon_theta, avg_photon_length, use_directionality):
    """Compute total expected photons in a DOM based on the (t,r,theta)-binned
    Retro DOM tables applied to a the generated photon info `pinfo_gen`.

    Parameters
    ----------
    pinfo_gen : shape (N, 8) numpy ndarray, dtype float64
    hit_time : float
    dom_coord : shape (3,) numpy ndarray, dtype float64
    survival_prob
    avg_photon_theta
    avg_photon_length
    use_directionality : bool

    Returns
    -------
    expected_photon_count : float

    """
    expected_photon_count = 0.0
    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :]

        # A photon that starts immediately in the past (before the DOM was hit)
        # will show up in the Retro DOM tables in the _last_ bin.
        # Therefore, invert the sign of the t coordinate and index sequentially
        # via e.g. -1, -2, ....
        dt = hit_time - t
        dx = x - dom_coord[0]
        dy = y - dom_coord[1]
        dz = z - dom_coord[2]

        rho2 = dx**2 + dy**2
        r = np.sqrt(rho2 + dz**2)

        #spacetime_sep = SPEED_OF_LIGHT_M_PER_NS*dt - r
        #if spacetime_sep < 0 or spacetime_sep >= POL_TABLE_RMAX:
        #    print('spacetime_sep:', spacetime_sep)
        #    print('MAX_POL_TABLE_SPACETIME_SEP:', POL_TABLE_RMAX)
        #    continue

        tbin_idx = int(np.floor(dt / POL_TABLE_DT))
        #if tbin_idx < 0 or tbin_idx >= -POL_TABLE_DT:
        if tbin_idx < -POL_TABLE_NTBINS or tbin_idx >= 0:
            #print('t')
            continue
        rbin_idx = int(np.floor(r**(1/POL_TABLE_RPWR) / POL_TABLE_DRPWR))
        if rbin_idx < 0 or rbin_idx >= POL_TABLE_NRBINS:
            #print('r')
            continue
        thetabin_idx = int(dz / (r * POL_TABLE_DCOSTHETA))
        if thetabin_idx < 0 or thetabin_idx >= POL_TABLE_NTHETABINS:
            #print('theta')
            continue
        #print(tbin_idx, rbin_idx, thetabin_idx)
        #raise Exception()
        surviving_count = p_count * survival_prob[tbin_idx, rbin_idx, thetabin_idx]

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        # TODO: Incorporate photon direction info
        if use_directionality:
            pass

        expected_photon_count += surviving_count

    return expected_photon_count


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def pexp_xyz(pinfo_gen, x_min, y_min, z_min, nx, ny, nz, binwidth, survival_prob, avg_photon_x, avg_photon_y, avg_photon_z, use_directionality):
    expected_photon_count = 0.0
    for pgen_idx in range(pinfo_gen.shape[0]):
        _, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :]
        x_idx = int(np.round((x - x_min) / binwidth))
        if x_idx < 0 or x_idx >= nx:
            continue
        y_idx = int(np.round((y - y_min) / binwidth))
        if y_idx < 0 or y_idx >= ny:
            continue
        z_idx = int(np.round((z - z_min) / binwidth))
        if z_idx < 0 or z_idx >= nz:
            continue
        sp = survival_prob[x_idx, y_idx, z_idx]
        surviving_count = p_count * sp

        # TODO: Incorporate photon direction info
        if use_directionality:
            pass

        expected_photon_count += surviving_count

    return expected_photon_count


class DOMTimePolarTables(object):
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

    """
    def __init__(self, tables_dir, hash_val, geom, use_directionality):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        #assert 0 < scale < np.inf

        self.tables_dir = tables_dir
        self.hash_val = hash_val
        self.geom = geom
        self.use_directionality = use_directionality
        #self.scale = scale
        self.tables = {'ic': {}, 'dc': {}}

    def load_table(self, string, depth_idx, force_reload=False):
        """Load a table from disk into memory.

        Parameters
        ----------
        string : int
            Indexed from 1

        depth_idx : int
        force_reload : bool

        """
        string_idx = string - 1
        if string_idx < 79:
            subdet = 'ic'
            fname_proto = IC_TABLE_FNAME_PROTO
            quantum_efficiency = IC_QUANT_EFF
        else:
            subdet = 'dc'
            fname_proto = DC_TABLE_FNAME_PROTO
            quantum_efficiency = DC_QUANT_EFF

        if not force_reload and depth_idx in self.tables[subdet]:
            return

        fpath = join(self.tables_dir, fname_proto.format(depth_idx=depth_idx))

        photon_info, _ = extract_photon_info(
            fpath=fpath,
            depth_idx=depth_idx,
            scale=quantum_efficiency #* scale
        )

        length = photon_info.length[depth_idx]
        deltaphi = photon_info.deltaphi[depth_idx]
        self.tables[subdet][depth_idx] = RetroPhotonInfo(
            survival_prob=photon_info.survival_prob[depth_idx],
            theta=photon_info.theta[depth_idx],
            deltaphi=photon_info.deltaphi[depth_idx],
            length=(photon_info.length[depth_idx]
                    * np.cos(photon_info.deltaphi[depth_idx]))
        )

    def load_tables(self):
        """Load all tables"""
        for string in range(1, 86+1):
            for depth_idx in range(60):
                self.load_table(string=string, depth_idx=depth_idx,
                                force_reload=False)

    def get_photon_expectation(self, pinfo_gen, hit_time, string, depth_idx,
                               use_directionality=None):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy ndarray, dtype float64

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

        string_idx = string - 1

        dom_coord = self.geom[string_idx, depth_idx]
        if string_idx < 79:
            subdet = 'ic'
        else:
            subdet = 'dc'
        table = self.tables[subdet][depth_idx]
        survival_prob = table.survival_prob
        avg_photon_theta = table.theta
        avg_photon_length = table.length
        return pexp_t_r_theta(pinfo_gen=pinfo_gen,
                              hit_time=hit_time,
                              dom_coord=dom_coord,
                              survival_prob=survival_prob,
                              avg_photon_theta=avg_photon_theta,
                              avg_photon_length=avg_photon_length,
                              use_directionality=use_directionality)


class TDICartTables(object):
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
    def __init__(self, tables_dir, use_directionality, proto_tile_hash,
                 scale=1, subvol=None):
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
        self.nx, self.ny, self.nz = None, None, None
        self.nx_tiles, self.ny_tiles, self.nz_tiles = None, None, None
        self.nx_per_tile, self.ny_per_tile, self.nz_per_tile = None, None, None

        self.tables_meta = None

        proto_table_fpath = glob(join(
            self.tables_dir,
            'retro_tdi_table_%s_*survival_prob.fits' % self.proto_tile_hash
        ))
        if not proto_table_fpath:
            raise ValueError('Could not find the prototypical table.')
        proto_table_fpath = proto_table_fpath[0]
        proto_meta = self.get_table_metadata(proto_table_fpath)
        if not proto_meta:
            raise ValueError('Could not figure out metadata from\n' + self.proto_table_fpath)
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
            raise NotImplementedError
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
        match = TDI_TABLE_FNAME_RE.match(fname)
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
        if self.tables_loaded and not force_reload:
            return

        t0 = time.time()

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
            self.tables_dir, 'retro_tdi_table_*survival_prob.fits'
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
            for float_idx, tile_width in zip([x_float_idx, y_float_idx, z_float_idx],
                                             [self.x_tile_width, self.y_tile_width, self.z_tile_width]):
                if abs(np.round(float_idx) - float_idx) * tile_width >= 1e-6:
                    continue

            # Extend the limits of the tiled volume to include this tile
            lower_corner = [meta['x_min'], meta['y_min'], meta['z_min']]
            upper_corner = [meta['x_max'], meta['y_max'], meta['z_max']]
            lowermost_corner = np.min([lowermost_corner, lower_corner], axis=0)
            uppermost_corner = np.max([uppermost_corner, upper_corner], axis=0)

            # Store the metadata by relative tile index
            rel_idx = tuple(int(np.round(i)) for i in [x_float_idx, y_float_idx, z_float_idx])
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
            print('x:', self.proto_meta['x_min'], self.proto_meta['x_max'], self.proto_meta['x_width'])
            print('y:', self.proto_meta['y_min'], self.proto_meta['y_max'], self.proto_meta['y_width'])
            print('z:', self.proto_meta['z_min'], self.proto_meta['z_max'], self.proto_meta['z_width'])
            print('')
            for v in to_load_meta.values():
                print(v['tdi_hash'])
                print('x:', v['x_min'], v['x_max'], v['x_width'])
                print('y:', v['y_min'], v['y_max'], v['y_width'])
                print('z:', v['z_min'], v['z_max'], v['z_width'])
                print('')
            raise ValueError('WTF? How did we get here? to_load_meta = %d, n_tiles = %d' % (len(to_load_meta), n_tiles))

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

        anisotropy_str = get_anisotropy_str(self.anisotropy)

        tables_meta = {} #[[[None]*nz_tiles]*ny_tiles]*nx_tiles
        for meta in to_load_meta.values():
            tile_x_idx = int(np.round((meta['x_min'] - x_min) / self.x_tile_width))
            tile_y_idx = int(np.round((meta['y_min'] - y_min) / self.y_tile_width))
            tile_z_idx = int(np.round((meta['z_min'] - z_min) / self.z_tile_width))

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
                    TDI_TABLE_FNAME_PROTO.format(
                        table_name=table_name, anisotropy_str=anisotropy_str,
                        **kwargs
                    ).lower()
                )

                with pyfits.open(fpath) as fits_table:
                    data = fits_table[0].data
                if data.dtype.byteorder == '>':
                    data = data.byteswap().newbyteorder()

                if self.scale != 1 and table_name == 'survival_prob':
                    data = 1 - (1 - data)**self.scale

                table[bin_idx_range] = data

            tables_meta[(tile_x_idx, tile_y_idx, tile_z_idx)] = meta

        # Since we have made it to the end successfully, it is now safe to
        # store the above-computed info to the object for later use
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nx_tiles, self.ny_tiles, self.nz_tiles = nx_tiles, ny_tiles, nz_tiles
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
        print('Time to load: %s' % timediffstamp(time.time() - t0))

    def get_photon_expectation(self, pinfo_gen):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy ndarray, dtype float64

        Returns
        -------
        expected_photon_count : float
            Total expected surviving photon count

        """
        if not self.tables_loaded:
            raise Exception("Tables haven't been loaded")

        kwargs = dict(
            pinfo_gen=pinfo_gen,
            x_min=self.x_min, y_min=self.y_min, z_min=self.z_min,
            nx=self.nx, ny=self.ny, nz=self.nz,
            binwidth=self.binwidth,
            survival_prob=self.survival_prob,
            avg_photon_x=self.avg_photon_x,
            avg_photon_y=self.avg_photon_y,
            avg_photon_z=self.avg_photon_z,
            use_directionality=self.use_directionality
        )
        photon_expectation = pexp_xyz(**kwargs)

        return photon_expectation

    def plot_slices(self, x_slice=slice(None), y_slice=slice(None),
                    z_slice=slice(None)):
        # Formulate a slice through the table to look at
        slx = slice(dom_x_idx - ncells,
                    dom_x_idx + ncells,
                    1)
        sly = slice(dom_y_idx - ncells,
                    dom_y_idx + ncells,
                    1)
        slz = dom_z_idx
        sl = (x_slice, y_slice, slz)

        # Slice the x and y directions
        pxsl = binned_px[sl]
        pysl = binned_py[sl]

        xmid = (xlims[0] + x_bw/2.0 + x_bw * np.arange(nx))[x_slice]
        ymid = (ylims[0] + y_bw/2.0 + y_bw * np.arange(ny))[y_slice]
        zmid = zlims[0] + z_bw/2.0 + z_bw * dom_z_idx

        x_inner_lim = (xmid.min() - x_bw/2.0, xmid.max() + x_bw/2.0)
        y_inner_lim = (ymid.min() - y_bw/2.0, ymid.max() + y_bw/2.0)
        X, Y = np.meshgrid(xmid, ymid, indexing='ij')

        fig = plt.figure(1, figsize=(10, 10), dpi=72)
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(
            dom_x, dom_y,
            'ro', ms=8, lw=0.5,
            label='Actual DOM location'
        )
        ax.plot(
            xlims[0] + x_os_bw*dom_x_os_idx,
            ylims[0] + y_os_bw*dom_y_os_idx,
            'go', ms=8, lw=0.5,
            label='DOM location used for binning'
        )
        ax.quiver(
            X, Y, pxsl, pysl,
            label='Binned average photon direction'
        )

        ax.axis('image')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        ax.set_xticks(np.arange(xlims[0], xlims[1]+x_bw, x_bw), minor=False)
        ax.grid(which='major', b=True)
        if x_oversample > 1:
            ax.set_xticks(
                np.arange(x_inner_lim[0]+x_os_bw, x_inner_lim[1], x_os_bw),
                minor=True
            )
            ax.grid(which='minor', b=True, ls=':', alpha=0.6)

        if y_oversample > 1:
            ax.set_yticks(
                np.arange(y_inner_lim[0]+y_os_bw, y_inner_lim[1], y_os_bw),
                minor=True
            )
            ax.grid(which='minor', b=True, ls=':', alpha=0.6)

        ax.set_xlim(x_inner_lim)
        ax.set_ylim(y_inner_lim)
        ax.legend(loc='upper left', fancybox=True, framealpha=0.9)
        ax.set_title('Detail of table, XY-slice through center of DOM')
        fig.savefig('xyslice_detail.png', dpi=300)
        fig.savefig('xyslice_detail.pdf')

    def plot_projections(self):
        pass
