# pylint: disable=wrong-import-position

"""
Classes for reading and getting info from Retro tables.
"""


from __future__ import absolute_import, division, print_function

import math
import os
from os.path import abspath, dirname, isdir, join

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


__all__ = []


@numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def pexp_t_r_theta(pinfo_array, dom_coord, table, use_directionality=False):
    """Compute total expected photons in a DOM.

    Parameters
    ----------
    pinfo_array : shape (N, 8) numpy ndarray, dtype float32
    dom_coord : shape (3,) numpy ndarray, dtype float64
    table :
    use_directionality : bool

    Returns
    -------
    expected_photon_count

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

    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float64

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts.

    scale : float

    """
    def __init__(self, tables_dir, geom, use_directionality, scale=1):
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

        fpath = join(self.tables_dir, fname_proto.format(dom=depth_idx))

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

    def load_all_tables(self):
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
        and table to modify the expected surviving photon counts.

    x_bw, y_bw, z_bw
        Tables x-, y-, and z-bin widths

    x_oversample, y_oversample, z_oversample
        Tables x, y, and z oversampling

    antialias_factor
        Tables antialiasing factor

    scale : float

    """
    def __init__(self, tables_dir, geom, use_directionality, x_bw, y_bw, z_bw,
                 x_oversample, y_oversample, z_oversample, antialias_factor,
                 scale=1):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        assert 0 < scale < np.inf

        self.tables_dir = tables_dir
        self.geom = geom
        self.geom_hash = hash_obj(geom)
        self.use_directionality = use_directionality
        self.scale = scale

        self.tables = []
        self.table_extents = []
        self._table_file_paths = []
        self._table_file_extents = None

    def _find_tables(self):
        all_files = os.listdir(self.tables_dir)

    def load_table(self, limits, force_reload=False):
        """Load a table from disk into memory.

        Parameters
        ----------
        limits : Cart3DCoord of 2-tuples
            (min, max) limits for region requested in x, y, and z directions.

        force_reload : bool

        """
        if not force_reload and depth_idx in self.tables[subdet]:
            return

        fpath = join(self.tables_dir, fname_proto.format(dom=depth_idx))

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
