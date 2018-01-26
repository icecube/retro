# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Classes for reading and getting info from Retro tables.
"""


from __future__ import absolute_import, division, print_function


__all__ = ['load_clsim_table', 'load_t_r_theta_table', 'pexp_t_r_theta',
           'pexp_xyz', 'CLSimTable', 'DOMTimePolarTables', 'TDICartTable']

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


from collections import OrderedDict
from copy import deepcopy
from glob import glob
from os import makedirs, remove
from os.path import abspath, basename, dirname, isdir, isfile, join, splitext
from StringIO import StringIO
from subprocess import Popen, PIPE
import sys
import time

import numpy as np
import pyfits

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import (DFLT_NUMBA_JIT_KWARGS, IC_DOM_QUANT_EFF, DC_DOM_QUANT_EFF,
                   CLSIM_TABLE_FNAME_V2_PROTO,
                   RETRO_DOM_TABLE_FNAME_PROTO,
                   TDI_TABLE_FNAME_PROTO, TDI_TABLE_FNAME_RE,
                   SPEED_OF_LIGHT_M_PER_NS, POL_TABLE_NTHETABINS,
                   POL_TABLE_DCOSTHETA, POL_TABLE_NRBINS, POL_TABLE_DRPWR,
                   ZSTD_EXTENSIONS)
from retro import RetroPhotonInfo, TimeSphCoord
from retro import (expand, force_little_endian, generate_anisotropy_str,
                   interpret_clsim_table_fname, linear_bin_centers, numba_jit)
from retro.generate_t_r_theta_table import generate_t_r_theta_table


def load_clsim_table(fpath):
    """Load a CLSim table from disk (optionally compressed with zstd).

    Parameters
    ----------
    fpath : string
        Path to file to be loaded. If the file has extension 'zst', 'zstd', or
        'zstandard', the file will be decompressed using the `python-zstandard`
        Python library before passing to `pyfits` for interpreting.

    Returns
    -------
    table : OrderedDict
        Items include
        - 'table' : np.ndarray
        - 'table_shape' : tuple of int
        - 'n_photons' :
        - 'phase_refractive_index' :

        If the table is 5D, items also include
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :

    """
    fpath = abspath(expand(fpath))
    assert isfile(fpath)
    _, ext = splitext(fpath)
    ext = ext.lstrip('.').lower()

    table = OrderedDict()
    if ext in ZSTD_EXTENSIONS:
        # -c sends decompressed output to stdout
        proc = Popen(['zstd', '-d', '-c', fpath], stdout=PIPE)
        # Read from stdout
        (proc_stdout, _) = proc.communicate()
        # Give the string from stdout a file-like interface
        fobj = StringIO(proc_stdout)
    elif ext in ('fits',):
        fobj = open(fpath, 'rb')
    else:
        raise ValueError('Unhandled extension "{}"'.format(ext))

    try:
        pf_table = pyfits.open(fobj)

        table_shape_incl_overflow = pf_table[0].data.shape # pylint: disable=no-member
        n_photons = force_little_endian(pf_table[0].header['_i3_n_photons']) # pylint: disable=no-member
        phase_refractive_index = force_little_endian(pf_table[0].header['_i3_n_phase']) # pylint: disable=no-member
        raw_table = force_little_endian(pf_table[0].data) # pylint: disable=no-member

        table_shape = tuple(dim - 2 for dim in table_shape_incl_overflow)
        n_dims = len(table_shape)
        table['table_shape'] = table_shape

        # Cut off first and last bin in each dimension (underflow and
        # overflow bins)
        slice_wo_overflow = (slice(1, -1),) * n_dims
        table_vals = raw_table[slice_wo_overflow]

        underflow, overflow = [], []
        for n in range(n_dims):
            sl = tuple([slice(1, -1)]*n + [0] + [slice(1, -1)]*(n_dims - 1 - n))
            underflow.append(raw_table[sl].sum())

            sl = tuple([slice(1, -1)]*n + [-1] + [slice(1, -1)]*(n_dims - 1 - n))
            overflow.append(raw_table[sl].sum())

        table['table'] = table_vals
        table['underflow'] = np.array(underflow)
        table['overflow'] = np.array(overflow)
        table['n_photons'] = n_photons
        table['phase_refractive_index'] = phase_refractive_index

        if n_dims == 5:
            # Space-time dimensions
            r_bin_edges = force_little_endian(pf_table[1].data) # meters # pylint: disable=no-member
            costheta_bin_edges = force_little_endian(pf_table[2].data) # pylint: disable=no-member
            t_bin_edges = force_little_endian(pf_table[3].data) # nanoseconds # pylint: disable=no-member

            # Photon directionality
            costhetadir_bin_edges = force_little_endian(pf_table[4].data) # pylint: disable=no-member
            deltaphidir_bin_edges = force_little_endian(pf_table[5].data) # pylint: disable=no-member

            #table['norm'] = (
            #    1
            #    / table['n_photons']
            #    / (SPEED_OF_LIGHT_M_PER_NS / table['phase_refractive_index']
            #       * np.mean(np.diff(table['t_bin_edges'])))
            #    * table['angular_acceptance_fract']
            #    * (len(table['costheta_bin_edges']) - 1)
            #)

            table['r_bin_edges'] = r_bin_edges
            table['costheta_bin_edges'] = costheta_bin_edges
            table['t_bin_edges'] = t_bin_edges
            table['costhetadir_bin_edges'] = costhetadir_bin_edges
            table['deltaphidir_bin_edges'] = deltaphidir_bin_edges

        else:
            raise NotImplementedError(
                '{}-dimensional table not handled'.format(n_dims)
            )

    finally:
        del pf_table
        if hasattr(fobj, 'close'):
            fobj.close()
        del fobj

    return table


def load_t_r_theta_table(fpath, depth_idx, scale=1, exponent=1,
                         photon_info=None):
    """Extract info from a file containing a (t, r, theta)-binned Retro table.

    Parameters
    ----------
    fpath : string
        Path to FITS file corresponding to the passed ``depth_idx``.

    depth_idx : int
        Depth index (e.g. from 0 to 59)

    scale : float
        Scaling factor to apply to the photon survival probability from the
        table, e.g. for quantum efficiency. This is applied _before_
        `exponent`. See `Notes` for more info.

    exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs. This is
        applied to each DOM's table _after_ `scale`. See `Notes` for more
        info.

    photon_info : None or RetroPhotonInfo namedtuple of dicts
        If None, creates a new RetroPhotonInfo namedtuple with empty dicts to
        fill. If one is provided, the existing component dictionaries are
        updated.

    Returns
    -------
    photon_info : RetroPhotonInfo namedtuple of dicts
        Tuple fields are 'survival_prob', 'theta', 'phi', and 'length'. Each
        dict is keyed by `depth_idx` and values are the arrays loaded
        from the FITS file.

    bin_edges : TimeSphCoord namedtuple
        Each element of the tuple is an array of bin edges.

    Notes
    -----
    The parameters `scale` and `exponent` modify a table's probability `P` by::

        P = 1 - (1 - P*scale)**exponent

    This allows for `scale` (which must be from 0 to 1) to be used for e.g.
    quantum efficiency--which always reduces the detection probability--and
    `exponent` (which must be 0 or greater) to be used as a systematic that
    modifies the post-`scale` probabilities up and down while keeping them
    valid (i.e., between 0 and 1). Larger values of `scale` (i.e., closer to 1)
    indicate a more efficient DOM. Likewise, values of `exponent` greater than
    one scale up the DOM efficiency, while values of `exponent` between 0 and 1
    scale the efficiency down.

    """
    # pylint: disable=no-member
    assert 0 <= scale <= 1
    assert exponent >= 0

    if photon_info is None:
        empty_dicts = []
        for _ in RetroPhotonInfo._fields:
            empty_dicts.append({})
        photon_info = RetroPhotonInfo(*empty_dicts)

    with pyfits.open(expand(fpath)) as table:
        data = force_little_endian(table[0].data)

        if scale == exponent == 1:
            photon_info.survival_prob[depth_idx] = data
        else:
            photon_info.survival_prob[depth_idx] = (
                1 - (1 - data * scale)**exponent
            )

        data = force_little_endian(table[1].data)
        photon_info.theta[depth_idx] = data

        data = force_little_endian(table[2].data)
        photon_info.deltaphi[depth_idx] = data

        data = force_little_endian(table[3].data)
        photon_info.length[depth_idx] = data

        # Note that we invert (reverse and multiply by -1) time edges; also,
        # no phi edges are defined in these tables.
        data = force_little_endian(table[4].data)
        t = - data[::-1]

        data = force_little_endian(table[5].data)
        r = data

        data = force_little_endian(table[6].data)
        theta = data

        bin_edges = TimeSphCoord(t=t, r=r, theta=theta,
                                 phi=np.array([], dtype=t.dtype))

    return photon_info, bin_edges


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pexp_t_r_theta(pinfo_gen, hit_time, dom_coord, survival_prob,
                   avg_photon_theta, avg_photon_length, use_directionality,
                   t_min, t_max, n_t_bins, r_min, r_max, r_power, n_r_bins,
                   n_costheta_bins):
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
    t_min : float
    t_max : float
    n_t_bins : int
    r_min : float
    r_max : float
    r_power : float
    n_r_bins : int
    n_costheta_bins : int

    Returns
    -------
    expected_photon_count : float

    """
    table_dt = (t_max - t_min) / n_t_bins
    expected_photon_count = 0.0
    inv_r_power = 1 / r_power
    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable

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

        tbin_idx = int(np.floor((dt - t_min) / table_dt))
        #if tbin_idx < 0 or tbin_idx >= -POL_TABLE_DT:
        if tbin_idx < -n_t_bins or tbin_idx >= 0:
            #print('t')
            continue

        rbin_idx = int(r**inv_r_power // POL_TABLE_DRPWR)
        if rbin_idx < 0 or rbin_idx >= POL_TABLE_NRBINS:
            #print('r')
            continue

        thetabin_idx = int(dz / (r * POL_TABLE_DCOSTHETA))
        if thetabin_idx < 0 or thetabin_idx >= POL_TABLE_NTHETABINS:
            #print('theta')
            continue

        #print(tbin_idx, rbin_idx, thetabin_idx)
        #raise Exception()
        surviving_count = (
            p_count * survival_prob[tbin_idx, rbin_idx, thetabin_idx]
        )

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        # TODO: Incorporate photon direction info
        if use_directionality:
            pass

        expected_photon_count += surviving_count

    return expected_photon_count


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pexp_xyz(pinfo_gen, x_min, y_min, z_min, nx, ny, nz, binwidth,
             survival_prob, avg_photon_x, avg_photon_y, avg_photon_z,
             use_directionality):
    """Compute the expected number of detected photons in _all_ DOMs at _all_
    times.

    Parameters
    ----------
    pinfo_gen :
    x_min, y_min, z_min :
    nx, ny, nz :
    binwidth :
    survival_prob :
    avg_photon_x, avg_photon_y, avg_photon_z :
    use_directionality : bool

    """
    expected_photon_count = 0.0
    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable
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
            raise NotImplementedError('Directionality cannot be used yet')

        expected_photon_count += surviving_count

    return expected_photon_count


class CLSimTable(object):
    """Load and use information from a single "raw" individual-DOM
    (time, r, theta, thetadir, deltaphidir)-binned Retro table.

    Note that this is the table generated by CLSim, prior to any manipulations
    performed for using the table with Retro, and is in the FITS file format.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    string : int
        Indexed from 1

    depth_idx : int

    angular_acceptance_fract : float
        Normalization for angular acceptance being less than 1

    """
    def __init__(self, fpath=None, tables_dir=None, hash_val=None, string=None,
                 depth_idx=None, seed=None, angular_acceptance_fract=0.338019664877):
        # Translation and validation of args
        assert 0 < angular_acceptance_fract <= 1
        self.angular_acceptance_fract = angular_acceptance_fract

        if fpath is None:
            tables_dir = expand(tables_dir)
            assert isdir(tables_dir)
            self.tables_dir = tables_dir
            self.hash_val = hash_val
            self.seed = seed

            if isinstance(string, basestring):
                self.string = string.strip().lower()
                assert self.string in ('ic', 'dc')
                self.subdet = self.string
                self.string_idx = None

            else:
                self.string = string
                self.string_idx = self.string - 1

                if self.string_idx < 79:
                    self.subdet = 'ic'
                else:
                    self.subdet = 'dc'

            self.depth_idx = depth_idx

            fname = CLSIM_TABLE_FNAME_V2_PROTO.format(
                hash_val=hash_val,
                string=self.string,
                depth_idx=self.depth_idx,
                seed=self.seed
            )
            self.fpath = join(self.tables_dir, fname)

        else:
            assert tables_dir is None
            assert hash_val is None
            assert string is None
            assert depth_idx is None

            self.fpath = fpath
            self.tables_dir = dirname(fpath)
            info = interpret_clsim_table_fname(fpath)
            self.string = info['string']
            if isinstance(self.string, int):
                self.string_idx = self.string - 1
            else:
                self.string_idx = None
            self.hash_val = info['hash_val']
            self.depth_idx = info['depth_idx']
            self.seed = info['seed']

        assert self.subdet in ('ic', 'dc')
        self.dtp_fname_proto = RETRO_DOM_TABLE_FNAME_PROTO

        table_info = load_clsim_table(self.fpath)

        self.table = table_info.pop('table')
        self.table_shape = table_info.pop('table_shape')
        self.n_dims = len(self.table_shape)
        self.n_photons = table_info.pop('n_photons')
        self.phase_refractive_index = table_info.pop('phase_refractive_index')
        self.angular_acceptance_fract = angular_acceptance_fract

        self.r_bin_edges = table_info.pop('r_bin_edges')
        self.t_bin_edges = table_info.pop('t_bin_edges')
        self.costheta_bin_edges = table_info.pop('costheta_bin_edges')

        self.theta_bin_edges = np.arccos(self.costheta_bin_edges) # radians
        self.costheta_centers = linear_bin_centers(self.costheta_bin_edges)
        self.theta_centers = np.arccos(self.costheta_centers) # radians

        t_bin_widths = np.diff(self.t_bin_edges)
        assert np.allclose(t_bin_widths, t_bin_widths[0])
        self.t_bin_width = np.mean(t_bin_widths)

        if self.n_dims == 5:
            self.costhetadir_bin_edges = table_info.pop('costhetadir_bin_edges')
            self.deltaphidir_bin_edges = table_info.pop('deltaphidir_bin_edges') # radians
            self.costhetadir_centers = linear_bin_centers(self.costhetadir_bin_edges)
            self.thetadir_bin_edges = np.arccos(self.costhetadir_bin_edges) # radians
            self.thetadir_centers = np.arccos(self.costhetadir_centers) # radians
            self.deltaphidir_centers = linear_bin_centers(self.deltaphidir_bin_edges) # radians
        else:
            raise NotImplementedError(
                'Can only work with CLSim tables with 5 dimensions; got %d'
                % self.n_dims
            )

        del table_info

        #with pyfits.open(self.fpath) as table:
        #    # Cut off first and last bin in each dimension (underflow and

        #    self.n_photons = force_little_endian(table[0].header['_i3_n_photons'])
        #    self.phase_refractive_index = force_little_endian(table[0].header['_i3_n_phase'])

        #    # Space-time dimensions
        #    self.r_bin_edges = force_little_endian(table[1].data) # meters
        #    self.costheta_bin_edges = force_little_endian(table[2].data)
        #    self.t_bin_edges = force_little_endian(table[3].data) # nanoseconds

        #    # Photon directionality
        #    self.costhetadir_bin_edges = force_little_endian(table[4].data)
        #    self.deltaphidir_bin_edges = force_little_endian(table[5].data)

        # Multiply the tabulated photon counts by a normalization factor to
        # arrive at a (reasonable, but still imperfect) suvival
        # probability. Normalization is performed by:
        # * Dividing by total photons thrown
        # * Dividing by (speed of light in ice, c / n, in m/ns
        #    multiplied by the bin width in ns; overall units are m)
        # * Multiplying by a correction for angular acceptance, in [0, 1]
        # * Multiplying by the number of costheta bins (>= 1)
        # The result of applying this norm is in units of 1/meter, where
        # this unit accounts for the fact that CLSim tabulates the same
        # photon every meter it travels, hence the same photon results in
        # multiple counts (as many as there are meters in its path).
        self.norm = (
            1
            / self.n_photons
            / (SPEED_OF_LIGHT_M_PER_NS / self.phase_refractive_index
               * self.t_bin_width)
            * self.angular_acceptance_fract
            * (len(self.costheta_bin_edges) - 1)
        )

        # The photon direction is tabulated in dimensions 3 and 4
        #self.survival_prob = self.data.sum(axis=(3, 4)) * self.norm

    def export_t_r_theta_table(self, outdir=None, overwrite=True):
        """Distill binned photon directionality information into a single
        vector per bin and force azimuthal symmetry to reduce the table from a
        5D histogram of photon counts binned in
        (t, r, costheta, phi, thetadir, deltaphidir) to a 3D histogram binned in
        (t, r, costheta) where each bin contains a probability and an average
        direction vector.

        The resulting file will be placed in the same directory as the source
        table and the file name will be the source filename suffixed by
        "_r_cz_t_angles" (prior to the extension).

        Parameters
        ----------
        outdir : string, optional
            If specified, store the DOM-time-polar table into this directory.
            Otherwise, if not specified, the table is stored in the same
            directory as the source table.

        """
        if outdir is None:
            outdir = self.tables_dir
        outdir = expand(outdir)

        new_fname = self.dtp_fname_proto.format(depth_idx=self.depth_idx)
        new_fpath = join(outdir, new_fname)

        if not isdir(outdir):
            makedirs(outdir)

        if isfile(new_fpath):
            if overwrite:
                print('WARNING: overwriting existing file at "%s"' % new_fpath)
                remove(new_fpath)
            else:
                print('ERROR: There is an existing file at "%s"; not'
                      ' proceeding.' % new_fpath)
                return

        survival_probs, average_thetas, average_phis, lengths = \
            generate_t_r_theta_table(
                table=self.table,
                n_photons=self.n_photons,
                phase_refractive_index=self.phase_refractive_index,
                t_bin_width=self.t_bin_width,
                angular_acceptance_fract=self.angular_acceptance_fract,
                thetadir_centers=self.thetadir_centers,
                deltaphidir_centers=self.deltaphidir_centers,
                theta_bin_edges=self.theta_bin_edges
            )
        objects = [
            pyfits.PrimaryHDU(survival_probs),
            pyfits.ImageHDU(average_thetas.astype(np.float32)),
            pyfits.ImageHDU(average_phis.astype(np.float32)),
            pyfits.ImageHDU(lengths.astype(np.float32)),
            pyfits.ImageHDU(self.t_bin_edges.astype(np.float32)),
            pyfits.ImageHDU(self.r_bin_edges.astype(np.float32)),
            pyfits.ImageHDU(self.theta_bin_edges[::-1].astype(np.float32))
        ]

        hdulist = pyfits.HDUList(objects)
        hdulist.writeto(new_fpath)


class DOMTimePolarTables(object):
    """Load and use information from individual-dom (t,r,theta)-binned Retro
    tables.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float64

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts.

    ic_exponent, dc_exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs.
        `ic_exponent` is applied to IceCube (non-DeepCore) DOMs and
        `dc_exponent` is applied to DeepCore DOMs. Note that this is applied to
        each DOM's table after the appropriate quantum efficiency scale factor
        has already been applied (quantum efficiency is applied as a simple
        multiplier; see :attr:`retro.IC_DOM_QUANT_EFF` and
        :attr:`retro.DC_DOM_QUANT_EFF`).

    """
    def __init__(self, tables_dir, hash_val, geom, use_directionality,
                 ic_exponent=1, dc_exponent=1):
        # Translation and validation of args
        tables_dir = expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        assert ic_exponent >= 0
        assert dc_exponent >= 0

        self.tables_dir = tables_dir
        self.hash_val = hash_val
        self.geom = geom
        self.use_directionality = use_directionality
        self.ic_exponent = ic_exponent
        self.dc_exponent = dc_exponent
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
            dom_quant_eff = IC_DOM_QUANT_EFF
            exponent = self.ic_exponent
        else:
            subdet = 'dc'
            dom_quant_eff = DC_DOM_QUANT_EFF
            exponent = self.dc_exponent

        if not force_reload and depth_idx in self.tables[subdet]:
            return

        det= 'IC' if subdet=='ic' else 'DC'

        #fpath = join(
        #    self.tables_dir,
        #    RETRO_DOM_TABLE_FNAME_PROTO.format(depth_idx=depth_idx)
        #)
        fpath = join(
            self.tables_dir,
            RETRO_DOM_TABLE_FNAME_PROTO.format(dom=depth_idx,det=det)
        )

        photon_info, _ = load_t_r_theta_table(
            fpath=fpath,
            depth_idx=depth_idx,
            scale=dom_quant_eff,
            exponent=exponent
        )

        #length = photon_info.length[depth_idx]
        #deltaphi = photon_info.deltaphi[depth_idx]
        self.tables[subdet][depth_idx] = RetroPhotonInfo(
            survival_prob=photon_info.survival_prob[depth_idx],
            theta=photon_info.theta[depth_idx],
            deltaphi=photon_info.deltaphi[depth_idx],
            length=(photon_info.length[depth_idx]
                    * np.cos(photon_info.deltaphi[depth_idx]))
        )

    def load_tables(self):
        """Load all tables"""
        # TODO: parallelize the loading of each table to reduce CPU overhead
        # time (though most time I expect to be disk-read times, this could
        # still help speed the process up)
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
        """Load all tables that match the `proto_tile_hash`; if multiple tables
        match, then stitch these together into one large TDI table."""
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
            expand(self.tables_dir), 'retro_tdi_table_*survival_prob.fits'
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
                    TDI_TABLE_FNAME_PROTO.format(
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
        print('Time to load: {} s'.format(np.round(time.time() - t0, 3)))

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
