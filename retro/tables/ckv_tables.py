# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for using a set of 5D (r, costheta, t, costhetadir, deltaphidir)
Cherenkov Retro tables (5D CLSim tables with directionality map convolved with
a Cherenkov cone)
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    CKV_TABLE_KEYS
    load_ckv_table
    CKVTables
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

from collections import OrderedDict
from os.path import abspath, dirname, isdir, isfile, join
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DEBUG
from retro.tables.pexp_5d import TBL_KIND_CKV, generate_pexp_5d_function
from retro.tables.clsim_tables import TABLE_NORM_KEYS, get_table_norm
from retro.utils.misc import expand, wstderr


CKV_TABLE_KEYS = [
    'n_photons', 'phase_refractive_index', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges', 'costhetadir_bin_edges',
    'deltaphidir_bin_edges', 'ckv_table', #'t_indep_ckv_table'
]


def load_ckv_table(fpath, mmap, step_length=None):
    """Load a Cherenkov table from disk.

    Parameters
    ----------
    fpath : string
        Path to directory containing the table's .npy files.

    mmap : bool
        Whether to memory map the table (if it's stored in a directory
        containing .npy files).

    step_length : float > 0 in units of meters, optional
        Required if computing the `t_indep_table` (if `gen_t_indep` is True).

    Returns
    -------
    table : OrderedDict
        Items are
        - 'n_photons' :
        - 'phase_refractive_index' :
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'ckv_table' : np.ndarray
        - 't_indep_ckv_table' : np.ndarray

    """
    fpath = expand(fpath)
    table = OrderedDict()

    if DEBUG:
        wstderr('Loading table from {} ...\n'.format(fpath))

    assert isdir(fpath), fpath
    t0 = time()
    indir = fpath

    if mmap:
        mmap_mode = 'r'
    else:
        mmap_mode = None

    for key in CKV_TABLE_KEYS: # TODO: + ['t_indep_ckv_table']:
        fpath = join(indir, key + '.npy')
        if DEBUG:
            wstderr('    loading {} from "{}" ...'.format(key, fpath))

        t1 = time()
        if isfile(fpath):
            table[key] = np.load(fpath, mmap_mode=mmap_mode)
        elif key != 't_indep_ckv_table':
            raise ValueError(
                'Could not find file "{}" for loading table key "{}"'
                .format(fpath, key)
            )

        if DEBUG:
            wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))

    if step_length is not None and 'step_length' in table:
        assert step_length == table['step_length']

    if DEBUG:
        wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))

    return table


class CKVTables(object):
    """
    Class to interact with and obtain photon survival probabilities from a set
    of Retro 5D Cherenkov tables.

    Parameters
    ----------
    geom : shape-(n_strings, n_doms, 3) array
        x, y, z coordinates of all DOMs, in meters relative to the IceCube
        coordinate system

    rde : shape-(n_strings, n_doms) array
        Relative DOM efficiencies (this accounts for quantum efficiency). Any
        DOMs with either 0 or NaN rde will be disabled and return 0's for
        expected photon counts.

    noise_rate_hz : shape-(n_strings, n_doms) array
        Noise rate for each DOM, in Hz.

    use_directionality : bool
        Enable or disable directionality when computing expected photons at
        the DOMs

    norm_version : string
        (Temporary) Which version of the norm to use. Only for experimenting,
        and will be removed once we figure the norm out.

    """
    def __init__(
            self, geom, rde, noise_rate_hz, use_directionality, norm_version
        ):
        assert len(geom.shape) == 3
        self.geom = geom
        self.use_directionality = use_directionality
        self.norm_version = norm_version

        zero_mask = rde == 0
        nan_mask = np.isnan(rde)
        inf_mask = np.isinf(rde)
        num_zero = np.count_nonzero(zero_mask)
        num_nan = np.count_nonzero(nan_mask)
        num_inf = np.count_nonzero(inf_mask)

        if num_nan or num_inf or num_zero:
            print(
                "WARNING: RDE is zero for {} DOMs, NaN for {} DOMs and +/-inf"
                " for {} DOMs.\n"
                "These DOMs will be disabled and return 0's forexpected photon"
                " computations."
                .format(num_zero, num_nan, num_inf)
            )
        mask = zero_mask | nan_mask | inf_mask

        self.operational_doms = ~mask
        self.rde = np.ma.masked_where(mask, rde)
        self.quantum_efficiency = 0.25 * self.rde
        self.noise_rate_hz = np.ma.masked_where(mask, noise_rate_hz)

        self.tables = {}
        self.string_aggregation = None
        self.depth_aggregation = None
        self.pexp_func = None
        self.binning_info = None

    def load_table(
            self, fpath, string, dom, step_length, angular_acceptance_fract,
            mmap
        ):
        """Load a table into the set of tables.

        Parameters
        ----------
        fpath : string
            Path to the directory containing the table's .npy files.

        string : int in [1, 86] or str in {'ic', 'dc', or 'all'}

        dom : int in [1, 60] or str == 'all'

        step_length : float > 0
            The stepLength parameter (in meters) used in CLSim tabulator code
            for tabulating a single photon as it travels. This is a hard-coded
            paramter set to 1 meter in the trunk version of the code, but it's
            something we might play with to optimize table generation speed, so
            just be warned that this _can_ change.

        angular_acceptance_fract : float in (0, 1]
            Constant normalization factor to apply to correct for the integral
            of the angular acceptance curve used in the simulation that
            produced this table.

        mmap : bool
            Whether to attempt to memory map the table.

        """
        single_dom_spec = True
        if isinstance(string, basestring):
            string = string.strip().lower()
            assert string in ['ic', 'dc', 'all']
            agg_mode = 'all' if string == 'all' else 'subdetector'
            if self.string_aggregation is None:
                self.string_aggregation = agg_mode
            assert agg_mode == self.string_aggregation
            single_dom_spec = False
        else:
            if self.string_aggregation is None:
                self.string_aggregation = False
            # `False` is ok but `None` is not ok
            assert self.string_aggregation == False # pylint: disable=singleton-comparison
            assert 1 <= string <= 86

        if isinstance(dom, basestring):
            dom = dom.strip().lower()
            assert dom == 'all'
            if self.depth_aggregation is None:
                self.depth_aggregation = True
            assert self.depth_aggregation
            single_dom_spec = False
        else:
            if self.depth_aggregation is None:
                self.depth_aggregation = False
            # `False` is ok but `None` is not ok
            assert self.depth_aggregation == False # pylint: disable=singleton-comparison
            assert 1 <= dom <= 60

        assert step_length > 0
        assert 0 < angular_acceptance_fract <= 1

        if single_dom_spec and not self.operational_doms[string - 1, dom - 1]:
            print(
                'WARNING: String {}, DOM {} is not operational, skipping'
                ' loading the corresponding table'.format(string, dom)
            )
            return

        table = load_ckv_table(
            fpath=fpath,
            step_length=step_length,
            mmap=mmap
        )

        table['step_length'] = step_length
        table['table_norm'] = get_table_norm(
            angular_acceptance_fract=angular_acceptance_fract,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: table[k] for k in TABLE_NORM_KEYS}
        )
        #table['t_indep_table_norm'] = angular_acceptance_fract

        pexp_5d, _ = generate_pexp_5d_function(
            table=table,
            table_kind=TBL_KIND_CKV,
            use_directionality=self.use_directionality,
            num_phi_samples=0,
            ckv_sigma_deg=0
        )

        # NOTE: original tables have underflow (bin 0) and overflow (bin -1)
        # bins, so whole-axis slices must exclude the first and last bins.
        self.tables[(string, dom)] = (
            pexp_5d,
            #table['t_indep_ckv_table'],
            #table['t_indep_table_norm'],
            table['ckv_table'],
            table['table_norm'],
        )

    #@profile
    def get_photon_expectation(
            self, pinfo_gen, hit_time, time_window, string, dom
        ):
        """
        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy.ndarray
            Info about photons generated photons by the event hypothesis.

        hit_time : float, units of ns
        time_window : float, units of ns
        string : int in [1, 86]
        dom : int in [1, 60]

        Returns
        -------
        total_photon_count, expected_photon_count : float
            See pexp_t_r_theta

        """
        # `string` and `dom` are 1-indexed but array indices are 0-indexed
        string_idx, dom_idx = string - 1, dom - 1
        if not self.operational_doms[string_idx, dom_idx]:
            return 0, 0

        dom_coord = self.geom[string_idx, dom_idx]
        dom_quantum_efficiency = self.quantum_efficiency[string_idx, dom_idx]
        dom_noise_rate_hz = self.noise_rate_hz[string_idx, dom_idx]

        if self.string_aggregation == 'all':
            string = 'all'
        elif self.string_aggregation == 'subdetector':
            if string < 79:
                string = 'ic'
            else:
                string = 'dc'

        if self.depth_aggregation:
            dom = 'all'

        #print('string =', string, 'dom =', dom)

        (pexp_5d,
         #t_indep_ckv_table,
         #t_indep_table_norm,
         ckv_table,
         table_norm) = self.tables[(string, dom)]

        return pexp_5d(
            pinfo_gen=pinfo_gen,
            hit_time=hit_time,
            time_window=time_window,
            dom_coord=dom_coord,
            noise_rate_hz=dom_noise_rate_hz,
            quantum_efficiency=dom_quantum_efficiency,
            table=ckv_table,
            table_norm=table_norm,
            #t_indep_table=t_indep_ckv_table,
            #t_indep_table_norm=t_indep_table_norm,
        )
