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


# TODO remove the following; this and CLSimTables should be supplanted by
# single class, Retro5DTables

#class CKVTables(CLSimTables):
#    """
#    Class to interact with and obtain photon survival probabilities from a set
#    of Retro 5D Cherenkov tables.
#
#    Parameters
#    ----------
#    geom : shape-(n_strings, n_doms, 3) array
#        x, y, z coordinates of all DOMs, in meters relative to the IceCube
#        coordinate system
#
#    rde : shape-(n_strings, n_doms) array
#        Relative DOM efficiencies (this accounts for quantum efficiency). Any
#        DOMs with either 0 or NaN rde will be disabled and return 0's for
#        expected photon counts.
#
#    compute_t_indep_exp : bool
#
#    noise_rate_hz : shape-(n_strings, n_doms) array
#        Noise rate for each DOM, in Hz.
#
#    use_directionality : bool
#        Enable or disable directionality when computing expected photons at
#        the DOMs
#
#    norm_version : string
#        (Temporary) Which version of the norm to use. Only for experimenting,
#        and will be removed once we figure the norm out.
#
#    """
#    def __init__(
#            self, geom, rde, compute_t_indep_exp, noise_rate_hz,
#            use_directionality, norm_version
#        ):
#        super(CKVTables, self).__init__(
#            geom=geom,
#            rde=rde,
#            compute_t_indep_exp=compute_t_indep_exp,
#            noise_rate_hz=noise_rate_hz,
#            use_directionality=use_directionality,
#            num_phi_samples=0,
#            ckv_sigma_deg=0,
#            norm_version=norm_version
#        )
#        self.table_loader_func = load_ckv_table
#        self.table_kind = TBL_KIND_CKV
#        self.usable_table_slice = (slice(None),)*5
#        self.t_indep_table_name = 't_indep_ckv_table'
#        self.table_name = 'ckv_table'
#
#    #@profile
#    def get_photon_expectation(
#            self, sources, hit_times, string, dom, include_noise=False,
#            time_window=None
#        ):
#        """
#        Parameters
#        ----------
#        sources : shape (num_sources,) array of dtype SRC_DTYPE
#            Info about photons generated photons by the event hypothesis.
#
#        hit_times : shape (num_hits,) array of floats, units of ns
#
#        string : int in [1, 86]
#
#        dom : int in [1, 60]
#
#        include_noise : bool
#            Include noise in the photon expectations (both at hit time and
#            time-independent). Non-operational DOMs return 0 for both return values
#
#        time_window : float in units of ns
#            Time window for computing the "time-independent" noise expectation.
#            Used (and required) if `include_noise` is True.
#
#        Returns
#        -------
#        photons_at_hit_times : shape (num_hits,) array of floats
#        photons_at_all_times : float
#
#        """
#        # `string` and `dom` are 1-indexed but numpy array indices are
#        # 0-indexed
#        string_idx = string - 1
#        dom_idx = dom - 1
#        if not self.operational_doms[string_idx, dom_idx]:
#            return np.zeros_like(hit_times, dtype=np.float64), np.float64(0)
#
#        dom_coord = self.geom[string_idx, dom_idx]
#        dom_quantum_efficiency = self.quantum_efficiency[string_idx, dom_idx]
#
#        if self.string_aggregation == AGG_STR_ALL:
#            string = STR_ALL
#        elif self.string_aggregation == AGG_STR_SUBDET:
#            if string < 79:
#                string = STR_IC
#            else:
#                string = STR_DC
#
#        if self.depth_aggregation:
#            dom = DOM_ALL
#
#        (pexp_5d,
#         t_indep_ckv_table,
#         t_indep_table_norm,
#         ckv_table,
#         table_norm) = self.tables[(string, dom)]
#
#        photons_at_hit_times, photons_at_all_times = pexp_5d(
#            sources=sources,
#            hit_times=hit_times,
#            dom_coord=dom_coord,
#            quantum_efficiency=dom_quantum_efficiency,
#            table=ckv_table,
#            table_norm=table_norm,
#            t_indep_table=t_indep_ckv_table,
#            t_indep_table_norm=t_indep_table_norm,
#        )
#
#        if include_noise:
#            dom_noise_rate_per_ns = self.noise_rate_per_ns[string_idx, dom_idx]
#            photons_at_hit_times += dom_noise_rate_per_ns
#            photons_at_all_times += dom_noise_rate_per_ns * time_window
#
#        return photons_at_hit_times, photons_at_all_times
