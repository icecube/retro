# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, range-builtin-not-iterating

"""
Class for using a set of "raw" 5D (r, costheta, t, costhetadir, deltaphidir)
Retro tables, 5D Cherenkov tables, or template-compressed versions thereof.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'TABLE_NORM_KEYS',
    'TABLE_KINDS',
    'NORM_VERSIONS',
    'Retro5DTables',
    'get_table_norm',
]

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
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.const import (
    ALL_STRS_DOMS, ALL_STRS_DOMS_SET, NUM_DOMS_TOT, SPEED_OF_LIGHT_M_PER_NS,
    PI, TWO_PI, get_string_dom_pair
)
from retro.i3info.angsens_model import load_angsens_model
from retro.retro_types import DOM_INFO_T
from retro.tables.pexp_5d import generate_pexp_5d_function
from retro.utils.geom import spherical_volume
from retro.utils.misc import expand


TABLE_NORM_KEYS = [
    'n_photons', 'group_refractive_index', 'step_length', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges'
]
"""All besides 'quantum_efficiency' and 'avg_angsens'"""

TABLE_KINDS = [
    'raw_uncompr', 'raw_templ_compr', 'ckv_uncompr', 'ckv_templ_compr'
]

NORM_VERSIONS = [
    'avgsurfarea', 'binvol', 'binvol2', 'binvol3', 'binvol4', 'binvol5',
    'binvol6', 'binvol7', 'pde', 'wtf', 'wtf2'
]


class Retro5DTables(object):
    """
    Class to interact with and obtain photon survival probabilities from a set
    of Retro 5D tables.

    These include "raw" tables produced directly by CLSim, Cherenkov tables
    (the former convolved with a Cherenkov cone), and either of these employing
    template-based compression.

    Parameters
    ----------
    table_kind

    geom : shape-(n_strings, n_doms, 3) array
        x, y, z coordinates of all DOMs, in meters relative to the IceCube
        coordinate system

    rde : shape-(n_strings, n_doms) array
        Relative DOM efficiencies (this accounts for quantum efficiency). Any
        DOMs with either 0 or NaN rde will be disabled and return 0's for
        expected photon counts.

    noise_rate_hz : shape-(n_strings, n_doms) array
        Noise rate for each DOM, in Hz.

    angsens_model : string

    compute_t_indep_exp : bool

    use_directionality : bool
        Enable or disable directionality when computing expected photons at
        the DOMs

    norm_version : string
        (Temporary) Which version of the norm to use. Only for experimenting,
        and will be removed once we figure the norm out.

    num_phi_samples : int > 0
        If using directionality, set how many samples around the Cherenkov cone
        are used to find which (costhetadir, deltaphidir) bins to use to
        compute expected photon survival probability.

    ckv_sigma_deg : float >= 0
        If using directionality, Gaussian-smear the Cherenkov angle by this
        meany degrees by randomly distributing the phi samples. Higher
        `ckv_sigma_deg` could necessitate higher `num_phi_samples` to get an
        accurate "smearing."

    template_library : shape-(n_templates, n_dir_theta, n_dir_deltaphi) array
        Containing the directionality templates for compressed tables

    use_sd_indices : sequence of int, optional
        Only use a subset of DOMs. If not specified, all in-ice DOMs are used.

    """
    def __init__(
            self,
            table_kind,
            geom,
            rde,
            noise_rate_hz,
            angsens_model,
            compute_t_indep_exp,
            use_directionality,
            norm_version,
            num_phi_samples=None,
            ckv_sigma_deg=None,
            template_library=None,
            use_sd_indices=ALL_STRS_DOMS
        ):
        self.angsens_poly, self.avg_angsens = load_angsens_model(angsens_model)
        self.angsens_model = angsens_model
        self.compute_t_indep_exp = compute_t_indep_exp
        self.table_kind = table_kind

        self.use_sd_indices = np.asarray(use_sd_indices, dtype=np.uint32)
        if not set(self.use_sd_indices) == ALL_STRS_DOMS_SET:
            raise ValueError('Requesting to use fewer than all DOMs is not supported.')
        self.use_sd_indices_set = set(use_sd_indices)
        self.loaded_sd_indices = np.empty(shape=0, dtype=np.uint32)

        self.tbl_is_raw = table_kind in ['raw_uncompr', 'raw_templ_compr']
        self.tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']

        self.tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']

        # xor: either raw or ckv, but not both
        assert ((self.tbl_is_raw or self.tbl_is_ckv)
                and not (self.tbl_is_raw and self.tbl_is_ckv))

        if self.tbl_is_templ_compr and template_library is None:
            raise ValueError('Template library is needed to use compressed table')
        self.template_library = template_library

        if self.tbl_is_templ_compr:
            if not self.tbl_is_ckv:
                raise NotImplementedError('Can only handle ckv template-compr tables')
            from retro.tables.template_compr_ckv_tables import (
                load_template_compr_ckv_table
            )
            self.table_loader_func = load_template_compr_ckv_table
            self.usable_table_slice = (slice(None),)*3
            self.t_indep_table_name = 't_indep_ckv_table'
            self.table_name = 'ckv_template_map'
        elif self.tbl_is_raw:
            from retro.tables.clsim_tables import load_clsim_table_minimal
            self.table_loader_func = load_clsim_table_minimal
            # NOTE: original tables have underflow (bin 0) and overflow
            # (bin -1) bins, so whole-axis slices must exclude the first and
            # last bins.
            self.usable_table_slice = (slice(1, -1),)*5
            self.t_indep_table_name = 't_indep_table'
            self.table_name = 'table'
        else: # self.tbl_is_ckv:
            from retro.tables.ckv_tables import load_ckv_table
            self.table_loader_func = load_ckv_table
            self.usable_table_slice = (slice(None),)*5
            self.t_indep_table_name = 't_indep_ckv_table'
            self.table_name = 'ckv_table'

        assert len(geom.shape) == 3
        self.use_directionality = use_directionality
        self.num_phi_samples = num_phi_samples
        self.ckv_sigma_deg = ckv_sigma_deg
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
                "These DOMs will be disabled and return 0's for expected"
                " photon computations. {} DOMs remain."
                .format(num_zero, num_nan, num_inf,
                        NUM_DOMS_TOT - num_zero - num_nan - num_inf)
            )
        mask = zero_mask | nan_mask | inf_mask
        operational_doms = ~mask

        self.dom_info = np.empty(NUM_DOMS_TOT, dtype=DOM_INFO_T)
        for sd_idx in range(NUM_DOMS_TOT):
            string_num, dom_num = get_string_dom_pair(sd_idx)
            string_idx, dom_idx = string_num - 1, dom_num - 1

            this_dom_info = self.dom_info[sd_idx]
            this_dom_info['sd_idx'] = sd_idx

            operational = operational_doms[string_idx, dom_idx]
            if sd_idx in self.use_sd_indices:
                if not operational:
                    self.use_sd_indices_set.remove(sd_idx)
                    self.use_sd_indices = np.array(sorted(self.use_sd_indices_set),
                                                   dtype=np.uint32)
            else:
                operational = False
            this_dom_info['operational'] = operational

            this_dom_info['x'] = geom[string_idx, dom_idx, 0]
            this_dom_info['y'] = geom[string_idx, dom_idx, 1]
            this_dom_info['z'] = geom[string_idx, dom_idx, 2]
            this_dom_info['quantum_efficiency'] = 0.25 * rde[string_idx, dom_idx]
            this_dom_info['noise_rate_per_ns'] = (
                operational * noise_rate_hz[string_idx, dom_idx] / 1e9
            )

        self.tables = []
        self.t_indep_tables = []
        self.table_norms = []
        self.t_indep_table_norms = []
        self.n_photons_per_table = []
        self.table_meta = None
        self.table_norm = None
        self.t_indep_table_norm = None

        # Create `sd_idx_table_indexer` which, at each string-dom index
        # `sd_idx` stores the index `table_idx` that points to the table to use
        # for that dom. E.g., to retrive the table to use for a given `sd_idx`:
        #
        #   table = self.tables[self.sd_idx_table_indexer[sd_idx]]
        #
        # Note that the value ``np.iinfo(itype).min`` should be an invalid
        # index into an array of length <= ``np.iinfo(itype).min`` since there
        # is one more negative integer than positive in IEEE representations of
        # integers.
        self.sd_idx_table_indexer = np.full(
            shape=NUM_DOMS_TOT,
            fill_value=np.iinfo(np.int32).min,
            dtype=np.int32
        )

        self._pexp = None
        self._get_llh = None
        self.pexp_meta = None
        self.is_stacked = None
        self.t_is_residual_time = None

    def get_llh(self, *args, **kwargs):
        return self._get_llh(*args, **kwargs)

    def pexp(self, *args, **kwargs):
        if self._pexp is None:
            raise ValueError('No pexp function initialized. (Are tables loaded?)')
        return self._pexp(*args, **kwargs)

    def load_stacked_tables(
            self,
            stacked_tables_meta_fpath,
            stacked_tables_fpath,
            stacked_t_indep_tables_fpath,
            mmap_tables=False,
            mmap_t_indep=False,
    ):
        if self.is_stacked is not None:
            assert self.is_stacked

        stacked_tables_meta_fpath = expand(stacked_tables_meta_fpath)
        stacked_tables_fpath = expand(stacked_tables_fpath)
        stacked_t_indep_tables_fpath = expand(stacked_t_indep_tables_fpath)

        tables_mmap_mode = 'r' if mmap_tables else None
        t_indep_mmap_mode = 'r' if mmap_t_indep else None

        self.table_meta = load_pickle(stacked_tables_meta_fpath)
        self.tables = np.load(stacked_tables_fpath, mmap_mode=tables_mmap_mode)
        self.tables.setflags(write=False, align=True, uic=False)
        num_tables = self.tables.shape[0]

        self.t_is_residual_time = bool(self.table_meta.get('t_is_residual_time', False))

        self.t_indep_tables = np.load(
            stacked_t_indep_tables_fpath,
            mmap_mode=t_indep_mmap_mode
        )
        self.t_indep_tables.setflags(write=False, align=True, uic=False)
        assert self.t_indep_tables.shape[0] == num_tables

        if self._pexp is None:
            pexp, get_llh, pexp_meta = generate_pexp_5d_function(
                table=self.table_meta,
                table_kind=self.table_kind,
                t_is_residual_time=self.t_is_residual_time,
                compute_t_indep_exp=self.compute_t_indep_exp,
                compute_unhit_doms=True, # TODO: modify when TDI table works
                use_directionality=self.use_directionality,
                num_phi_samples=self.num_phi_samples,
                ckv_sigma_deg=self.ckv_sigma_deg,
                template_library=self.template_library
            )
            self._pexp = pexp
            self._get_llh = get_llh
            self.pexp_meta = pexp_meta

        self.sd_idx_table_indexer = deepcopy(self.table_meta['sd_idx_table_indexer'])
        self.sd_idx_table_indexer.setflags(write=False, align=True, uic=False)

        self.loaded_sd_indices = np.array([
            sd_idx for sd_idx in range(NUM_DOMS_TOT)
            if self.dom_info[sd_idx]['operational']
        ], dtype=np.uint32)
        assert set(self.loaded_sd_indices) == self.use_sd_indices_set

        self.n_photons_per_table = self.table_meta['n_photons_per_table']

        # Note that in creating the stacked tables, each indiividual table
        # is scaled such that the effective number of photons used to generate
        # the table is one (to avoid different norms across the tables if
        # different number of photons was used originally to create each).
        self.table_norm, self.t_indep_table_norm = get_table_norm(
            avg_angsens=self.avg_angsens,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: self.table_meta[k] for k in TABLE_NORM_KEYS}
        )

        self.table_norms = [self.table_norm] * num_tables
        self.t_indep_table_norms = [self.t_indep_table_norm] * num_tables

        self.is_stacked = True

    def load_table(self, fpath, sd_indices, mmap, step_length=None):
        """Load a single table (possibly applicable to more than one DOM) into
        the set of tables.

        Parameters
        ----------
        fpath : string
            Path to the table .fits file or table directory (in the case of the
            Retro-formatted directory with .npy files).

        sd_indices : sd_idx or iterable thereof
            See const.get_sd_idx

        mmap : bool
            Whether to attempt to memory map the table (only applicable for
            Retro npy-files-in-a-dir tables).

        step_length : float > 0, optional
            The stepLength parameter (in meters) used in CLSim tabulator code
            for tabulating a single photon as it travels. This is a hard-coded
            parameter set to 1 meter in the trunk version of the code, but it's
            something we might play with to optimize table generation speed, so
            just be warned that this _can_ change. Note that this is only
            required for CLSim .fits tables, which do not record this
            parameter.

        """
        if self.is_stacked is not None:
            assert not self.is_stacked

        if isinstance(sd_indices, int):
            sd_indices = (sd_indices,)

        sd_indices = sorted(set(sd_indices).intersection(
            self.dom_info[self.dom_info['operational']]['sd_idx']
        ))

        table = self.table_loader_func(fpath=fpath, mmap=mmap)
        if 'step_length' in table:
            if step_length is None:
                step_length = table['step_length']
            else:
                assert step_length == table['step_length']
        else:
            assert step_length is not None
            table['step_length'] = step_length

        self.table_meta = OrderedDict()
        binning = OrderedDict()
        for key, val in table.items():
            if 'bin_edges' not in key:
                continue
            self.table_meta[key] = val
            binning[key] = val

        for k in TABLE_NORM_KEYS:
            self.table_meta[k] = table[k]
        self.table_meta['binning'] = binning
        self.t_is_residual_time = bool(table['t_is_residual_time'])

        self.table_norm, self.t_indep_table_norm = get_table_norm(
            avg_angsens=self.avg_angsens,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: v for k, v in self.table_meta.items() if k in TABLE_NORM_KEYS}
        )

        if self._pexp is None:
            pexp, get_llh, pexp_meta = generate_pexp_5d_function(
                table=table,
                table_kind=self.table_kind,
                t_is_residual_time=self.t_is_residual_time,
                compute_t_indep_exp=self.compute_t_indep_exp,
                use_directionality=self.use_directionality,
                compute_unhit_doms=True, # TODO: modify when TDI works
                num_phi_samples=self.num_phi_samples,
                ckv_sigma_deg=self.ckv_sigma_deg,
                template_library=self.template_library
            )
            self._pexp = pexp
            self._get_llh = get_llh
            self.pexp_meta = pexp_meta

        self.is_stacked = False

        self.tables.append(table[self.table_name])
        self.table_norms.append(self.table_norm)
        self.n_photons_per_table.append(table['n_photons'])

        if self.compute_t_indep_exp:
            t_indep_table = table[self.t_indep_table_name]
            self.t_indep_tables.append(t_indep_table)
            self.t_indep_table_norms.append(self.t_indep_table_norm)

        table_idx = len(self.tables) - 1
        self.sd_idx_table_indexer[sd_indices] = table_idx

        self.loaded_sd_indices = np.sort(np.concatenate([
            self.loaded_sd_indices,
            np.atleast_1d(sd_indices).astype(np.uint32)
        ]))


def get_table_norm(
        n_photons, group_refractive_index, step_length, r_bin_edges,
        costheta_bin_edges, t_bin_edges, quantum_efficiency,
        avg_angsens, norm_version
    ):
    """Get the normalization array to use a raw CLSim table with Retro reco.

    Note that the `norm` array returned is meant to _multiply_ the counts in
    the raw CLSim table to obtain a survival probability.

    Parameters
    ----------
    n_photons : int > 0
        Number of photons thrown in the simulation.

    group_refractive_index : float > 0
        Group refractive index in the medium.

    step_length : float > 0
        Step length used in CLSim tabulator, in units of meters. (Hard-coded to
        1 m in CLSim, but this is a parameter that ultimately could be
        changed.)

    r_bin_edges : 1D array, ascending non-negative values (meters)
        Radial bin edges in units of meters.

    costheta_bin_edges : 1D array, ascending values in [-1, 1]
        Cosine of the zenith angle bin edges; all must be equally spaced.

    t_bin_edges : 1D array, ascending values > 0 (nanoseconds)
        Time bin eges in units of nanoseconds.

    quantum_efficiency : float in (0, 1], optional
        Average DOM quantum efficiency for converting photons to
        photo electrons. Note that any shape to the quantum efficiency should
        already be accounted for by simulating photons according to the
        shape of that distribution. If not specific, defaults to 1.

    avg_angsens : float in (0, 1], optional
        Average DOM angular acceptance sensitivity, which modifies the
        "efficiency" beyond that accounted for by `quantum_efficiency`. Note
        that any shape to the angular accptance sensitivity should already be
        accounted for by simulating photons according to the shape of that
        distribution.

    norm_version : string

    Returns
    -------
    table_norm : shape (n_r_bins, n_t_bins) array, values >= 0
        The normalization is a function of both r- and t-bin (we assume
        costheta binning is "regular"). To obtain a survival probability,
        multiply the value in the CLSim table's bin by the appropriate
        `table_norm` entry. I.e.:
        ``survival_prob = raw_bin_val * table_norm[r_bin_idx, t_bin_idx]``.

    t_indep_table_norm : shape (n_r_bins,) array

    """
    n_costheta_bins = len(costheta_bin_edges) - 1

    r_bin_widths = np.diff(r_bin_edges)
    costheta_bin_widths = np.diff(costheta_bin_edges)
    t_bin_widths = np.diff(t_bin_edges)

    t_bin_range = np.max(t_bin_edges) - np.min(t_bin_edges)

    # We need costheta bins to all have same width for the logic below to hold
    costheta_bin_width = np.mean(costheta_bin_widths)
    assert np.allclose(costheta_bin_widths, costheta_bin_width), costheta_bin_widths

    constant_part = (
        # Number of photons, divided equally among the costheta bins
        1 / (n_photons / n_costheta_bins)

        # Correction for quantum efficiency of the DOM
        * quantum_efficiency

        # Correction for additional loss of sensitivity due to angular
        # acceptance model
        * avg_angsens
    )

    # A photon is tabulated every step_length meters; we want the
    # average probability in each bin, so the count in the bin must be
    # divided by the number of photons in the bin times the number of
    # times each photon is counted.
    speed_of_light_in_medum = ( # units = m/ns
        SPEED_OF_LIGHT_M_PER_NS / group_refractive_index
    )

    # t bin edges are in ns and speed_of_light_in_medum is m/ns
    t_bin_widths_in_m = t_bin_widths * speed_of_light_in_medum

    # TODO: Note that the actual counts will be rounded down (or are one fewer
    # than indicated by this ratio if the ratio comres out to an integer). We
    # don't account for this here, but we _might_ want to if it seems that this
    # will make a difference (e.g. for small bins). Of course if the number
    # comes out to zero, then... should we clip the lower-bound to 1? Go with
    # the fraction we come up with here (since the first step is randomized)?
    # And in fact, since the first step is randomized, it seems this ratio
    # should be fine as is; but it's the 2D toy simulation that indicates that
    # maybe the `floor` might be necessary. Someday we should really sort this
    # whole normalization thing out!
    counts_per_r = r_bin_widths / step_length
    counts_per_t = t_bin_widths_in_m / step_length

    inner_edges = r_bin_edges[:-1]
    outer_edges = r_bin_edges[1:]

    radial_midpoints = 0.5 * (inner_edges + outer_edges)
    #inner_radius = np.where(inner_edges == 0, 0.01* radial_midpoints, inner_edges)
    avg_radius = (
        3/4 * (outer_edges**4 - inner_edges**4)
        / (outer_edges**3 - inner_edges**3)
    )

    surf_area_at_avg_radius = avg_radius**2 * TWO_PI * costheta_bin_width
    #surf_area_at_inner_radius = inner_radius**2 * TWO_PI * costheta_bin_width
    surf_area_at_midpoints = radial_midpoints**2 * TWO_PI * costheta_bin_width

    bin_vols = np.abs(
        spherical_volume(
            rmin=inner_edges,
            rmax=outer_edges,
            dcostheta=-costheta_bin_width,
            dphi=TWO_PI
        )
    )

    # Take the smaller of counts_per_r and counts_per_t
    table_step_length_norm = np.minimum.outer(counts_per_r, counts_per_t) # pylint: disable=no-member
    assert table_step_length_norm.shape == (counts_per_r.size, counts_per_t.size)
    #print('table_step_length_norm')
    #print(table_step_length_norm)
    #raise Exception()

    if norm_version == 'avgsurfarea':
        radial_norm = 1 / surf_area_at_avg_radius
        #radial_norm = 1 / surf_area_at_inner_radius

        # Shape of the radial norm is 1D: (n_r_bins,)
        table_norm = (
            constant_part * table_step_length_norm
            * radial_norm[:, np.newaxis]
        )

    elif norm_version == 'binvol':
        radial_norm = 1 / bin_vols
        table_norm = (
            constant_part * table_step_length_norm * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t) * t_bin_range

    elif norm_version == 'binvol2':
        radial_norm = 1 / bin_vols
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t) * t_bin_range

    elif norm_version == 'binvol3':
        radial_norm = 1 / bin_vols / avg_radius
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t)

    elif norm_version == 'binvol4':
        radial_norm = 1 / bin_vols / avg_radius**0.5
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t)

    elif norm_version == 'binvol5':
        radial_norm = 1 / bin_vols / avg_radius**(1/3)
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t)

    elif norm_version == 'binvol6':
        radial_norm = 1 / bin_vols / avg_radius**(3/4)
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t)

    elif norm_version == 'binvol7':
        radial_norm = 1 / bin_vols / avg_radius**(2/3)
        table_norm = (
            constant_part * counts_per_t * radial_norm[:, np.newaxis]
        )
        t_indep_table_norm = constant_part * radial_norm / np.sum(1/counts_per_t)

    # copied norm from Philipp / generate_t_r_theta_table :
    elif norm_version == 'pde':
        table_norm = np.outer(
            4 * PI / surf_area_at_midpoints,
            np.full(
                shape=(len(t_bin_edges) - 1,),
                fill_value=(
                    1
                    / n_photons
                    / (SPEED_OF_LIGHT_M_PER_NS / group_refractive_index)
                    / np.mean(t_bin_widths)
                    * avg_angsens
                    * quantum_efficiency
                    * n_costheta_bins
                )
            )
        )

    elif norm_version == 'wtf':
        not_really_volumes = surf_area_at_midpoints * (outer_edges - inner_edges)
        not_really_volumes *= costheta_bin_width

        const_bit = (
            1 / 2.451488118071
            / n_photons
            / (SPEED_OF_LIGHT_M_PER_NS / group_refractive_index)
            / np.mean(t_bin_widths)
            * avg_angsens
            * quantum_efficiency
            * n_costheta_bins
        )

        table_norm = np.outer(
            1 / not_really_volumes,
            np.full(shape=(len(t_bin_edges) - 1,), fill_value=(const_bit))
        )
        t_indep_table_norm = const_bit / not_really_volumes / np.sum(1/counts_per_t) * t_bin_range

    elif norm_version == 'wtf2':
        not_really_volumes = surf_area_at_midpoints * (outer_edges - inner_edges)
        not_really_volumes *= costheta_bin_width

        table_norm = np.outer(
            1/(7.677767105803*0.8566215612075752) / not_really_volumes,
            np.full(shape=(len(t_bin_edges) - 1,), fill_value=constant_part)
        )

    else:
        raise ValueError('unhandled `norm_version` "{}"'.format(norm_version))

    return table_norm, t_indep_table_norm
