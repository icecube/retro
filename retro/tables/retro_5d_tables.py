# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for using a set of "raw" 5D (r, costheta, t, costhetadir, deltaphidir)
Retro tables, 5D Cherenkov tables, or template-compressed versions thereof.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    TABLE_NORM_KEYS
    TABLE_KINDS
    NORM_VERSIONS
    Retro5DTables
    get_table_norm
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

from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import SPEED_OF_LIGHT_M_PER_NS, PI, TWO_PI
from retro.tables.pexp_5d import generate_pexp_5d_function
from retro.utils.geom import spherical_volume


TABLE_NORM_KEYS = [
    'n_photons', 'group_refractive_index', 'step_length', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges'
]
"""All besides 'quantum_efficiency' and 'angular_acceptance_fract'"""

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

    """
    def __init__(
            self, table_kind, geom, rde, noise_rate_hz, compute_t_indep_exp,
            use_directionality, norm_version, num_phi_samples=None,
            ckv_sigma_deg=None
        ):
        self.compute_t_indep_exp = compute_t_indep_exp
        self.table_kind = table_kind

        self.tbl_is_raw = table_kind in ['raw_uncompr', 'raw_templ_compr']
        self.tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']
        self.tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']

        if self.tbl_is_raw:
            from retro.tables.clsim_tables import load_clsim_table_minimal
            self.table_loader_func = load_clsim_table_minimal
            # NOTE: original tables have underflow (bin 0) and overflow
            # (bin -1) bins, so whole-axis slices must exclude the first and
            # last bins.
            self.usable_table_slice = (slice(1, -1),)*5
            self.t_indep_table_name = 't_indep_table'
            self.table_name = 'table'
        elif self.tbl_is_ckv:
            from retro.tables.ckv_tables import load_ckv_table
            self.table_loader_func = load_ckv_table
            self.usable_table_slice = (slice(None),)*5
            self.t_indep_table_name = 't_indep_ckv_table'
            self.table_name = 'ckv_table'

        assert len(geom.shape) == 3
        self.geom = geom
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
                "These DOMs will be disabled and return 0's forexpected photon"
                " computations."
                .format(num_zero, num_nan, num_inf)
            )
        mask = zero_mask | nan_mask | inf_mask

        self.operational_doms = ~mask
        self.rde = np.ma.masked_where(mask, rde)
        self.quantum_efficiency = 0.25 * self.rde
        self.noise_rate_hz = np.ma.masked_where(mask, noise_rate_hz)
        self.noise_rate_per_ns = self.noise_rate_hz / 1e9

        self.tables = {}
        self.string_aggregation = None
        self.depth_aggregation = None
        self.pexp_func = None
        self.pexp_meta = None

    def load_table(
            self, fpath, string, dom, mmap, angular_acceptance_fract,
            step_length=None
        ):
        """Load a table into the set of tables.

        Parameters
        ----------
        fpath : string
            Path to the table .fits file or table directory (in the case of the
            Retro-formatted directory with .npy files).

        string : int in [1, 86] or str in {'ic', 'dc', or 'all'}

        dom : int in [1, 60] or str == 'all'

        angular_acceptance_fract : float in (0, 1]
            Constant normalization factor to apply to correct for the integral
            of the angular acceptance curve used in the simulation that
            produced this table.

        mmap : bool
            Whether to attempt to memory map the table (only applicable for
            Retro npy-files-in-a-dir tables).

        step_length : float > 0, optional
            The stepLength parameter (in meters) used in CLSim tabulator code
            for tabulating a single photon as it travels. This is a hard-coded
            paramter set to 1 meter in the trunk version of the code, but it's
            something we might play with to optimize table generation speed, so
            just be warned that this _can_ change. Note that this is only
            required for CLSim .fits tables, which do not record this
            parameter.

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

        assert 0 < angular_acceptance_fract <= 1

        if single_dom_spec and not self.operational_doms[string - 1, dom - 1]:
            print(
                'WARNING: String {}, DOM {} is not operational, skipping'
                ' loading the corresponding table'.format(string, dom)
            )
            return

        table = self.table_loader_func(fpath=fpath, mmap=mmap)
        if 'step_length' in table:
            if step_length is None:
                step_length = table['step_length']
            else:
                assert step_length == table['step_length']
        else:
            assert step_length is not None
            table['step_length'] = step_length

        table_norm, t_indep_table_norm = get_table_norm(
            angular_acceptance_fract=angular_acceptance_fract,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: table[k] for k in TABLE_NORM_KEYS}
        )
        table['table_norm'] = table_norm
        table['t_indep_table_norm'] = t_indep_table_norm

        pexp_5d, pexp_meta = generate_pexp_5d_function(
            table=table,
            table_kind=self.table_kind,
            compute_t_indep_exp=self.compute_t_indep_exp,
            use_directionality=self.use_directionality,
            num_phi_samples=self.num_phi_samples,
            ckv_sigma_deg=self.ckv_sigma_deg
        )
        if self.pexp_func is None:
            self.pexp_func = pexp_5d
            self.pexp_meta = pexp_meta
        elif pexp_meta != self.pexp_meta:
            raise ValueError(
                'All binnings and table parameters currently must be equal to'
                ' one another.'
            )

        table_tup = (
            table[self.table_name][self.usable_table_slice],
            table['table_norm'],
        )

        if self.tbl_is_templ_compr:
            table_tup += (table['table_map'],)

        if self.compute_t_indep_exp:
            table_tup += (
                table[self.t_indep_table_name],
                table['t_indep_table_norm']
            )
            if self.tbl_is_templ_compr:
                table_tup += (table['t_indep_table_map'],)

        self.tables[(string, dom)] = table_tup

    def get_expected_det(
            self, sources, hit_times, string, dom, include_noise=False,
            time_window=None
        ):
        """
        Parameters
        ----------
        sources : shape (num_sources,) array of dtype SRC_DTYPE
            Info about photons generated photons by the event hypothesis.

        hit_times : shape (num_hits,) array of floats, units of ns

        string : int in [1, 86]

        dom : int in [1, 60]

        include_noise : bool
            Include noise in the photon expectations (both at hit time and
            time-independent). Non-operational DOMs return 0 for both return values

        time_window : float in units of ns
            Time window for computing the "time-independent" noise expectation.
            Used (and required) if `include_noise` is True.

        Returns
        -------
        exp_p_at_all_times : float64

        exp_p_at_hit_times : shape (num_hits,) array of float64

        """
        # `string` and `dom` are 1-indexed but array indices are 0-indexed
        string_idx, dom_idx = string - 1, dom - 1
        if not self.operational_doms[string_idx, dom_idx]:
            return np.float64(0.0), np.zeros_like(hit_times, dtype=np.float64)

        dom_coord = self.geom[string_idx, dom_idx]
        dom_quantum_efficiency = self.quantum_efficiency[string_idx, dom_idx]

        if self.string_aggregation == 'all':
            string = 'all'
        elif self.string_aggregation == 'subdetector':
            if string < 79:
                string = 'ic'
            else:
                string = 'dc'

        if self.depth_aggregation:
            dom = 'all'

        table_tup = self.tables[(string, dom)]

        exp_p_at_all_times, exp_p_at_hit_times = self.pexp_func(
            sources,
            hit_times,
            dom_coord,
            dom_quantum_efficiency,
            *table_tup
        )

        if include_noise:
            dom_noise_rate_per_ns = self.noise_rate_per_ns[string_idx, dom_idx]
            exp_p_at_hit_times += dom_noise_rate_per_ns
            exp_p_at_all_times += dom_noise_rate_per_ns * time_window

        return exp_p_at_all_times, exp_p_at_hit_times


def get_table_norm(
        n_photons, group_refractive_index, step_length, r_bin_edges,
        costheta_bin_edges, t_bin_edges, quantum_efficiency,
        angular_acceptance_fract, norm_version
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
        1 m in CLSim, but this is a paramter that ultimately could be changed.)

    r_bin_edges : 1D numpy.ndarray, ascending values => 0 (meters)
        Radial bin edges in units of meters.

    costheta_bin_edges : 1D numpy.ndarray, ascending values in [-1, 1]
        Cosine of the zenith angle bin edges; all must be equally spaced.

    t_bin_edges : 1D numpy.ndarray, ascending values > 0 (nanoseconds)
        Time bin eges in units of nanoseconds.

    quantum_efficiency : float in (0, 1], optional
        Average DOM quantum efficiency for converting photons to
        photo electrons. Note that any shape to the quantum efficiency should
        already be accounted for by simulating photons according to the
        shape of that distribution. If not specific, defaults to 1.

    angular_acceptance_fract : float in (0, 1], optional
        Average DOM angular acceptance fraction, which modifies the
        "efficiency" beyond that accounted for by `quantum_efficiency`.
        Note that any shape to the angular accptance should already be
        accounted for by simulating photons according to the
        shape of that distribution. If not specified, defaults to 1.

    norm_version : string

    Returns
    -------
    table_norm : numpy.ndarray of shape (n_r_bins, n_t_bins), values >= 0
        The normalization is a function of both r- and t-bin (we assume
        costheta binning is "regular"). To obtain a survival probability,
        multiply the value in the CLSim table's bin by the appropriate
        `table_norm` entry. I.e.:
        ``survival_prob = raw_bin_val * table_norm[r_bin_idx, t_bin_idx]``.

    t_indep_table_norm : numpy.ndarray of shape (n_r_bins,)

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
        * angular_acceptance_fract
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
    inner_radius = np.where(inner_edges == 0, 0.01* radial_midpoints, inner_edges)
    avg_radius = (
        3/4 * (outer_edges**4 - inner_edges**4)
        / (outer_edges**3 - inner_edges**3)
    )

    surf_area_at_avg_radius = avg_radius**2 * TWO_PI * costheta_bin_width
    surf_area_at_inner_radius = inner_radius**2 * TWO_PI * costheta_bin_width
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
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            4 * PI / surf_area_at_midpoints,
            np.full(
                shape=(len(t_bin_edges) - 1,),
                fill_value=(
                    1
                    / n_photons
                    / (SPEED_OF_LIGHT_M_PER_NS / group_refractive_index)
                    / np.mean(t_bin_widths)
                    * angular_acceptance_fract
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
            * angular_acceptance_fract
            * quantum_efficiency
            * n_costheta_bins
        )

        table_norm = np.outer(
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            1 / not_really_volumes,
            np.full(shape=(len(t_bin_edges) - 1,), fill_value=(const_bit))
        )
        t_indep_table_norm = const_bit / not_really_volumes / np.sum(1/counts_per_t) * t_bin_range

    elif norm_version == 'wtf2':
        not_really_volumes = surf_area_at_midpoints * (outer_edges - inner_edges)
        not_really_volumes *= costheta_bin_width

        table_norm = np.outer(
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            1/(7.677767105803*0.8566215612075752) / not_really_volumes,
            np.full(shape=(len(t_bin_edges) - 1,), fill_value=constant_part)
        )

    else:
        raise ValueError('unhandled `norm_version` "{}"'.format(norm_version))

    return table_norm, t_indep_table_norm
