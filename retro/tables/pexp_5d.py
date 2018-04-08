# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    MACHINE_EPS
    generate_pexp_5d_function
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
import math
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit
from retro.const import PI, EMPTY_HITS, SRC_OMNI
from retro.utils.ckv import (
    survival_prob_from_cone, survival_prob_from_smeared_cone
)
from retro.utils.geom import infer_power


MACHINE_EPS = 1e-16


def generate_pexp_5d_function(
        table,
        table_kind,
        compute_t_indep_exp,
        compute_unhit_doms,
        use_directionality,
        num_phi_samples=None,
        ckv_sigma_deg=None,
        template_library=None
    ):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function to use.

    Parameters
    ----------
    table : mapping
        As returned by `load_clsim_table_minimal`

    table_kind : str in {'raw_uncompr', 'ckv_uncompr', 'ckv_templ_compr'}

    compute_t_indep_exp : bool
        Whether to use the single-DOM tables for computing time-independent
        expectations. Set to True unless e.g. you use a TDI table to get this
        information.

    compute_unhit_doms : bool
        Whether to compute expecations for unhit DOMs. Set to False e.g. if you
        get this information from a TDI table. If `compute_t_indep_exp` is
        False, this must also be False.

    use_directionality : bool
        If the source photons have directionality, use it in computing photon
        expectations at the DOM.

    num_phi_samples : int
        Number of samples in the phi_dir to average over bin counts.
        (Irrelevant if `use_directionality` is False or if you use a Cherenkov
        table, which already has this parameter integrated into it.)

    ckv_sigma_deg : float
        Standard deviation in degrees for Cherenkov angle. (Irrelevant if
        `use_directionality` is False or if you use a Cherenkov table, which
        already has this parameter integrated into it.)

    template_library : shape-(n_templates, n_dir_theta, n_dir_deltaphi) array
        Containing the directionality templates for compressed tables

    Returns
    -------
    pexp_5d : callable
        Function usable to extract photon expectations from a table of
        `table_kind` and with the binning of `table`. Note that this returns
        two values (photon expectation at hit time and time-independent photon
        expectation) even if `compute_t_indep_exp` is False (whereupon the
        latter number should be ignored.

    meta : OrderedDict
        Paramters, including the binning, that uniquely identify what the
        capabilities of the returned `pexp_5d`. (Use this to eliminate
        redundant pexp_5d functions.)

    """
    tbl_is_raw = table_kind in ['raw_uncompr', 'raw_templ_compr']
    tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']
    assert tbl_is_raw or tbl_is_ckv

    if not compute_t_indep_exp:
        assert not compute_unhit_doms

    meta = OrderedDict(
        table_kind=table_kind,
        compute_t_indep_exp=compute_t_indep_exp,
        use_directionality=use_directionality,
        num_phi_samples=None if tbl_is_ckv or not use_directionality else num_phi_samples,
        ckv_sigma_deg=None if tbl_is_ckv or not use_directionality else ckv_sigma_deg,
    )

    if num_phi_samples is None:
        num_phi_samples = 0
    if ckv_sigma_deg is None:
        ckv_sigma_deg = 0

    r_min = np.min(table['r_bin_edges'])

    # Ensure r_min is zero; this removes need for lower-bound checks and a
    # subtraction each time computing bin index
    assert r_min == 0

    r_max = np.max(table['r_bin_edges'])
    rsquared_max = r_max*r_max
    r_power = infer_power(table['r_bin_edges'])
    assert r_power == 2
    inv_r_power = 1 / r_power
    n_r_bins = len(table['r_bin_edges']) - 1
    table_dr_pwr = (r_max - r_min)**inv_r_power / n_r_bins

    n_costheta_bins = len(table['costheta_bin_edges']) - 1
    table_dcostheta = 2 / n_costheta_bins

    t_min = np.min(table['t_bin_edges'])

    # Ensure t_min is zero; this removes need for lower-bound checks and a
    # subtraction each time computing bin index
    assert t_min == 0

    t_max = np.max(table['t_bin_edges'])
    n_t_bins = len(table['t_bin_edges']) - 1
    table_dt = (t_max - t_min) / n_t_bins

    assert table['costhetadir_bin_edges'][0] == -1
    assert table['costhetadir_bin_edges'][-1] == 1
    n_costhetadir_bins = len(table['costhetadir_bin_edges']) - 1
    table_dcosthetadir = 2 / n_costhetadir_bins
    assert np.allclose(np.diff(table['costhetadir_bin_edges']), table_dcosthetadir)
    last_costhetadir_bin_idx = n_costhetadir_bins - 1

    assert table['deltaphidir_bin_edges'][0] == 0
    assert np.isclose(table['deltaphidir_bin_edges'][-1], PI)
    n_deltaphidir_bins = len(table['deltaphidir_bin_edges']) - 1
    table_dphidir = PI / n_deltaphidir_bins
    assert np.allclose(np.diff(table['deltaphidir_bin_edges']), table_dphidir)
    last_deltaphidir_bin_idx = n_deltaphidir_bins - 1

    binning_info = dict(
        r_min=r_min, r_max=r_max, n_r_bins=n_r_bins, r_power=r_power,
        n_costheta_bins=n_costheta_bins,
        t_min=t_min, t_max=t_max, n_t_bins=n_t_bins,
        n_costhetadir_bins=n_costhetadir_bins,
        n_deltaphidir_bins=n_deltaphidir_bins,
        deltaphidir_one_sided=True
    )
    meta['binning_info'] = binning_info

    random_delta_thetas = np.array([])
    if tbl_is_raw and use_directionality and ckv_sigma_deg > 0:
        rand = np.random.RandomState(0)
        random_delta_thetas = rand.normal(
            loc=0,
            scale=np.deg2rad(ckv_sigma_deg),
            size=num_phi_samples
        )

    empty_1d_array = np.array([], dtype=np.float64).reshape((0,))
    empty_4d_array = np.array([], dtype=np.float64).reshape((0,)*4)

    if tbl_is_templ_compr:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx):
            """Helper function for directionality-averaged table lookup"""
            # Original axes ordering
            templ = table[r_bin_idx, costheta_bin_idx, t_bin_idx]

            # Reordered axes (_should_ be faster, but... alas, didn't seem to be)
            #templ = table[costheta_bin_idx, r_bin_idx, t_bin_idx]

            return templ['weight'] / template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(table, r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx):
            """Helper function for table lookup"""
            # Original axes ordering
            templ = table[r_bin_idx, costheta_bin_idx, t_bin_idx]

            # Reordered axes (_should_ be faster, but... alas, didn't seem to be)
            #templ = table[costheta_bin_idx, r_bin_idx, t_bin_idx]

            return (
                templ['weight']
                * template_library[templ['index'], costhetadir_bin_idx, deltaphidir_bin_idx]
            )

    else:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx):
            """Helper function for directionality averaged table lookup"""
            return np.mean(table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :])

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(table, r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx):
            """Helper function for table lookup"""
            return table[r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx]

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d_generic( # pylint: disable=missing-docstring, too-many-locals
            sources,
            hits,
            dom_info,
            time_window,
            table,
            table_norm,
            t_indep_table=empty_4d_array,
            t_indep_table_norm=empty_1d_array,
        ):
        r"""For a set of generated photons `sources`, compute the expected
        photons in a particular DOM at `hit_time` and the total expected
        photons, independent of time.

        This function utilizes the relative space-time coordinates _and_
        directionality of the generated photons (via "raw" 5D CLSim tables) to
        determine how many photons are expected to arrive at the DOM.

        Retro DOM tables applied to the generated photon info `sources`,
        and the total expected photon count (time integrated) -- the
        normalization of the pdf.

        Parameters
        ----------
        sources : shape (num_sources,) array of dtype SRC_T
            A discrete sequence of points describing expected sources of
            photons that result from a hypothesized event.

        hit_times : shape (num_hits,) array of dtype float64, units of ns
            Time at which the DOM recorded a hit (or multiple simultaneous
            hits). Use np.nan to indicate no hit occurred.

        dom_coord : shape (3,) array
            DOM (x, y, z) coordinate in meters (in terms of the IceCube
            coordinate system).

        quantum_efficiency : float in (0, 1]
            Scale factor that reduces detected photons due to average quantum
            efficiency of the DOM.

        table : array
            Time-dependent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir)
            while if you use a template-compressed table, this will have shape
                (n_templates, n_costhetadir, n_deltaphidir)

        table_norm : shape (n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions and therefore is an array.

        t_indep_table : array, optional
            Time-independent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_costhetadir, n_deltaphidir)
            while if using a

        t_indep_table_norm : array, optional
            r-dependent normalization (any t-dep normalization is assumed to
            already have been applied to generate the t_indep_table).

        Returns
        -------
        t_indep_exp : float64
            If `compute_t_indep_exp` is True, return the total photons due to
            the hypothesis expected to arrive at the specified DOM for _all_
            times. If `compute_t_indep_exp` is False, return value is 0.

        sum_log_exp_at_hit_times : float64
            .. math::

                \Sum_i q_i \log( \rm{QE} \bar N_{\gamma, i} + \rm{noise} )

            $q_i$ is the hit multiplicity at the hit time, $t_i$, QE is quantum
            efficiency of the DOM, $\bar N_{\gamma, i}$ is the expected number
            of photons at the DOM at $t_i$, and noise is the noise _rate_ in
            the DOM.

        """
        if not dom_info['operational']:
            return np.float64(0), np.float64(0)

        num_hits = len(hits)

        # Initialize accumulators (use double precision, as accumulation
        # compounds finite-precision errors)
        t_indep_exp = np.float64(0)
        exp_at_hit_times = np.zeros(num_hits, dtype=np.float64)

        # Extract the components of the DOM coordinate just once, here
        dom_x = dom_info['x']
        dom_y = dom_info['y']
        dom_z = dom_info['z']

        for source in sources:
            dx = dom_x - source['x']
            dy = dom_y - source['y']
            dz = dom_z - source['z']

            rhosquared = dx*dx + dy*dy
            rsquared = rhosquared + dz*dz

            # Continue if photon is outside the radial binning limits
            if rsquared >= rsquared_max:
                continue

            r = math.sqrt(rsquared)
            r_bin_idx = int(math.sqrt(r) / table_dr_pwr)
            costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

            source_kind = source['kind']

            if compute_t_indep_exp and source_kind == SRC_OMNI:
                t_indep_surv_prob = np.mean(
                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                )

            else: # (not compute_t_indep_exp) or (source_kind == SRC_CKV_BETA1):
                # Note that for these tables, we have to invert the photon
                # direction relative to the vector from the DOM to the photon's
                # vertex since simulation has photons going _away_ from the DOM
                # that in reconstruction will hit the DOM if they're moving
                # _towards_ the DOM.

                # Zenith angle is indep. of photon position relative to DOM
                pdir_costheta = source['dir_costheta']

                rho = math.sqrt(rhosquared)

                # \Delta\phi depends on photon position relative to the DOM...

                # Below is the projection of pdir into the (x, y) plane and the
                # projection of that onto the vector in that plane connecting
                # the photon source to the DOM. We get the cosine of the angle
                # between these vectors by solving the identity
                #   `a dot b = |a| |b| cos(deltaphi)`
                # for cos(deltaphi), where the `a` and `b` vectors are the
                # projections of the aforementioned vectors onto the xy-plane.

                if rho <= MACHINE_EPS:
                    pdir_cosdeltaphi = np.float64(1)
                    pdir_sindeltaphi = np.float64(0)
                else:
                    pdir_cosdeltaphi = (
                        source['dir_cosphi'] * dx/rho + source['dir_sinphi'] * dy/rho
                    )
                    # Note that the max and min here here in case numerical
                    # precision issues cause the dot product to blow up.
                    pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))
                    if tbl_is_raw:
                        pdir_sindeltaphi = math.sqrt(1 - pdir_cosdeltaphi*pdir_cosdeltaphi)

                if tbl_is_raw:
                    pdir_sintheta = source['dir_sintheta']

                    # Cherenkov angle is encoded as the projection of a
                    # length-1 vector going in the Ckv direction onto the
                    # charged particle's direction. Ergo, in the length of the
                    # pdir vector is the cosine of the ckv angle.
                    ckv_costheta = source['ckv_costheta']
                    ckv_theta = source['ckv_theta']

                    if ckv_sigma_deg > 0:
                        if compute_t_indep_exp:
                            t_indep_surv_prob, _a, _b = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                                theta=ckv_theta,
                                num_phi=num_phi_samples,
                                rot_costheta=pdir_costheta,
                                rot_sintheta=pdir_sintheta,
                                rot_cosphi=pdir_cosdeltaphi,
                                rot_sinphi=pdir_sindeltaphi,
                                directional_survival_prob=(
                                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                                ),
                                num_costheta_bins=n_costhetadir_bins,
                                num_deltaphi_bins=n_deltaphidir_bins,
                                random_delta_thetas=random_delta_thetas
                            )
                    else:
                        ckv_sintheta = source['ckv_sintheta']
                        if compute_t_indep_exp:
                            t_indep_surv_prob, _a, _b = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                                costheta=ckv_costheta,
                                sintheta=ckv_sintheta,
                                num_phi=num_phi_samples,
                                rot_costheta=pdir_costheta,
                                rot_sintheta=pdir_sintheta,
                                rot_cosphi=pdir_cosdeltaphi,
                                rot_sinphi=pdir_sindeltaphi,
                                directional_survival_prob=(
                                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                                ),
                                num_costheta_bins=n_costhetadir_bins,
                                num_deltaphi_bins=n_deltaphidir_bins,
                            )

                else: # tbl_is_ckv
                    costhetadir_bin_idx = int((pdir_costheta + np.float64(1)) / table_dcosthetadir)

                    # Make upper edge inclusive
                    if costhetadir_bin_idx > last_costhetadir_bin_idx:
                        costhetadir_bin_idx = last_costhetadir_bin_idx

                    pdir_deltaphi = math.acos(pdir_cosdeltaphi)
                    deltaphidir_bin_idx = int(abs(pdir_deltaphi) / table_dphidir)

                    # Make upper edge inclusive
                    if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                        deltaphidir_bin_idx = last_deltaphidir_bin_idx

                    t_indep_surv_prob = t_indep_table[
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]

            #else:
            #    raise NotImplementedError('Source kind not implmented')

            source_photons = source['photons']

            if compute_t_indep_exp:
                ti_norm = t_indep_table_norm[r_bin_idx]
                t_indep_exp += (
                    source_photons * ti_norm * t_indep_surv_prob
                )

            for hit_t_idx in range(num_hits):
                hit_time = hits[hit_t_idx]['time']

                # Causally impossible? (Note the comparison is written such that it
                # will evaluate to True if hit_time is NaN.)
                source_t = source['time']
                if not source_t <= hit_time:
                    continue

                # A photon that starts immediately in the past (before the DOM
                # was hit) will show up in the Retro DOM tables in bin 0; the
                # further in the past the photon started, the higher the time
                # bin index. Therefore, subract source time from hit time.
                dt = hit_time - source_t

                # Is relative time outside binning?
                if dt >= t_max:
                    continue

                t_bin_idx = int(dt / table_dt)

                r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]

                if source_kind == SRC_OMNI:
                    surv_prob_at_hit_t = table_lookup_mean(
                        table, r_bin_idx, costheta_bin_idx, t_bin_idx
                    )

                else: #elif source_kind == SRC_CKV_BETA1:
                    # TODO: get this working again
                    #if tbl_is_raw and not tbl_is_templ_compr:
                    #    if ckv_sigma_deg > 0:
                    #        surv_prob_at_hit_t, _c, _d = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                    #            theta=ckv_theta,
                    #            num_phi=num_phi_samples,
                    #            rot_costheta=pdir_costheta,
                    #            rot_sintheta=pdir_sintheta,
                    #            rot_cosphi=pdir_cosdeltaphi,
                    #            rot_sinphi=pdir_sindeltaphi,
                    #            directional_survival_prob=(
                    #                table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                    #            ),
                    #            num_costheta_bins=n_costhetadir_bins,
                    #            num_deltaphi_bins=n_deltaphidir_bins,
                    #            random_delta_thetas=random_delta_thetas
                    #        )
                    #    else:
                    #        surv_prob_at_hit_t, _c, _d = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                    #            costheta=ckv_costheta,
                    #            sintheta=ckv_sintheta,
                    #            num_phi=num_phi_samples,
                    #            rot_costheta=pdir_costheta,
                    #            rot_sintheta=pdir_sintheta,
                    #            rot_cosphi=pdir_cosdeltaphi,
                    #            rot_sinphi=pdir_sindeltaphi,
                    #            directional_survival_prob=(
                    #                table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                    #            ),
                    #            num_costheta_bins=n_costhetadir_bins,
                    #            num_deltaphi_bins=n_deltaphidir_bins,
                    #        )
                    #
                    #else: # tbl_is_ckv
                    surv_prob_at_hit_t = table_lookup(
                        table,
                        r_bin_idx,
                        costheta_bin_idx,
                        t_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    )

                #else:
                #    raise NotImplementedError('Source kind not implemented')

                exp_at_hit_times[hit_t_idx] += source_photons * r_t_bin_norm * surv_prob_at_hit_t

        quantum_efficiency = dom_info['quantum_efficiency']
        noise_rate_per_ns = dom_info['noise_rate_per_ns']

        sum_log_exp_at_hit_times = np.float64(0)
        for hit_idx in range(num_hits):
            exp_at_hit_time = exp_at_hit_times[hit_idx]
            hit_mult = hits[hit_idx]['charge']
            sum_log_exp_at_hit_times += (
                hit_mult * math.log(quantum_efficiency * exp_at_hit_time + noise_rate_per_ns)
            )

        if compute_t_indep_exp:
            t_indep_exp = (
                quantum_efficiency * t_indep_exp + noise_rate_per_ns * time_window
            )

        return t_indep_exp, sum_log_exp_at_hit_times

    # DEBUG
    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d_ckv_templ_compr_compute_t_indep_noop(
            sources,
            hits,
            dom_info,
            time_window,
            table,
            table_norm,
            t_indep_table,
            t_indep_table_norm,
            # DEBUG: add two args
            #table_indices,
            #t_indep_indices
        ):
        """Use instead of pexp_5d_ckv_compute_t_indep for debugging speed"""
        # pylint: disable=unused-variable, unused-argument, line-too-long
        if not dom_info['operational']:
            return np.float64(0), np.float64(0)

        num_hits = len(hits)

        # Initialize accumulators (use double precision, as accumulation
        # compounds finite-precision errors)
        t_indep_exp = np.float64(0)
        exp_at_hit_times = np.zeros(num_hits, dtype=np.float64)

        # Extract the components of the DOM coordinate just once, here
        dom_x = dom_info['x']
        dom_y = dom_info['y']
        dom_z = dom_info['z']

        for source in sources:
            dx = dom_x - source['x']
            dy = dom_y - source['y']
            dz = dom_z - source['z']

            rhosquared = dx*dx + dy*dy
            rsquared = rhosquared + dz*dz

            # Continue if photon is outside the radial binning limits
            if rsquared >= rsquared_max:
                continue

            r = math.sqrt(rsquared)
            r_bin_idx = int(math.sqrt(r) / table_dr_pwr)
            costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

            source_kind = source['kind']

            if source_kind == SRC_OMNI:
                #t_indep_indices.append((r_bin_idx, costheta_bin_idx))
                t_indep_surv_prob = 1.0 / np.float64(r_bin_idx + costheta_bin_idx) #np.mean(
                #    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                #)

            else: #elif source_kind == SRC_CKV_BETA1:
                # Note that for these tables, we have to invert the photon
                # direction relative to the vector from the DOM to the photon's
                # vertex since simulation has photons going _away_ from the DOM
                # that in reconstruction will hit the DOM if they're moving
                # _towards_ the DOM.

                # Zenith angle is indep. of photon position relative to DOM
                pdir_costheta = source['dir_costheta']

                rho = math.sqrt(rhosquared)

                # \Delta\phi depends on photon position relative to the DOM...

                # Below is the projection of pdir into the (x, y) plane and the
                # projection of that onto the vector in that plane connecting
                # the photon source to the DOM. We get the cosine of the angle
                # between these vectors by solving the identity
                #   `a dot b = |a| |b| cos(deltaphi)`
                # for cos(deltaphi), where the `a` and `b` vectors are the
                # projections of the aforementioned vectors onto the xy-plane.

                if rho <= MACHINE_EPS:
                    pdir_cosdeltaphi = np.float64(1)
                else:
                    pdir_cosdeltaphi = (
                        source['dir_cosphi'] * dx/rho + source['dir_sinphi'] * dy/rho
                    )
                    # Note that the max and min here here in case numerical
                    # precision issues cause the dot product to blow up.
                    pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))

                costhetadir_bin_idx = int((pdir_costheta + np.float64(1)) / table_dcosthetadir)

                # Make upper edge inclusive
                if costhetadir_bin_idx > last_costhetadir_bin_idx:
                    costhetadir_bin_idx = last_costhetadir_bin_idx

                pdir_deltaphi = math.acos(pdir_cosdeltaphi)
                deltaphidir_bin_idx = int(abs(pdir_deltaphi) / table_dphidir)

                # Make upper edge inclusive
                if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                    deltaphidir_bin_idx = last_deltaphidir_bin_idx

                #t_indep_indices.append((r_bin_idx, costheta_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx))
                t_indep_surv_prob = 1.0 / np.float64(r_bin_idx + costheta_bin_idx + costhetadir_bin_idx + deltaphidir_bin_idx)
                t_indep_surv_prob = 1 #t_indep_table[
                #    r_bin_idx,
                #    costheta_bin_idx,
                #    costhetadir_bin_idx,
                #    deltaphidir_bin_idx
                #]

            source_photons = source['photons']

            #ti_norm = t_indep_table_norm[r_bin_idx]
            #t_indep_exp += (
            #    source_photons * ti_norm * t_indep_surv_prob
            #)
            ti_norm = t_indep_table_norm[r_bin_idx]
            t_indep_exp += (
                source_photons * t_indep_surv_prob
            )

            for hit_t_idx in range(num_hits):
                hit_time = hits[hit_t_idx]['time']

                # Causally impossible? (Note the comparison is written such that it
                # will evaluate to True if hit_time is NaN.)
                source_t = source['time']
                if not source_t <= hit_time:
                    continue

                # A photon that starts immediately in the past (before the DOM
                # was hit) will show up in the Retro DOM tables in bin 0; the
                # further in the past the photon started, the higher the time
                # bin index. Therefore, subract source time from hit time.
                dt = hit_time - source_t

                # Is relative time outside binning?
                if dt >= t_max:
                    continue

                t_bin_idx = int(dt / table_dt)

                if source_kind == SRC_OMNI:
                    # DEBUG
                    #table_indices.append((r_bin_idx, costheta_bin_idx, t_bin_idx))
                    surv_prob_at_hit_t = 1.0 / np.float64(r_bin_idx + costheta_bin_idx + t_bin_idx) #table_lookup_mean(
                    #    table, r_bin_idx, costheta_bin_idx, t_bin_idx
                    #)

                else: #elif source_kind == SRC_CKV_BETA1:
                    # DEBUG
                    #table_indices.append((r_bin_idx, costheta_bin_idx, t_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx))
                    surv_prob_at_hit_t = 1.0 / np.float64(r_bin_idx + costheta_bin_idx + t_bin_idx + costhetadir_bin_idx + deltaphidir_bin_idx) #table_lookup(
                    #    table,
                    #    r_bin_idx,
                    #    costheta_bin_idx,
                    #    t_bin_idx,
                    #    costhetadir_bin_idx,
                    #    deltaphidir_bin_idx
                    #)

                #r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]
                #exp_at_hit_times[hit_t_idx] += source_photons * r_t_bin_norm * surv_prob_at_hit_t
                exp_at_hit_times[hit_t_idx] += source_photons * surv_prob_at_hit_t

        quantum_efficiency = dom_info['quantum_efficiency']
        noise_rate_per_ns = dom_info['noise_rate_per_ns']

        sum_log_exp_at_hit_times = np.float64(0)
        for hit_idx in range(num_hits):
            exp_at_hit_time = exp_at_hit_times[hit_idx]
            hit_mult = hits[hit_idx]['charge']
            sum_log_exp_at_hit_times += (
                hit_mult * math.log(quantum_efficiency * exp_at_hit_time + noise_rate_per_ns)
            )

        t_indep_exp = (
            quantum_efficiency * t_indep_exp + noise_rate_per_ns * time_window
        )

        return t_indep_exp, sum_log_exp_at_hit_times

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d_ckv_compute_t_indep(
            sources,
            hits,
            dom_info,
            time_window,
            table,
            table_norm,
            t_indep_table,
            t_indep_table_norm,
        ):
        r"""For a set of generated photons `sources`, compute the expected
        photons in a particular DOM at `hit_time` and the total expected
        photons, independent of time.

        This function utilizes the relative space-time coordinates _and_
        directionality of the generated photons (via "raw" 5D CLSim tables) to
        determine how many photons are expected to arrive at the DOM.

        Retro DOM tables applied to the generated photon info `sources`,
        and the total expected photon count (time integrated) -- the
        normalization of the pdf.

        Parameters
        ----------
        sources : shape (num_sources,) array of dtype SRC_T
            A discrete sequence of points describing expected sources of
            photons that result from a hypothesized event.

        hits

        dom_info

        table : array
            Time-dependent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir)
            while if you use a template-compressed table, this will have shape
                (n_templates, n_costhetadir, n_deltaphidir)

        table_norm : shape (n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions and therefore is an array.

        t_indep_table : array, optional
            Time-independent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_costhetadir, n_deltaphidir)
            while if using a

        t_indep_table_norm : array, optional
            r-dependent normalization (any t-dep normalization is assumed to
            already have been applied to generate the t_indep_table).

        Returns
        -------
        t_indep_exp : float64
            If `compute_t_indep_exp` is True, return the total photons due to
            the hypothesis expected to arrive at the specified DOM for _all_
            times. If `compute_t_indep_exp` is False, return value is 0.

        sum_log_exp_at_hit_times : float64
            .. math::

                \Sum_i q_i \log( \rm{QE} \bar N_{\gamma, i} + \rm{noise} )

            $q_i$ is the hit multiplicity at the hit time, $t_i$, QE is quantum
            efficiency of the DOM, $\bar N_{\gamma, i}$ is the expected number
            of photons at the DOM at $t_i$, and noise is the noise _rate_ in
            the DOM.

        """
        if not dom_info['operational']:
            return np.float64(0), np.float64(0)

        num_hits = len(hits)

        # Initialize accumulators (use double precision, as accumulation
        # compounds finite-precision errors)
        t_indep_exp = np.float64(0)
        exp_at_hit_times = np.zeros(num_hits, dtype=np.float64)

        # Extract the components of the DOM coordinate just once, here
        dom_x = dom_info['x']
        dom_y = dom_info['y']
        dom_z = dom_info['z']
        quantum_efficiency = dom_info['quantum_efficiency']
        noise_rate_per_ns = dom_info['noise_rate_per_ns']

        for source in sources:
            dx = dom_x - source['x']
            dy = dom_y - source['y']
            dz = dom_z - source['z']

            rhosquared = dx*dx + dy*dy
            rsquared = rhosquared + dz*dz

            # Continue if photon is outside the radial binning limits
            if rsquared >= rsquared_max:
                continue

            r = math.sqrt(rsquared)
            r_bin_idx = int(math.sqrt(r) / table_dr_pwr)
            costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

            source_kind = source['kind']

            if source_kind == SRC_OMNI:
                # Original axes ordering
                t_indep_surv_prob = np.mean(
                    t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                )
                # Reordered axes (_should_ be faster, but... alas, didn't seem to be)
                #t_indep_surv_prob = np.mean(
                #    t_indep_table[:, costheta_bin_idx, r_bin_idx, :]
                #)

            else: # source_kind == SRC_CKV_BETA1:
                # Note that for these tables, we have to invert the photon
                # direction relative to the vector from the DOM to the photon's
                # vertex since simulation has photons going _away_ from the DOM
                # that in reconstruction will hit the DOM if they're moving
                # _towards_ the DOM.

                # Zenith angle is indep. of photon position relative to DOM
                pdir_costheta = source['dir_costheta']

                rho = math.sqrt(rhosquared)

                # \Delta\phi depends on photon position relative to the DOM...

                # Below is the projection of pdir into the (x, y) plane and the
                # projection of that onto the vector in that plane connecting
                # the photon source to the DOM. We get the cosine of the angle
                # between these vectors by solving the identity
                #   `a dot b = |a| |b| cos(deltaphi)`
                # for cos(deltaphi), where the `a` and `b` vectors are the
                # projections of the aforementioned vectors onto the xy-plane.

                if rho <= MACHINE_EPS:
                    pdir_cosdeltaphi = np.float64(1)
                else:
                    pdir_cosdeltaphi = (
                        source['dir_cosphi'] * dx/rho + source['dir_sinphi'] * dy/rho
                    )
                    # Note that the max and min here here in case numerical
                    # precision issues cause the dot product to blow up.
                    pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))

                costhetadir_bin_idx = int((pdir_costheta + np.float64(1)) / table_dcosthetadir)

                # Make upper edge inclusive
                if costhetadir_bin_idx > last_costhetadir_bin_idx:
                    costhetadir_bin_idx = last_costhetadir_bin_idx

                pdir_deltaphi = math.acos(pdir_cosdeltaphi)
                deltaphidir_bin_idx = int(abs(pdir_deltaphi) / table_dphidir)

                # Make upper edge inclusive
                if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                    deltaphidir_bin_idx = last_deltaphidir_bin_idx

                # Original axes ordering
                t_indep_surv_prob = t_indep_table[
                    r_bin_idx,
                    costheta_bin_idx,
                    costhetadir_bin_idx,
                    deltaphidir_bin_idx
                ]

                # Reordered axes (_should_ be faster, but... alas, didn't seem to be)
                #t_indep_surv_prob = t_indep_table[
                #    costhetadir_bin_idx,
                #    costheta_bin_idx,
                #    r_bin_idx,
                #    deltaphidir_bin_idx
                #]

            source_photons = source['photons']

            ti_norm = t_indep_table_norm[r_bin_idx]
            t_indep_exp += (
                source_photons * ti_norm * t_indep_surv_prob
            )

            for hit_t_idx in range(num_hits):
                hit_time = hits[hit_t_idx]['time']

                # Causally impossible? (Note the comparison is written such that it
                # will evaluate to True if hit_time is NaN.)
                source_t = source['time']
                if not source_t <= hit_time:
                    continue

                # A photon that starts immediately in the past (before the DOM
                # was hit) will show up in the Retro DOM tables in bin 0; the
                # further in the past the photon started, the higher the time
                # bin index. Therefore, subract source time from hit time.
                dt = hit_time - source_t

                # Is relative time outside binning?
                if dt >= t_max:
                    continue

                t_bin_idx = int(dt / table_dt)

                if source_kind == SRC_OMNI:
                    surv_prob_at_hit_t = table_lookup_mean(
                        table, r_bin_idx, costheta_bin_idx, t_bin_idx
                    )

                else: # source_kind == SRC_CKV_BETA1
                    surv_prob_at_hit_t = table_lookup(
                        table,
                        r_bin_idx,
                        costheta_bin_idx,
                        t_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    )

                r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]
                exp_at_hit_times[hit_t_idx] += (
                    source_photons * r_t_bin_norm * surv_prob_at_hit_t
                )

        sum_log_exp_at_hit_times = np.float64(0)
        for hit_idx in range(num_hits):
            exp_at_hit_time = exp_at_hit_times[hit_idx]
            hit_mult = hits[hit_idx]['charge']
            sum_log_exp_at_hit_times += (
                hit_mult * math.log(quantum_efficiency * exp_at_hit_time + noise_rate_per_ns)
            )

        t_indep_exp = (
            quantum_efficiency * t_indep_exp + noise_rate_per_ns * time_window
        )

        return t_indep_exp, sum_log_exp_at_hit_times

    if tbl_is_ckv and compute_t_indep_exp:
        pexp_5d = pexp_5d_ckv_compute_t_indep
    else:
        pexp_5d = pexp_5d_generic

    # DEBUG
    #pexp_5d = pexp_5d_ckv_templ_compr_compute_t_indep_noop

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh_all_doms(
            sources,
            hits,
            hits_indexer,
            unhit_sd_indices,
            sd_idx_table_indexer,
            time_window,
            dom_info,
            tables,
            table_norm,
            t_indep_tables,
            t_indep_table_norm,
        ):
        """Compute log likelihood for hypothesis sources given an event.

        This version of get_llh is specialized to compute log-likelihoods for
        all DOMs, whether or not they were hit. Use this if you aren't already
        using a TDI table to get the likelihood term corresponding to
        time-independent expectation.

        Parameters
        ----------
        sources : shape (n_sources,) array of dtype SRC_T
        hits : shape (n_hits_total,) array of dtype HIT_T
        hits_indexer : shape (n_hit_doms,) array of dtype SD_INDEXER_T
        unhit_sd_indices : shape (n_unhit_doms,) array of dtype uint32
        sd_idx_table_indexer : shape (n_doms_tot,) array of dtype uint32
        time_window : float64
        dom_info : shape (n_strings, n_doms_per_string) array of dtype DOM_INFO_T
        tables
            Stacked tables
        table_norm
            Single norm for all stacked tables
        t_indep_tables
            Stacked time-independent tables
        t_indep_table_norm
            Single norm for all stacked time-independent tables

        """
        llh = np.float64(0)

        # Loop through all DOMs we know didn't receive hits
        for sd_idx1 in unhit_sd_indices:
            table_idx = sd_idx_table_indexer[sd_idx1]
            t_indep_exp, sum_log_exp_at_hit_times = pexp_5d(
                sources=sources,
                hits=EMPTY_HITS,
                dom_info=dom_info[sd_idx1],
                time_window=time_window,
                table=tables[table_idx],
                table_norm=table_norm,
                t_indep_table=t_indep_tables[table_idx],
                t_indep_table_norm=t_indep_table_norm,
            )
            llh += sum_log_exp_at_hit_times - t_indep_exp

        # Loop through all DOMs that are in the sd_idx range where hits
        # occurred, checking each for whether or not it was hit. We assume that
        # the DOMs in the hits_indexer are sorted in ascending sd_idx order to
        # decrease the amount of looping necessary.
        for indexer_entry in hits_indexer:
            sd_idx2 = indexer_entry['sd_idx']
            start = indexer_entry['offset']
            stop = start + indexer_entry['num']
            table_idx = sd_idx_table_indexer[sd_idx2]
            t_indep_exp, sum_log_exp_at_hit_times = pexp_5d(
                sources=sources,
                hits=hits[start:stop],
                dom_info=dom_info[sd_idx2],
                time_window=time_window,
                table=tables[table_idx],
                table_norm=table_norm,
                t_indep_table=t_indep_tables[table_idx],
                t_indep_table_norm=t_indep_table_norm,
            )
            llh += sum_log_exp_at_hit_times - t_indep_exp
        return llh

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh_hit_doms(
            sources,
            hits,
            hits_indexer,
            time_window,
            dom_info,
            tables,
            table_norm,
            t_indep_tables,
            t_indep_table_norm,
            sd_idx_table_indexer
        ):
        """Compute log likelihood for hypothesis sources given an event.

        This version of get_llh is specialized to compute log-likelihoods only
        for DOMs that were hit. Use this e.g. if you use a TDI table to get the
        other parts of the likelihood function.

        Parameters
        ----------
        sources : shape (n_sources,) array of dtype SRC_T
        hits : shape (n_hits_total,) array of dtype HIT_T
        hits_indexer : shape (n_hit_doms,) array of dtype SD_INDEXER_T
        time_window : float64
        dom_info : shape (n_strings, n_doms_per_string) array of dtype DOM_INFO_T
        tables
        table_norm
        t_indep_tables
        t_indep_table_norm
        sd_idx_table_indexer

        """
        llh = np.float64(0)
        for indexer_entry in hits_indexer:
            start = indexer_entry['offset']
            stop = start + indexer_entry['num']
            sd_idx = indexer_entry['sd_idx']
            table_idx = sd_idx_table_indexer[sd_idx]
            t_indep_exp, sum_log_exp_at_hit_times = pexp_5d(
                sources=sources,
                hits=hits[start:stop],
                dom_info=dom_info[sd_idx],
                time_window=time_window,
                table=tables[table_idx],
                table_norm=table_norm,
                t_indep_table=t_indep_tables[table_idx],
                t_indep_table_norm=t_indep_table_norm
            )
            llh += sum_log_exp_at_hit_times - t_indep_exp
        return llh

    if compute_t_indep_exp:
        get_llh = get_llh_all_doms
    else:
        get_llh = get_llh_hit_doms

    return pexp_5d, get_llh, meta
