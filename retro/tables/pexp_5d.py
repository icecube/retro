# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['MACHINE_EPS',
           'generate_pexp_5d_function',
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
from retro.retro_types import EXP_DOM_T


MACHINE_EPS = 1e-16
LLH_VERSION = 2


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
        def table_lookup_mean(tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx):
            """Helper function for directionality-averaged table lookup"""
            # Original axes ordering
            templ = tables[table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx]

            # Reordered axes (_should_ be faster, but... alas, didn't seem to be)
            #templ = table[costheta_bin_idx, r_bin_idx, t_bin_idx]

            return templ['weight'] / template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx):
            """Helper function for table lookup"""
            # Original axes ordering
            templ = tables[table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx]

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
    def pexp_5d_ckv_compute_t_indep(
            sources,
            sources_start,
            sources_stop,
            hits,
            event_dom_info,
            time_window,
            tables,
            table_norm,
            t_indep_tables,
            t_indep_table_norm,
            dom_exp,
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

        sources_start, sources_stop : int, int
            starting and stopping index for part of the array on which to work on

        hits

        event_dom_info

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

        Out:
        ----

        dom_exp :  array containing expectations

        """

        for dom_idx in range(len(event_dom_info)):
            hits_start = event_dom_info['hits_start_idx'][dom_idx]
            hits_stop = event_dom_info['hits_stop_idx'][dom_idx]
            num_hits = hits_stop - hits_start

            # Extract the components of the DOM coordinate just once, here
            dom_x = event_dom_info['x'][dom_idx]
            dom_y = event_dom_info['y'][dom_idx]
            dom_z = event_dom_info['z'][dom_idx]
            quantum_efficiency = event_dom_info['quantum_efficiency'][dom_idx]
            table_idx = event_dom_info['table_idx'][dom_idx]

            for source_idx in range(sources_start, sources_stop):
                source = sources[source_idx]
                dx = dom_x - source['x']
                dy = dom_y - source['y']
                dz = dom_z - source['z']
                source_t = source['time']

                rhosquared = dx*dx + dy*dy
                rsquared = rhosquared + dz*dz

                # Continue if photon is outside the radial binning limits
                if rsquared >= rsquared_max:
                    continue

                r = math.sqrt(rsquared)
                r = max(r, MACHINE_EPS)
                r_bin_idx = int(math.sqrt(r) / table_dr_pwr)
                costheta_bin_idx = int((1 - dz/r) / table_dcostheta)

                source_kind = source['kind']

                if source_kind == SRC_OMNI:
                    t_indep_surv_prob = np.mean(
                        t_indep_tables[table_idx, r_bin_idx, costheta_bin_idx, :, :]
                    )

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

                    t_indep_surv_prob = t_indep_tables[
                        table_idx,
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]

                source_photons = source['photons']

                ti_norm = t_indep_table_norm[r_bin_idx]
                dom_exp['total_expected_charge'][dom_idx] += (
                    source_photons * ti_norm * t_indep_surv_prob * quantum_efficiency
                )

                for hit_idx in range(hits_start, hits_stop):
                    hit_time = hits[hit_idx]['time']

                    # Causally impossible? (Note the comparison is written such that it
                    # will evaluate to True if hit_time is NaN.)
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
                            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx
                        )

                    else: # source_kind == SRC_CKV_BETA1
                        surv_prob_at_hit_t = table_lookup(
                            tables,
                            table_idx,
                            r_bin_idx,
                            costheta_bin_idx,
                            t_bin_idx,
                            costhetadir_bin_idx,
                            deltaphidir_bin_idx
                        )

                    r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]
                    dom_exp['exp_at_hit_times'][dom_idx] += (
                        source_photons * r_t_bin_norm * surv_prob_at_hit_t * quantum_efficiency
                    )


    if tbl_is_ckv and compute_t_indep_exp:
        pexp_5d = pexp_5d_ckv_compute_t_indep
    else:
        raise NotImplementedError()

            
    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def llh(dom_exp,
            time_window):
        '''
        helper function to calculate llh value
        '''

        llh = np.float64(0)

        for i in range(len(dom_exp)):
            llh -= dom_exp['total_expected_charge'][i]
            llh -= dom_exp['noise_rate_per_ns'][i] * time_window

            if dom_exp['total_observed_charge'][i] > 0:
                expr = math.log(dom_exp['total_expected_charge'][i] +
                                dom_exp['noise_rate_per_ns'][i] * time_window)
                if dom_exp['total_expected_charge'][i] > 0:
                    expr += dom_exp['exp_at_hit_times'][i] / dom_exp['total_expected_charge'][i] * (1. - 1./time_window)
                expr += 1./time_window
                llh += dom_exp['total_observed_charge'][i] * expr
        return llh

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh_all_doms(
            sources,
            pegleg_sources,
            hits,
            time_window,
            event_dom_info,
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
        pegleg_sources : shape (n_sources,) array of dtype SRC_T
            over these sources will be maximized in order
        hits : shape (n_hits_total,) array of dtype HIT_T
        time_window : float64
        event_dom_info : array of dtype EVT_DOM_INFO_T
        tables
            Stacked tables
        table_norm
            Single norm for all stacked tables
        t_indep_tables
            Stacked time-independent tables
        t_indep_table_norm
            Single norm for all stacked time-independent tables

        """
        dom_exp = np.zeros(shape=event_dom_info.shape, dtype=EXP_DOM_T)

        # realy necessary?
        # also this is a bit stupid
        for i in range(len(dom_exp)):
            dom_exp['noise_rate_per_ns'][i] = event_dom_info['noise_rate_per_ns'][i]
            for hit_idx in range(event_dom_info['hits_start_idx'][i], event_dom_info['hits_stop_idx'][i]):
                dom_exp['total_observed_charge'][i] += hits[hit_idx]['charge']

        # get expectations
        pexp_5d(
            sources=sources,
            sources_start=0,
            sources_stop=len(sources),
            hits=hits,
            event_dom_info=event_dom_info,
            time_window=time_window,
            tables=tables,
            table_norm=table_norm,
            t_indep_tables=t_indep_tables,
            t_indep_table_norm=t_indep_table_norm,
            dom_exp=dom_exp,
        )

        n_llhs = 1 + len(pegleg_sources)
        llhs = np.zeros(n_llhs, dtype=np.float64)
            
        # compute initial LLH (and set all elements to that one)
        llhs += llh(
                    dom_exp=dom_exp,
                    time_window=time_window,
                    )

        for pegleg_idx in range(len(pegleg_sources)):
            # update with additional source
            pexp_5d(
                sources=pegleg_sources,
                sources_start=pegleg_idx,
                sources_stop=pegleg_idx+1,
                hits=hits,
                event_dom_info=event_dom_info,
                time_window=time_window,
                tables=tables,
                table_norm=table_norm,
                t_indep_tables=t_indep_tables,
                t_indep_table_norm=t_indep_table_norm,
                dom_exp=dom_exp,
            )
            llhs[pegleg_idx+1] = llh(
                                     dom_exp=dom_exp,
                                     time_window=time_window,
                                     )
            #still improving?
            best_idx = np.argmax(llhs)
            best_llh = llhs[best_idx]
            # if we weren't improving for the last 20 steps, break
            if pegleg_idx > best_idx + 20:
                #print('no improvement')
                break
            # if improvements were small, break:
            if pegleg_idx > 30:
                delta_llh = llhs[pegleg_idx+1] - llhs[pegleg_idx - 30]
                if delta_llh < 1:
                    #print('little improvement')
                    break
            
        return best_llh, best_idx

    get_llh = get_llh_all_doms

    return pexp_5d, get_llh, meta
