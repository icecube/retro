# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals, consider-using-enumerate

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['MACHINE_EPS', 'generate_pexp_5d_function']

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
from retro.const import SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI
from retro.utils.geom import generate_digitizer


MACHINE_EPS = 1e-16
LLH_VERSION = 2


def generate_pexp_5d_function(
        table,
        table_kind,
        use_residual_time,
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

    use_residual_time : bool
        Whether time axis is actually time residual, defined to be
        (actual time) - (fastest time a photon could get to spatial coordinate)

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

    get_llh : callable

    meta : OrderedDict
        Paramters, including the binning, that uniquely identify what the
        capabilities of the returned `pexp_5d`. (Use this to eliminate
        redundant pexp_5d functions.)

    """
    # pylint: disable=missing-docstring
    tbl_is_raw = table_kind in ['raw_uncompr', 'raw_templ_compr']
    tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']
    assert tbl_is_raw or tbl_is_ckv

    if not compute_t_indep_exp:
        assert not compute_unhit_doms

    meta = OrderedDict([
        ('table_kind', table_kind),
        ('compute_t_indep_exp', compute_t_indep_exp),
        ('use_directionality', use_directionality),
        ('num_phi_samples', None if tbl_is_ckv or not use_directionality else num_phi_samples),
        ('ckv_sigma_deg', None if tbl_is_ckv or not use_directionality else ckv_sigma_deg),
        ('binning', OrderedDict([
            ('r_bin_edges', table['r_bin_edges']),
            ('costheta_bin_edges', table['costheta_bin_edges']),
            ('t_bin_edges', table['t_bin_edges']),
            ('costhetadir_bin_edges', table['costhetadir_bin_edges']),
            ('deltaphidir_bin_edges', table['deltaphidir_bin_edges']),
        ])),
    ])

    # Replace None with 0
    num_phi_samples = num_phi_samples or 0
    ckv_sigma_deg = ckv_sigma_deg or 0

    # NOTE: For now, we only support absolute value of deltaphidir (which
    # assumes azimuthal symmetry). In future, this could be revisited (and then
    # the abs(...) applied before binning in the pexp code will have to be
    # removed or replaced with behavior that depend on the range of the
    # deltaphidir_bin_edges).
    assert np.min(table['deltaphidir_bin_edges']) >= 0, 'only abs(deltaphidir) supported'

    # -- Define things used by `pexp_5d*` closures defined below -- #

    # Constants
    rsquared_max = np.max(table['r_bin_edges'])**2
    last_costhetadir_bin_idx = len(table['costhetadir_bin_edges']) - 1
    last_deltaphidir_bin_idx = len(table['deltaphidir_bin_edges']) - 1
    t_max = np.max(table['t_bin_edges'])
    recip_max_group_vel = table['group_refractive_index'] / SPEED_OF_LIGHT_M_PER_NS

    # Digitization functions for each binning dimension
    digitize_r = generate_digitizer(table['r_bin_edges'])
    digitize_ct = generate_digitizer(table['costheta_bin_edges'])
    digitize_t = generate_digitizer(table['t_bin_edges'])
    digitize_costhetadir = generate_digitizer(table['costhetadir_bin_edges'])
    digitize_deltaphidir = generate_digitizer(table['deltaphidir_bin_edges'])

    # Indexing functions for table types omni / directional lookups
    if tbl_is_templ_compr:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx):
            templ = tables[table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx]
            return templ['weight'] / template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx):
            templ = tables[table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx]
            return (
                templ['weight']
                * template_library[templ['index'], costhetadir_bin_idx, deltaphidir_bin_idx]
            )
    else: # table is not template-compressed
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(table, r_bin_idx, costheta_bin_idx, t_bin_idx):
            return np.mean(table[r_bin_idx, costheta_bin_idx, t_bin_idx])

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(table, r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx):
            return table[r_bin_idx, costheta_bin_idx, t_bin_idx,
                         costhetadir_bin_idx, deltaphidir_bin_idx]
    table_lookup_mean.__doc__ = (
        """Helper function for directionality-averaged table lookup"""
    )
    table_lookup.__doc__ = """Helper function for directional table lookup"""

    # -- Define `pexp_5d*` closures (functions that use things defined above) -- #

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d_ckv_compute_t_indep(
            sources,
            sources_start,
            sources_stop,
            event_dom_info,
            event_hit_info,
            tables,
            table_norm,
            t_indep_tables,
            t_indep_table_norm,
            dom_exp,
            hit_exp,
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
            Starting and stopping indices for the part of the array on which to
            work. Note that the latter is exclusive, i.e., following Python
            range / slice syntx. Hence, the following section of `sources` will
            be operated upon: .. ::

                sources[sources_start:sources_stop]

        event_dom_info : shape (n_operational_doms,) array of dtype EVT_DOM_INFO_T

        event_hit_info : shape (n_hits,) array of dtype EVT_HIT_INFO_T

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

        dom_exp : shape (n_operational_doms,) array of floats
            Expectation of hits for each operational DOM; initialize outside of
            calls to this function, as values are incremented within this
            function.

        hit_exp
            Time-dependent hit expectation at each (actual) hit time;
            initialize outside of this function, as values are incremented
            within this function.

        Out
        ---
        dom_exp
            See above
        hit_exp
            See above

        """
        for source_idx in range(sources_start, sources_stop):
            source = sources[source_idx]
            src_kind = source['kind']
            src_x = source['x']
            src_y = source['y']
            src_z = source['z']
            src_time = source['time']
            src_dir_cosphi = source['dir_cosphi']
            src_dir_sinphi = source['dir_sinphi']
            src_dir_costheta = source['dir_costheta']
            src_photons = source['photons']

            for dom_idx in range(len(event_dom_info)):
                dom = event_dom_info[dom_idx]
                dom_tbl_idx = dom['table_idx']
                dom_qe = dom['quantum_efficiency']
                dom_hits_start_idx = dom['hits_start_idx']
                dom_hits_stop_idx = dom['hits_stop_idx']

                dx = dom['x'] - src_x
                dy = dom['y'] - src_y
                neg_dz = src_z - dom['z']

                rhosquared = dx*dx + dy*dy
                rsquared = rhosquared + neg_dz*neg_dz

                # Continue if photon is outside the radial binning limits
                if rsquared >= rsquared_max:
                    continue

                r = max(math.sqrt(rsquared), MACHINE_EPS)

                r_bin_idx = digitize_r(r)
                costheta_bin_idx = digitize_ct(neg_dz/r)

                if src_kind == SRC_OMNI:
                    t_indep_surv_prob = np.mean(
                        t_indep_tables[dom_tbl_idx, r_bin_idx, costheta_bin_idx]
                    )

                else: # SRC_CKV_BETA1:
                    # Note that for these tables, we have to invert the photon
                    # direction relative to the vector from the DOM to the photon's
                    # vertex since simulation has photons going _away_ from the DOM
                    # that in reconstruction will hit the DOM if they're moving
                    # _towards_ the DOM.

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
                        pdir_deltaphi = 0
                    else:
                        pdir_cosdeltaphi = (src_dir_cosphi*dx + src_dir_sinphi*dy) / rho

                        # Note the max and min to clip value to [-1, 1];
                        # otherwise, numerical precision issues can cause the
                        # dot product to blow up.
                        pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))
                        pdir_deltaphi = math.acos(pdir_cosdeltaphi)

                    # Find directional bin indices & make upper edges inclusive
                    costhetadir_bin_idx = min(
                        digitize_costhetadir(src_dir_costheta),
                        last_costhetadir_bin_idx
                    )
                    deltaphidir_bin_idx = min(
                        digitize_deltaphidir(abs(pdir_deltaphi)),
                        last_deltaphidir_bin_idx
                    )

                    t_indep_surv_prob = t_indep_tables[
                        dom_tbl_idx,
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]

                ti_norm = t_indep_table_norm[r_bin_idx]
                dom_exp[dom_idx] += src_photons * ti_norm * t_indep_surv_prob * dom_qe

                for hit_idx in range(dom_hits_start_idx, dom_hits_stop_idx):
                    # A photon that starts immediately in the past (before the DOM
                    # was hit) will show up in the Retro DOM tables in bin 0; the
                    # further in the past the photon started, the higher the time
                    # bin index. Therefore, subract source time from hit time.
                    dt = event_hit_info[hit_idx]['time'] - src_time

                    if use_residual_time:
                        direct_time = r * recip_max_group_vel
                        t_bin_idx = digitize_t(dt - direct_time)
                    else: # time is simply "time from start of event"
                        # Causally impossible? (Note the comparison is written such that it
                        # will evaluate to True if hit_time is NaN.)
                        if not dt >= 0:
                            continue
                        # Is relative time outside binning?
                        if dt >= t_max:
                            continue
                        t_bin_idx = digitize_t(dt)

                    if src_kind == SRC_OMNI:
                        surv_prob_at_hit_t = table_lookup_mean(
                            tables,
                            dom_tbl_idx,
                            r_bin_idx,
                            costheta_bin_idx,
                            t_bin_idx,
                        )

                    else: # SRC_CKV_BETA1
                        surv_prob_at_hit_t = table_lookup(
                            tables,
                            dom_tbl_idx,
                            r_bin_idx,
                            costheta_bin_idx,
                            t_bin_idx,
                            costhetadir_bin_idx,
                            deltaphidir_bin_idx,
                        )

                    r_t_bin_norm = table_norm[r_bin_idx, t_bin_idx]
                    hit_exp[hit_idx] += (
                        src_photons * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                    )

    if tbl_is_ckv and compute_t_indep_exp:
        pexp_5d = pexp_5d_ckv_compute_t_indep
    else:
        raise NotImplementedError()

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def eval_llh(
            event_dom_info,
            event_hit_info,
            dom_exp,
            hit_exp,
            scaling_dom_exp,
            scaling_hit_exp,
            sum_noise_rate,
            time_window,
            recip_time_window,
            last_scalefactor
    ):
        """Helper function to calculate llh value

        Parameters:
        -----------
        event_dom_info : array
            containing all relevant event per DOM info
        dom_exp : array
            containing the expectations for a given hypo
        scaling_exp : array
            containing the expectations for a scaling hypo
        sum_noise_rate
        time_window : float
            event time_window in ns
        last_scalefactor : float
            starting pint for minimizer

        Returns
        -------
        llh
        scalefactor

        """
        sum_scaling_exp_charge = np.sum(scaling_dom_exp)
        sum_nonscaling_exp_charge = np.sum(dom_exp) + sum_noise_rate * time_window
        num_active_doms = len(event_dom_info)
        num_hits = len(event_hit_info)

        # -- Find the best scale factor -- #

        # Note `grad` defined as closure is faster than as external function
        def grad(scalefactor):
            """Gradient of time independent LLH part used for finding cascade energy.

            Parameters
            ----------
            scalefactor : float

            Returns
            -------
            neg_grad_llh : float

            """
            grad_llh = -sum_scaling_exp_charge # pylint: disable=invalid-unary-operand-type
            for dom_idx in range(num_active_doms):
                dom = event_dom_info[dom_idx]
                dom_total_observed_charge = dom['total_observed_charge']
                if dom_total_observed_charge > 0:
                    dom_scaling_dom_exp = scaling_dom_exp[dom_idx]
                    exp = dom_exp[dom_idx] + scalefactor * dom_scaling_dom_exp
                    exp += dom['noise_rate_per_ns'] * time_window
                    grad_llh += dom_total_observed_charge / exp * dom_scaling_dom_exp
            return -grad_llh

        # Gradient descent
        scalefactor = last_scalefactor
        gamma = 10.
        epsilon = 1e-1
        previous_step = 100
        n = 0
        while previous_step > epsilon and scalefactor >= 0 and scalefactor < 1000 and n < 100:
            gradient = grad(scalefactor)
            step = -gamma * gradient
            scalefactor += step
            previous_step = abs(step)
            n += 1

        # Make scalefactor >= 0
        scalefactor = max(0, scalefactor)

        # -- Calculate the scale factor -- #

        llh = -sum_nonscaling_exp_charge - scalefactor * sum_scaling_exp_charge
        # Second time-independent part
        for dom_idx in range(num_active_doms):
            dom = event_dom_info[dom_idx]
            dom_total_observed_charge = dom['total_observed_charge']
            if dom_total_observed_charge > 0:
                exp = dom_exp[dom_idx] + scalefactor * scaling_dom_exp[dom_idx]
                exp += dom['noise_rate_per_ns'] * time_window
                llh += dom_total_observed_charge * math.log(exp)

        # Time-dependent part
        for hit_idx in range(num_hits):
            hit = event_hit_info[hit_idx]
            dom_idx = hit['dom_idx']
            exp = hit_exp[hit_idx] + scalefactor * scaling_hit_exp[hit_idx]
            norm = dom_exp[dom_idx] + scalefactor * scaling_dom_exp[dom_idx]
            if norm > 0:
                p = exp / norm
            else:
                p = 0.
            llh += hit['charge'] * math.log(p * (1 - recip_time_window) + recip_time_window)

        return llh, scalefactor

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh_all_doms(
            sources,
            pegleg_sources,
            scaling_sources,
            event_hit_info,
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
        scaling_sources : shape (n_sources,) array of dtype SRC_T
            over these sources will be maximized using a scalefactor
        event_hit_info : shape (n_event_hit_info_total,) array of dtype HIT_T
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

        Returns
        -------
        llh : float
            Log-likelihood value at best pegleg hypo
        pegleg_idx : int
            Index for best pegleg hypo
        scalefactor : float
            Scale factor for scaling sources at best pegleg hypo

        """
        # Each pegleg iteration, include this many more pegleg sources
        pegleg_stepsize = 1

        num_pegleg_sources = len(pegleg_sources)
        num_llhs = 1 + int(num_pegleg_sources / pegleg_stepsize)
        num_active_doms = len(event_dom_info)
        num_hits = len(event_hit_info)

        recip_time_window = 1 / time_window
        sum_noise_rate = np.sum(event_dom_info['noise_rate_per_ns'])

        # Initialize expectations arrays
        dom_exp = np.zeros(shape=num_active_doms, dtype=np.float64)
        hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        # Save the scaling sources expectations in separate arrays
        scaling_dom_exp = np.zeros(shape=num_active_doms, dtype=np.float64)
        scaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        # Get scaling sources expectations first for a nominal (e.g. cascade)
        # hypothesis that will then be scaled below
        pexp_5d(
            sources=scaling_sources,
            sources_start=0,
            sources_stop=len(scaling_sources),
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            tables=tables,
            table_norm=table_norm,
            t_indep_tables=t_indep_tables,
            t_indep_table_norm=t_indep_table_norm,
            dom_exp=scaling_dom_exp,
            hit_exp=scaling_hit_exp,
        )

        # Get expectations for non-scaling sources
        pexp_5d(
            sources=sources,
            sources_start=0,
            sources_stop=len(sources),
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            tables=tables,
            table_norm=table_norm,
            t_indep_tables=t_indep_tables,
            t_indep_table_norm=t_indep_table_norm,
            dom_exp=dom_exp,
            hit_exp=hit_exp,
        )

        # Compute initial LLH (and set all elements to that one)
        llh, scalefactor = eval_llh(
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            dom_exp=dom_exp,
            hit_exp=hit_exp,
            scaling_dom_exp=scaling_dom_exp,
            scaling_hit_exp=scaling_hit_exp,
            sum_noise_rate=sum_noise_rate,
            time_window=time_window,
            recip_time_window=recip_time_window,
            last_scalefactor=10.,
        )

        llhs = np.full(shape=num_llhs, fill_value=llh, dtype=np.float64)
        scalefactors = np.zeros(shape=num_llhs, dtype=np.float64)
        scalefactors[0] = scalefactor

        best_idx = 0
        llh_at_best_idx = llh

        for i, pegleg_idx in enumerate(range(0, num_pegleg_sources, pegleg_stepsize)):
            llh_idx = i + 1
            # Update pegleg sources with additional segment of sources
            pexp_5d(
                sources=pegleg_sources,
                sources_start=pegleg_idx,
                sources_stop=pegleg_idx + pegleg_stepsize,
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                tables=tables,
                table_norm=table_norm,
                t_indep_tables=t_indep_tables,
                t_indep_table_norm=t_indep_table_norm,
                dom_exp=dom_exp,
                hit_exp=hit_exp,
            )

            # Get new best scaling factor, llh for the scaling sources
            llh, scalefactor = eval_llh(
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                dom_exp=dom_exp,
                hit_exp=hit_exp,
                scaling_dom_exp=scaling_dom_exp,
                scaling_hit_exp=scaling_hit_exp,
                sum_noise_rate=sum_noise_rate,
                time_window=time_window,
                recip_time_window=recip_time_window,
                last_scalefactor=scalefactor,
            )

            # Store this pegleg step's llh and best scalefactor
            llhs[llh_idx] = llh
            scalefactors[llh_idx] = scalefactor

            if llh > llh_at_best_idx:
                best_idx = pegleg_idx
                llh_at_best_idx = llh

            # TODO: make this more general, less hacky continue/stop condition

            #still improving?
            # if we weren't improving for the last 30 steps, break
            #if pegleg_idx > best_idx + 300:
            #    #print('no improvement')
            #    break
            # if improvements were small or none, break:
            if pegleg_idx > 300:
                delta_llh = llh - llhs[pegleg_idx - 300]
                if delta_llh < 0.5:
                    #print('little improvement')
                    break

        return llh_at_best_idx, best_idx, scalefactors[best_idx]

    get_llh = get_llh_all_doms

    return pexp_5d, get_llh, meta
