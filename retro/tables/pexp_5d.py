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

from scipy import stats

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit
from retro.const import SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI
from retro.utils.geom import generate_digitizer


MACHINE_EPS = 1e-10

#maximum radius to consider
MAX_RAD_SQ = 500**2

def generate_pexp_5d_function(
    table,
    table_kind,
    t_is_residual_time,
    compute_t_indep_exp,
    compute_unhit_doms,
    use_directionality,
    num_phi_samples=None,
    ckv_sigma_deg=None,
    template_library=None,
):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function to use.

    Parameters
    ----------
    table : mapping
        As returned by `load_clsim_table_minimal`

    table_kind : str in {'raw_uncompr', 'ckv_uncompr', 'ckv_templ_compr'}

    t_is_residual_time : bool
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
        Parameters, including the binning, that uniquely identify what the
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

    # Replace None with 0 for sending to Numba functions
    if num_phi_samples is None:
        num_phi_samples = 0
    if ckv_sigma_deg is None:
        ckv_sigma_deg = 0

    # NOTE: For now, we only support absolute value of deltaphidir (which
    # assumes azimuthal symmetry). In future, this could be revisited (and then
    # the abs(...) applied before binning in the pexp code will have to be
    # removed or replaced with behavior that depend on the range of the
    # deltaphidir_bin_edges).
    assert np.min(table['deltaphidir_bin_edges']) >= 0, 'only abs(deltaphidir) supported'

    # -- Define things used by `pexp_5d*` closures defined below -- #

    # Constants
    rsquared_max = min(np.max(table['r_bin_edges'])**2, MAX_RAD_SQ)
    last_costhetadir_bin_idx = len(table['costhetadir_bin_edges']) - 2
    last_deltaphidir_bin_idx = len(table['deltaphidir_bin_edges']) - 2
    t_max = np.max(table['t_bin_edges'])
    recip_max_group_vel = table['group_refractive_index'] / SPEED_OF_LIGHT_M_PER_NS

    # Digitization functions for each binning dimension
    digitize_r = generate_digitizer(table['r_bin_edges'])
    digitize_costheta = generate_digitizer(table['costheta_bin_edges'])
    digitize_t = generate_digitizer(table['t_bin_edges'])
    digitize_costhetadir = generate_digitizer(table['costhetadir_bin_edges'])
    digitize_deltaphidir = generate_digitizer(table['deltaphidir_bin_edges'])


    # to sample for DOM jitter etc
    #jitter_dt = np.arange(-10,11,2)
    #jitter_dt =  np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    jitter_dt =  np.array([0])
    #jitter_weights = stats.norm.pdf(jitter_dt, 0, 5)
    #jitter_weights /= np.sum(jitter_weights)

    # Indexing functions for table types omni / directional lookups
    if tbl_is_templ_compr:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx
        ):
            templ = tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx]
            return templ['weight'] / template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
            costhetadir_bin_idx, deltaphidir_bin_idx
        ):
            templ = tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx]
            return (
                templ['weight'] * template_library[
                    templ['index'],
                    costhetadir_bin_idx,
                    deltaphidir_bin_idx
                ]
            )

    else: # table is not template-compressed
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx
        ):
            return np.mean(tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx])

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
            costhetadir_bin_idx, deltaphidir_bin_idx
        ):
            return tables[table_idx][
                r_bin_idx,
                costheta_bin_idx,
                t_bin_idx,
                costhetadir_bin_idx,
                deltaphidir_bin_idx
            ]

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
        table_norms,
        t_indep_tables,
        t_indep_table_norms,
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

        table_norms : shape (n_tables, n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions.

        t_indep_table : array
            Time-independent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_costhetadir, n_deltaphidir)
            while if using a

        t_indep_table_norms : shape (n_tables, n_r) array
            r-dependent normalization (any t-dep normalization is assumed to
            already have been applied to generate the t_indep_table).

        hit_exp : shape (n_hits,) array of floats
            Time-dependent hit expectation at each (actual) hit time;
            initialize outside of this function, as values are incremented
            within this function. Values in `hit_exp` correspond to the values
            in `event_hit_info`.

        Returns
        -------
        t_indep_exp : float
            Expectation of total hits for all operational DOMs

        Out
        ---
        hit_exp
            See above

        """
        num_operational_doms = len(event_dom_info)
        t_indep_exp = 0.
        for source_idx in range(sources_start, sources_stop):
            src = sources[source_idx]
            src_kind = src['kind']
            src_x = src['x']
            src_y = src['y']
            src_z = src['z']
            src_time = src['time']
            src_dir_cosphi = src['dir_cosphi']
            src_dir_sinphi = src['dir_sinphi']
            src_dir_costheta = src['dir_costheta']
            src_photons = src['photons']

            for op_dom_idx in range(num_operational_doms):
                dom = event_dom_info[op_dom_idx]
                dom_tbl_idx = dom['table_idx']
                dom_qe = dom['quantum_efficiency']
                dom_hits_start_idx = dom['hits_start_idx']
                dom_hits_stop_idx = dom['hits_stop_idx']

                dx = src_x - dom['x']
                dy = src_y - dom['y']
                dz = src_z - dom['z']

                rhosquared = dx**2 + dy**2
                #rhosquared = max(MACHINE_EPS, rhosquared)
                rsquared = rhosquared + dz**2

                # Continue if photon is outside the radial binning limits
                if rsquared >= rsquared_max:
                    continue

                r = math.sqrt(rsquared)
                if r < MACHINE_EPS:
                    r = MACHINE_EPS
                r_bin_idx = digitize_r(r)

                costheta_bin_idx = digitize_costheta(dz/r)

                if src_kind == SRC_OMNI:
                    t_indep_surv_prob = np.mean(
                        t_indep_tables[dom_tbl_idx][r_bin_idx, costheta_bin_idx, :, :]
                    )

                else: # SRC_CKV_BETA1:
                    rho = math.sqrt(rhosquared)

                    # TODO/NOTE: apparently the dir signs are backwards in the
                    # below explanation. Figure out why & fix the explanation!

                    # Note in the following that we need to invert the
                    # directions of the sources to match the directions that
                    # Retro simulation comes up with, thus we work with
                    # -dir_vec. The angles associated with this that we want
                    # to work with are
                    #   cos(pi - thetadir) = -cos(thetadir),
                    #   sin(pi - thetadir) = sin(thetadir),
                    #   cos(phidir + pi) = -cos(phidir),
                    #   sin(phidir + pi) = sin(phidir)
                    #
                    # We bin cos(pi - thetadir), so need to simply bin the
                    # quantity `-src_dir_costheta`.
                    #
                    # We want to bin abs(deltaphidir), which is described now:
                    # Just look at vectors in the xy-plane, since we want
                    # difference of angle in this plane. Use dot product:
                    #   dot(-dir_vec_xy, pos_vec_xy) = |-dir_vec_xy| |pos_vec_xy| cos(deltaphidir)
                    # where the length of the directionality vector in the xy-plane is
                    #   |-dir_vec_xy| = rhodir = rdir * sin(pi - thetadir)
                    # and since rdir = 1 and the inversion of the angle above
                    #   |-dir_vec_xy| = rhodir = sin(thetadir).
                    # The length of the position vector in the xy-plane is
                    #   |pos_vec_xy| = rho = sqrt(dx^2 + dy^2)
                    # where dx and dy are src_x - dom_x and src_y - dom_y.
                    # Solving for cos(deltaphidir):
                    #   cos(deltaphidir) = dot(-dir_vec_xy, pos_vec_xy) / (rhodir * rho)
                    # we just need to write out the components of the dot
                    # product in terms of quantites we have:
                    #   -dir_vec_x = rhodir * cos(phidir + pi)
                    #   -dir_vec_y = rhodir * sin(phidir + pi)
                    #   pos_vec_x = dx
                    #   pos_vec_y = dy
                    # giving
                    #   cos(deltaphidir) = (rhodir*cos(phidir+pi)*dx + rhodir*sin(phidir+pi)*dy)/(rhodir*rho)
                    # cancel rhodir out
                    #   cos(deltaphidir) = (cos(phidir + pi)*dx + sin(phidir + pi)*dy) / rho
                    # and substitute the identities above
                    #   cos(deltaphidir) = (-cos(phidir)*dx - sin(phidir)*dy) / rho
                    # Finally, solve for deltaphidir
                    #   deltaphidir = acos((-cos(phidir)*dx - sin(phidir)*dy) / rho)

                    if rho <= MACHINE_EPS:
                        absdeltaphidir = 0
                    else:
                        cosdeltaphidir = - (src_dir_cosphi*dx + src_dir_sinphi*dy) / rho

                        # Clip cosdeltaphidir within range [-1, 1] in case of
                        # finite precision issues in the above
                        cosdeltaphidir = min(1, max(-1, cosdeltaphidir))
                        absdeltaphidir = abs(math.acos(cosdeltaphidir))

                    # Find directional bin indices
                    costhetadir_bin_idx = digitize_costhetadir(src_dir_costheta)
                    deltaphidir_bin_idx = digitize_deltaphidir(absdeltaphidir)

                    # Make upper edges inclusive
                    if costhetadir_bin_idx > last_costhetadir_bin_idx:
                        costhetadir_bin_idx = last_costhetadir_bin_idx

                    if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                        deltaphidir_bin_idx = last_deltaphidir_bin_idx

                    t_indep_surv_prob = t_indep_tables[dom_tbl_idx][
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]

                ti_norm = t_indep_table_norms[dom_tbl_idx][r_bin_idx]
                t_indep_exp += src_photons * ti_norm * t_indep_surv_prob * dom_qe

                for hit_idx in range(dom_hits_start_idx, dom_hits_stop_idx):
                    # A photon that starts immediately in the past (before the
                    # DOM was hit) will show up in the Retro DOM tables in bin
                    # 0; the further in the past the photon started, the
                    # higher the time bin index. Therefore, subract source
                    # time from hit time.
                    nominal_dt = event_hit_info[hit_idx]['time'] - src_time

                    if t_is_residual_time:
                        nominal_dt -= r * recip_max_group_vel

                    jitter_surv_probs = np.zeros_like(jitter_dt)

                    for jitter_idx in range(len(jitter_dt)):
                        dt = nominal_dt + jitter_dt[jitter_idx]
                        #weight = jitter_weights[jitter_idx]

                        # Note the comparison is written such that it will evaluate
                        # to True if hit_time is NaN.
                        if (not dt >= 0) or dt >= t_max:
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
                        jitter_surv_probs[jitter_idx] = surv_prob_at_hit_t

                    surv_prob_at_hit_t = np.max(jitter_surv_probs)

                    r_t_bin_norm = table_norms[dom_tbl_idx][r_bin_idx, t_bin_idx]
                    hit_exp[hit_idx] += (
                        src_photons * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                    )

        return t_indep_exp

    if tbl_is_ckv and compute_t_indep_exp:
        pexp_5d = pexp_5d_ckv_compute_t_indep
    else:
        raise NotImplementedError()

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_optimal_scalefactor(
        event_dom_info,
        event_hit_info,
        nonscaling_hit_exp,
        nonscaling_t_indep_exp,
        nominal_scaling_hit_exp,
        nominal_scaling_t_indep_exp,
        initial_scalefactor,
    ):
        """Find optimal (highest-likelihood) `scalefactor` for scaling sources.

        Parameters:
        -----------
        event_dom_info : array of dtype EVT_DOM_INFO_T
            containing all relevant event per DOM info
        event_hit_info : array of dtype EVT_HIT_INFO_T
        nominal_scaling_t_indep_exp : float
            Total charge expected across the detector due to scaling sources (Lambda^s
            in `likelihood_function_derivation.ipynb`)
        nominal_scaling_hit_exp : shape (n_hits,) array of dtype float
            Detected-charge-rate expectation at each hit time due to scaling sources at
            nominal values (i.e., with `scalefactor = 1`); this quantity is
            lambda_d^s(t_{k_d}) in `likelihood_function_derivation.ipynb`
        pegleg_hit_exp : shape (n_hits,) array of dtype float
            Detected-charge-rate expectation at each hit time due to pegleg sources;
            this is lambda_d^p(t_{k_d}) in `likelihood_function_derivation.ipynb`
        generic_hit_exp : shape (n_hits,) array of dtype float
            Detected-charge-rate expectation at each hit time due to generic sources;
            this is lambda_d^g(t_{k_d}) in `likelihood_function_derivation.ipynb`
        initial_scalefactor : float > 0
            Starting point for minimizer

        Returns
        -------
        scalefactor
        llh

        """
        # Note: defining as closure is faster than as external function
        def get_grad_neg_llh_wrt_scalefactor(scalefactor):
            """Compute the gradient of -LLH with respect to `scalefactor`.

            Typically we use `scalefactor` with cascade energy, .. ::

                cascade_energy = scalefactor * nominal_cascade_energy

            so the gradient is proportional to cascade energy by a factor of
            `nominal_cascade_energy`.

            Parameters
            ----------
            scalefactor : float

            Returns
            -------
            grad_neg_llh : float

            """
            # Time- and DOM-independent part of grad(-LLH)
            grad_neg_llh = nominal_scaling_t_indep_exp

            # Time-dependent part of grad(-LLH) (i.e., at hit times)
            for hit_idx, hit_info in enumerate(event_hit_info):
                grad_neg_llh -= (
                    hit_info['charge'] * nominal_scaling_hit_exp[hit_idx]
                    / (
                        event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                        + scalefactor * nominal_scaling_hit_exp[hit_idx]
                        + nonscaling_hit_exp[hit_idx]
                      )
                )

            return grad_neg_llh

        # -- Perform gradient descent on -LLH -- #

        # See, e.g., https://en.wikipedia.org/wiki/Gradient_descent#Python

        scalefactor = initial_scalefactor
        gamma = 10. # step size multiplier
        epsilon = 1e-1 # tolerance
        iters = 0 # iteration counter
        while True:
            gradient = get_grad_neg_llh_wrt_scalefactor(scalefactor)
            step = -gamma * gradient
            scalefactor += step
            iters += 1
            if (
                abs(step) < epsilon
                or scalefactor <= -100
                or scalefactor >= 1000
                or iters >= 100
            ):
                break

        scalefactor = max(0., min(1000., scalefactor))

        # -- Calculate llh at the optimal `scalefactor` found -- #

        # Time- and DOM-independent part of LLH
        llh = -scalefactor * nominal_scaling_t_indep_exp - nonscaling_t_indep_exp

        # Time-dependent part of LLH (i.e., at hit times)
        for hit_idx, hit_info in enumerate(event_hit_info):
            llh += hit_info['charge'] * math.log(
                event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                + scalefactor * nominal_scaling_hit_exp[hit_idx]
                + nonscaling_hit_exp[hit_idx]
            )

        return scalefactor, llh

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh(
        generic_sources,
        pegleg_sources,
        scaling_sources,
        event_hit_info,
        event_dom_info,
        tables,
        table_norms,
        t_indep_tables,
        t_indep_table_norms,
        pegleg_stepsize,
    ):
        """Compute log likelihood for hypothesis sources given an event.

        This version of get_llh is specialized to compute log-likelihoods for
        all DOMs, whether or not they were hit. Use this if you aren't already
        using a TDI table to get the likelihood term corresponding to
        time-independent expectation.

        Parameters
        ----------
        generic_sources : shape (n_generic_sources,) array of dtype SRC_T
            If NOT using the pegleg/scaling procedure, all light sources are placed in
            this array; when using the pegleg/scaling procedure, `generic_sources` will
            be empty (i.e., `n_generic_sources = 0`)
        pegleg_sources : shape (n_pegleg_sources,) array of dtype SRC_T
            If using the pegleg/scaling procedure, the likelihood is maximized by
            including more and more of these sources (in the order given); if not using
            the pegleg/scaling procedures, `pegleg_sources` will be an empty array
            (i.e., `n_pegleg_sources = 0`)
        scaling_sources : shape (n_scaling_sources,) array of dtype SRC_T
            If using the pegleg/scaling procedure, the likelihood is maximized by
            scaling the luminosity of these sources; if not using the pegleg/scaling
            procedure, `scaling_sources` will be an empty array (i.e.,
            `n_scaling_sources = 0`)
        event_hit_info : shape (n_hits,) array of dtype EVT_HIT_INFO_T
        event_dom_info : shape (n_operational_doms,) array of dtype EVT_DOM_INFO_T
        tables
            Stacked tables
        table_norms
            Norms for all stacked tables
        t_indep_tables
            Stacked time-independent tables
        t_indep_table_norms
            Norms for all stacked time-independent tables
        pegleg_stepsize : int > 0
            Number of pegleg sources to add each time around the pegleg loop; ignored if
            pegleg/scaling procedure is not performed

        Returns
        -------
        llh : float
            Log-likelihood value at best pegleg hypo
        pegleg_stop_idx : int
            Stop index for `pegleg_sources` to obtain optimal LLH .. ::
                pegleg_sources[:pegleg_stop_idx]
        scalefactor : float
            Best scale factor for `scaling_sources` at best pegleg hypo

        """
        # TODO: make pegleg_stepsize a kwarg param somehow
        # Each pegleg iteration, include this many more pegleg sources

        num_pegleg_sources = len(pegleg_sources)
        # take log steps
        logstep = np.log(num_pegleg_sources) / 300
        logspace = np.zeros(shape=301, dtype=np.int32)
        x = -1e-8
        for i in range(len(logspace)):
            logspace[i] = np.int32(np.exp(x))
            x+= logstep
        pegleg_steps = np.unique(logspace)
        assert pegleg_steps[0] == 0
        n_pegleg_steps = len(pegleg_steps)
        #print(pegleg_steps)

        #pegleg_stepsize = 1

        #num_pegleg_sources = len(pegleg_sources)
        #num_pegleg_llhs = 1 + int(num_pegleg_sources / pegleg_stepsize)
        num_operational_doms = len(event_dom_info)
        num_hits = len(event_hit_info)

        # -- Expectations due to nominal (`scalefactor = 1`) scaling sources -- #

        nominal_scaling_t_indep_exp = 0.
        nominal_scaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        nominal_scaling_t_indep_exp += pexp_5d(
            sources=scaling_sources,
            sources_start=0,
            sources_stop=len(scaling_sources),
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            tables=tables,
            table_norms=table_norms,
            t_indep_tables=t_indep_tables,
            t_indep_table_norms=t_indep_table_norms,
            hit_exp=nominal_scaling_hit_exp,
        )

        # -- Storage for exp due to generic + pegleg (non-scaling) sources -- #

        nonscaling_t_indep_exp = 0.
        nonscaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        # Expectations for generic-only sources (i.e. pegleg=0 at this point)
        nonscaling_t_indep_exp += pexp_5d(
            sources=generic_sources,
            sources_start=0,
            sources_stop=len(generic_sources),
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            tables=tables,
            table_norms=table_norms,
            t_indep_tables=t_indep_tables,
            t_indep_table_norms=t_indep_table_norms,
            hit_exp=nonscaling_hit_exp,
        )

        # Compute initial scalefactor & LLH for generic-only (no pegleg) sources
        scalefactor, llh = get_optimal_scalefactor(
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            nonscaling_hit_exp=nonscaling_hit_exp,
            nonscaling_t_indep_exp=nonscaling_t_indep_exp,
            nominal_scaling_hit_exp=nominal_scaling_hit_exp,
            nominal_scaling_t_indep_exp=nominal_scaling_t_indep_exp,
            initial_scalefactor=10.,
        )

        llhs = np.full(shape=n_pegleg_steps, fill_value=llh, dtype=np.float64)
        llhs[0] = llh

        scalefactors = np.zeros(shape=n_pegleg_steps, dtype=np.float64)
        scalefactors[0] = scalefactor

        best_llh = llh
        best_llh_idx = 0
        getting_worse_counter = 0

        # -- Pegleg loop -- #

        for pegleg_idx in range(1, n_pegleg_steps):
            # Update pegleg sources with additional segment of sources
            nonscaling_t_indep_exp += pexp_5d(
                sources=pegleg_sources,
                sources_start=pegleg_steps[pegleg_idx-1],
                sources_stop=pegleg_steps[pegleg_idx],
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                tables=tables,
                table_norms=table_norms,
                t_indep_tables=t_indep_tables,
                t_indep_table_norms=t_indep_table_norms,
                hit_exp=nonscaling_hit_exp,
            )

            # Find optimal scalefactor at this pegleg step
            scalefactor, llh = get_optimal_scalefactor(
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                nonscaling_hit_exp=nonscaling_hit_exp,
                nonscaling_t_indep_exp=nonscaling_t_indep_exp,
                nominal_scaling_hit_exp=nominal_scaling_hit_exp,
                nominal_scaling_t_indep_exp=nominal_scaling_t_indep_exp,
                initial_scalefactor=scalefactor,
            )

            # Store this pegleg step's llh and best scalefactor
            llhs[pegleg_idx] = llh
            scalefactors[pegleg_idx] = scalefactor

            if llh > best_llh:
                best_llh = llh
                best_llh_idx = pegleg_idx
                getting_worse_counter = 0

            elif llh < best_llh - 0.2:
                getting_worse_counter += 1

            # break condition
            if getting_worse_counter > 5:
                #for idx in range(pegleg_idx+1,n_pegleg_steps):
                #    # fill up with bad llhs. just to make sure they're not used
                #    llhs[idx] = best_llh - 100
                #print('break at step ',pegleg_idx)
                break

        # find the best pegleg idx:
        #best_llh = np.max(llhs)
        #n_good_indices = np.sum(llhs > best_llh - 0.1)
        #median_good_idx = max(1,np.int(n_good_indices/2))
        
        # search for that median pegleg index
        #counter = 0
        #for best_idx in range(n_pegleg_steps):
        #    if llhs[best_idx] > best_llh - 0.1:
        #        counter +=1
        #    if counter == median_good_idx:
        #        break
        
        #good_indices = np.argwhere(llhs > best_llh - 0.1)
        #best_idx = np.median(good_indices)

        return llhs[best_llh_idx], pegleg_steps[best_llh_idx], scalefactors[best_llh_idx]

    return pexp_5d, get_llh, meta
