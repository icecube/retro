# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals, consider-using-enumerate

"""
Define `generate_llh_function` to... generate the function for obtaining
log-likelihoods given a photon-expectation function, photon sources and tables.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'Minimizer',
    'StepSpacing',
    'LLHChoice',
    'SCALE_FACTOR_MINIMIZER',
    'PEGLEG_SPACING',
    'PEGLEG_LLH_CHOICE',
    'PEGLEG_BEST_DELTA_LLH_THRESHOLD',
    'generate_llh_function',
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

import enum
import math
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit


class Minimizer(enum.IntEnum):
    """Minimizer to use for scale factor"""
    GRADIENT_DESCENT = 0
    NEWTON = 1
    BINARY_SEARCH = 2


class StepSpacing(enum.IntEnum):
    """Pegleg step spacing"""
    LINEAR = 0
    LOG = 1


class LLHChoice(enum.IntEnum):
    """How to choose the "best" LLH"""
    MAX = 0
    MEAN = 1
    MEDIAN = 2


SCALE_FACTOR_MINIMIZER = Minimizer.BINARY_SEARCH
"""Choice of which minimizer to use for computing scaling factor for scaling sources"""

PEGLEG_SPACING = StepSpacing.LINEAR
"""Pegleg adds segments either linearly (same number of segments independent of energy)
or logarithmically (more segments are added the longer the track"""

PEGLEG_LLH_CHOICE = LLHChoice.MEAN
"""How to choose best LLH from all Pegleg steps"""

PEGLEG_BEST_DELTA_LLH_THRESHOLD = 0.1
"""For Pegleg `LLHChoice` that require a range of LLH and average (mean, median, etc.),
take all LLH that are within this threshold of the maximum LLH"""

# Validation that module-level constants are consistent
if PEGLEG_SPACING is StepSpacing.LOG:
    assert PEGLEG_LLH_CHOICE is LLHChoice.MAX


def generate_llh_function(
    pexp,
    dom_tables,
    tdi_tables=None,
    tdi_metas=None,
):
    """Generate a numba-compiled function for computing log likelihoods.

    Parameters
    ----------
    pexp

    dom_tables : Retro5DTables
        Fully-loaded set of single-DOM tables (time-dependent and, if no `tdi_tables`,
        time-independent)

    tdi_tables : sequence of 1 or 2 arrays, optional
        Time- and DOM-independent tables.

    tdi_metas : sequence of 1 or 2 mappings, optional
        If provided, sequence must contain two mappings where the first
        corresponds to the finely-binned TDI table and the second corresponds
        to the coarsely-binned table (the first table takes precedence over the
        second for looking up sources). Each of the mappings must contain keys
        "bin_edges", itself a mapping containing "x", "y", "z", "costhetadir",
        and "phidir"; values of these are arrays of the bin edges in each of
        these dimensions. "costhetadir" must span [-1, 1] (inclusive) and
        "phidir" must span [-pi, pi] inclusive). All edges must be strictly
        monotonic and increasing.

    Returns
    -------
    llh : callable
        Function to compute log likelihood

    """
    if tdi_tables is None:
        tdi_tables = ()
    if tdi_metas is None:
        tdi_metas = ()

    tbl_is_ckv = dom_tables.table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    if not tbl_is_ckv:
        raise NotImplementedError('Only Ckv tables are implemented.')

    if len(tdi_tables) == 1:
        tdi_tables = (tdi_tables[0], tdi_tables[0])

    # NOTE: For now, we only support absolute value of deltaphidir (which
    # assumes azimuthal symmetry). In future, this could be revisited (and then
    # the abs(...) applied before binning in the pexp code will have to be
    # removed or replaced with behavior that depend on the range of the
    # deltaphidir_bin_edges).
    assert dom_tables.table_meta['deltaphidir_bin_edges'][0] == 0, 'only abs(deltaphidir) supported'
    assert dom_tables.table_meta['deltaphidir_bin_edges'][-1] == np.pi

    # -- Define vars used by `get_llh_` closure defined below -- #

    num_tdi_tables = len(tdi_metas)
    if num_tdi_tables == 0:
        # Numba needs an object that it can determine type of
        tdi_tables = 0

    # Copy full dom_tables object to another var (suffixed with underscore) since
    # we'll use the `dom_tables` name for just the table array in the closure
    dom_tables_ = dom_tables

    # Extract everything we from the class attributes into "regular" vars that
    # Numba can handle
    dom_tables = dom_tables_.tables
    dom_table_norms = dom_tables_.table_norms
    dom_tables_template_library = dom_tables_.template_library
    t_indep_dom_tables = dom_tables_.t_indep_tables
    t_indep_dom_table_norms = dom_tables_.t_indep_table_norms

    if not isinstance(dom_tables, np.ndarray):
        dom_tables = np.stack(dom_tables, axis=0)
        print('dom_tables.shape:', dom_tables.shape)
    if not isinstance(dom_table_norms, np.ndarray):
        dom_table_norms = np.stack(dom_table_norms, axis=0)
        print('dom_table_norms.shape:', dom_table_norms.shape)
    if not isinstance(t_indep_dom_tables, np.ndarray):
        t_indep_dom_tables = np.stack(t_indep_dom_tables, axis=0)
        print('t_indep_dom_tables.shape:', t_indep_dom_tables.shape)
    if not isinstance(t_indep_dom_table_norms, np.ndarray):
        t_indep_dom_table_norms = np.stack(t_indep_dom_table_norms, axis=0)
        print('t_indep_dom_table_norms.shape:', t_indep_dom_table_norms.shape)

    dom_tables.flags.writeable = False
    dom_table_norms.flags.writeable = False
    dom_tables_template_library.flags.writeable = False
    t_indep_dom_tables.flags.writeable = False
    t_indep_dom_table_norms.flags.writeable = False

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def simple_llh(
        event_dom_info,
        event_hit_info,
        nonscaling_hit_exp,
        nonscaling_t_indep_exp,
    ):
        """Get llh if no scaling sources are present.

        Parameters:
        -----------
        event_dom_info : array of dtype EVT_DOM_INFO_T
            All relevant per-DOM info for the event

        event_hit_info : array of dtype EVT_HIT_INFO_T

        Returns
        -------
        llh

        """
        # Time- and DOM-independent part of LLH
        llh = -nonscaling_t_indep_exp

        # Time-dependent part of LLH (i.e., at hit times)
        for hit_idx, hit_info in enumerate(event_hit_info):
            llh += hit_info['charge'] * math.log(
                event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                + nonscaling_hit_exp[hit_idx]
            )

        return llh

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_optimal_scalefactor(
        event_dom_info,
        event_hit_info,
        nonscaling_hit_exp,
        nonscaling_t_indep_exp,
        nominal_scaling_hit_exp,
        nominal_scaling_t_indep_exp,
        initial_scalefactor,
        scaling_cascade_energy,
    ):
        """Find optimal (highest-likelihood) `scalefactor` for scaling sources.

        Parameters:
        -----------
        event_dom_info : array of dtype EVT_DOM_INFO_T
            All relevant per-DOM info for the event
        event_hit_info : array of dtype EVT_HIT_INFO_T
        nonscaling_hit_exp : shape (n_hits, 2) array of dtype float
            Detected-charge-rate expectation at each hit time due to pegleg sources;
            this is lambda_d^p(t_{k_d}) in `likelihood_function_derivation.ipynb`
        nonscaling_t_indep_exp : float
            Total charge expected across the detector due to non-scaling sources
            (Lambda^s in `likelihood_function_derivation.ipynb`)
        nominal_scaling_hit_exp : shape (n_hits, 2) array of dtype float
            Detected-charge-rate expectation at each hit time due to scaling sources at
            nominal values (i.e., with `scalefactor = 1`); this quantity is
            lambda_d^s(t_{k_d}) in `likelihood_function_derivation.ipynb`
        nominal_scaling_t_indep_exp : float
            Total charge expected across the detector due to nominal scaling sources
            (Lambda^s in `likelihood_function_derivation.ipynb`)
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

        def get_newton_step(scalefactor):
            """Compute the step for the newton method for the `scalefactor`

            the step is defined as -f'/f'' where f is the LLH(scalefactor)

            Parameters
            ----------
            scalefactor : float

            Returns
            -------
            step : float

            """
            # Time- and DOM-independent part of grad(-LLH)
            numerator = nominal_scaling_t_indep_exp
            denominator = 0

            # Time-dependent part of grad(-LLH) (i.e., at hit times)
            for hit_idx, hit_info in enumerate(event_hit_info):
                s = (hit_info['charge'] * nominal_scaling_hit_exp[hit_idx]
                    / (
                        event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                        + scalefactor * nominal_scaling_hit_exp[hit_idx]
                        + nonscaling_hit_exp[hit_idx]
                      )
                )
                numerator -= s
                denominator += s**2

            if denominator == 0:
                return -1
            return numerator/denominator

        if SCALE_FACTOR_MINIMIZER is Minimizer.GRADIENT_DESCENT:
            # See, e.g., https://en.wikipedia.org/wiki/Gradient_descent#Python

            #print('Initial scalefactor: ', initial_scalefactor)
            scalefactor = initial_scalefactor
            #previous_scalefactor = initial_scalefactor
            gamma = 0.1 # step size multiplier
            epsilon = 1e-2 # tolerance
            iters = 0 # iteration counter
            max_iter = 500
            while True:
                gradient = get_grad_neg_llh_wrt_scalefactor(scalefactor)

                if scalefactor < epsilon:
                    if gradient > 0:
                        #scalefactor = 0
                        #print('exiting because pos grad below 0')
                        break

                else:
                    step = -gamma * gradient

                scalefactor += step
                scalefactor = max(scalefactor, 0)
                #print('scalef: ',scalefactor)
                iters += 1
                if (
                    abs(step) < epsilon
                    or iters >= max_iter
                ):
                    break

            #print('arrived at ',scalefactor)
            if iters >= max_iter:
                print('exceeded gradient descent iteration limit!')
                print('arrived at ', scalefactor)
            #print('\n')
            scalefactor = max(0., min(1000./scaling_cascade_energy, scalefactor))

        elif SCALE_FACTOR_MINIMIZER is Minimizer.NEWTON:
            scalefactor = initial_scalefactor
            iters = 0 # iteration counter
            epsilon = 1e-2
            max_iter = 100
            while True:
                step = get_newton_step(scalefactor)
                if step == -1:
                    scalefactor = 0
                    break
                if scalefactor < epsilon and step > 0:
                    break
                scalefactor -= step
                #print(scalefactor)
                scalefactor = max(scalefactor, 0)
                iters += 1
                if abs(step) < epsilon or iters >= max_iter:
                    break

            #print('arrived at ',scalefactor, 'in iters = ', iters)
            #if iters >= max_iter:
            #    print('exceeded gradient descent iteration limit!')
            #    print('arrived at ',scalefactor)
            #print('\n')
            scalefactor = max(0., min(1000./scaling_cascade_energy, scalefactor))

        elif SCALE_FACTOR_MINIMIZER is Minimizer.BINARY_SEARCH:
            epsilon = 1e-2
            done = False
            first = 0.
            first_grad = get_grad_neg_llh_wrt_scalefactor(first)
            if first_grad > 0 or abs(first_grad) < epsilon:
                scalefactor = first
                done = True
                #print('trivial 0')
            if not done:
                last = 1000./scaling_cascade_energy
                last_grad = get_grad_neg_llh_wrt_scalefactor(last)
                if last_grad < 0 or abs(last_grad) < epsilon:
                    scalefactor = last
                    done = True
                    #print('trivial 1000')
            if not done:
                iters = 0
                while iters < 20:
                    iters += 1
                    test = (first + last)/2.
                    scalefactor = test
                    test_grad = get_grad_neg_llh_wrt_scalefactor(test)
                    #print('test :', test)
                    #print('test_grad :',test_grad)
                    if abs(test_grad) < epsilon:
                        break
                    elif test_grad < 0:
                        first = test
                    else:
                        last = test
            #print('found :',scalefactor)
            #print('\n')

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
    def get_llh_(
        generic_sources,
        pegleg_sources,
        scaling_sources,
        scaling_cascade_energy,
        event_hit_info,
        event_dom_info,
        pegleg_stepsize,
        dom_tables,
        dom_table_norms,
        t_indep_dom_tables,
        t_indep_dom_table_norms,
        tdi_tables,
    ): # pylint: disable=too-many-arguments
        """Compute log likelihood for hypothesis sources given an event.

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
        pegleg_stepsize : int > 0
            Number of pegleg sources to add each time around the pegleg loop; ignored if
            pegleg procedure is not performed (i.e., if there are no `pegleg_sources`)
        dom_tables
        dom_table_norms
        t_indep_dom_tables
        t_indep_dom_table_norms
        tdi_tables

        Returns
        -------
        llh : float
            Log-likelihood value at best pegleg hypo
        pegleg_stop_idx : int or float
            Pegleg stop index for `pegleg_sources` to obtain `llh`. If integer, .. ::
                pegleg_sources[:pegleg_stop_idx]
            but float can also be returned if `PEGLEG_LLH_CHOICE` is not LLHChoice.MAX.
            `pegleg_stop_idx` is designed to be fed to
            :func:`retro.hypo.discrete_muon_kernels.pegleg_eval`
        scalefactor : float
            Best scale factor for `scaling_sources` at best pegleg hypo

        """
        num_pegleg_sources = len(pegleg_sources)
        num_pegleg_steps = 1 + int(num_pegleg_sources / pegleg_stepsize)
        num_scaling_sources = len(scaling_sources)
        num_hits = len(event_hit_info)

        if num_scaling_sources > 0:
            # -- Storage for exp due to nominal (`scalefactor = 1`) scaling sources -- #
            nominal_scaling_t_indep_exp = 0.
            nominal_scaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

            nominal_scaling_t_indep_exp += pexp(
                sources=scaling_sources,
                sources_start=0,
                sources_stop=num_scaling_sources,
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                hit_exp=nominal_scaling_hit_exp,
                dom_tables=dom_tables,
                dom_table_norms=dom_table_norms,
                t_indep_dom_tables=t_indep_dom_tables,
                t_indep_dom_table_norms=t_indep_dom_table_norms,
                tdi_tables=tdi_tables,
            )

        # -- Storage for exp due to generic + pegleg (non-scaling) sources -- #

        nonscaling_t_indep_exp = 0.
        nonscaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        # Expectations for generic-only sources (i.e. pegleg=0 at this point)
        if len(generic_sources) > 0:
            nonscaling_t_indep_exp += pexp(
                sources=generic_sources,
                sources_start=0,
                sources_stop=len(generic_sources),
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                hit_exp=nonscaling_hit_exp,
                dom_tables=dom_tables,
                dom_table_norms=dom_table_norms,
                t_indep_dom_tables=t_indep_dom_tables,
                t_indep_dom_table_norms=t_indep_dom_table_norms,
                tdi_tables=tdi_tables,
            )

        if num_scaling_sources > 0:
            # Compute initial scalefactor & LLH for generic-only (no pegleg) sources
            scalefactor, llh = get_optimal_scalefactor(
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                nonscaling_hit_exp=nonscaling_hit_exp,
                nonscaling_t_indep_exp=nonscaling_t_indep_exp,
                nominal_scaling_hit_exp=nominal_scaling_hit_exp,
                nominal_scaling_t_indep_exp=nominal_scaling_t_indep_exp,
                initial_scalefactor=10.,
                scaling_cascade_energy=scaling_cascade_energy,
            )
        else:
            scalefactor = 0
            llh = simple_llh(
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                nonscaling_hit_exp=nonscaling_hit_exp,
                nonscaling_t_indep_exp=nonscaling_t_indep_exp,
            )

        if num_pegleg_sources == 0:
            # in this case we're done
            return (
                llh,
                0, # pegleg_stop_idx = 0: no pegleg sources
                scalefactor,
            )

        # -- Pegleg loop -- #

        if PEGLEG_SPACING is StepSpacing.LINEAR:
            pass
            #pegleg_steps = np.arange(num_pegleg_sources)
            #n_pegleg_steps = len(pegleg_steps)
        elif PEGLEG_SPACING is StepSpacing.LOG:
            raise NotImplementedError(
                'Only ``PEGLEG_SPACING = StepSpacing.LINEAR`` is implemented'
            )
            #logstep = np.log(num_pegleg_sources) / 300
            #x = -1e-8
            #logspace = np.zeros(shape=301, dtype=np.int32)
            #for i in range(len(logspace)):
            #    logspace[i] = np.int32(np.exp(x))
            #    x+= logstep
            #pegleg_steps = np.unique(logspace)
            #assert pegleg_steps[0] == 0
            #n_pegleg_steps = len(pegleg_steps)
        else:
            raise ValueError('Unknown `PEGLEG_SPACING`')

        # -- Loop initialization -- #

        num_llhs = num_pegleg_steps + 1
        llhs = np.full(shape=num_llhs, fill_value=-np.inf, dtype=np.float64)
        llhs[0] = llh

        scalefactors = np.zeros(shape=num_llhs, dtype=np.float64)
        scalefactors[0] = scalefactor

        best_llh = llh
        previous_llh = best_llh - 100
        pegleg_max_llh_step = 0
        getting_worse_counter = 0

        for pegleg_step in range(1, num_pegleg_steps):
            pegleg_stop_idx = pegleg_step * pegleg_stepsize
            pegleg_start_idx = pegleg_stop_idx - pegleg_stepsize

            # Add to expectations by including another "batch" or segment of pegleg
            # sources
            nonscaling_t_indep_exp += pexp(
                sources=pegleg_sources,
                sources_start=pegleg_start_idx,
                sources_stop=pegleg_stop_idx,
                event_dom_info=event_dom_info,
                event_hit_info=event_hit_info,
                hit_exp=nonscaling_hit_exp,
                dom_tables=dom_tables,
                dom_table_norms=dom_table_norms,
                t_indep_dom_tables=t_indep_dom_tables,
                t_indep_dom_table_norms=t_indep_dom_table_norms,
                tdi_tables=tdi_tables,
            )

            if num_scaling_sources > 0:
                # Find optimal scalefactor at this pegleg step
                scalefactor, llh = get_optimal_scalefactor(
                    event_dom_info=event_dom_info,
                    event_hit_info=event_hit_info,
                    nonscaling_hit_exp=nonscaling_hit_exp,
                    nonscaling_t_indep_exp=nonscaling_t_indep_exp,
                    nominal_scaling_hit_exp=nominal_scaling_hit_exp,
                    nominal_scaling_t_indep_exp=nominal_scaling_t_indep_exp,
                    initial_scalefactor=scalefactor,
                    scaling_cascade_energy=scaling_cascade_energy,
                )
            else:
                scalefactor = 0
                llh = simple_llh(
                    event_dom_info=event_dom_info,
                    event_hit_info=event_hit_info,
                    nonscaling_hit_exp=nonscaling_hit_exp,
                    nonscaling_t_indep_exp=nonscaling_t_indep_exp,
                )

            # Store this pegleg step's llh and best scalefactor
            llhs[pegleg_step] = llh
            scalefactors[pegleg_step] = scalefactor

            if llh > best_llh:
                best_llh = llh
                pegleg_max_llh_step = pegleg_step
                getting_worse_counter = 0
            elif llh < previous_llh:
                getting_worse_counter += 1
            else:
                getting_worse_counter -= 1
            previous_llh = llh

            # break condition
            if getting_worse_counter > 100: # 10?
                #for idx in range(pegleg_idx+1,n_pegleg_steps):
                #    # fill up with bad llhs. just to make sure they're not used
                #    llhs[idx] = best_llh - 100
                #print('break at step ',pegleg_idx)
                break

        if PEGLEG_LLH_CHOICE is LLHChoice.MAX:
            return (
                llhs[pegleg_max_llh_step],
                pegleg_max_llh_step * pegleg_stepsize,
                scalefactors[pegleg_max_llh_step],
            )

        elif PEGLEG_LLH_CHOICE is LLHChoice.MEAN:
            max_llh = llhs[pegleg_max_llh_step]
            total_llh_above_thresh = 0.
            total_idx_above_thresh = 0.
            total_scalefactor_above_thresh = 0.
            counter = 0
            for pegleg_step in range(num_llhs):
                if llhs[pegleg_step] > max_llh - PEGLEG_BEST_DELTA_LLH_THRESHOLD:
                    counter += 1
                    total_llh_above_thresh += llhs[pegleg_step]
                    total_idx_above_thresh += pegleg_step * pegleg_stepsize
                    total_scalefactor_above_thresh += scalefactors[pegleg_step]

            return (
                total_llh_above_thresh / counter,
                total_idx_above_thresh / counter,
                total_scalefactor_above_thresh / counter,
            )

        elif PEGLEG_LLH_CHOICE is LLHChoice.MEDIAN:
            raise NotImplementedError(
                '``PEGLEG_LLH_CHOICE == LLHChoice.MEDIAN`` not implemented.'
            )
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

            #print(llhs[:10])
            #print(scalefactors[:10])
            #print(pegleg_steps[:10])

        else:
            raise ValueError('Unknown `PEGLEG_LLH_CHOICE`')

    # Note: numba fails w/ TDI tables if this is set to be jit-compiled (why?)
    def get_llh(
        generic_sources,
        pegleg_sources,
        scaling_sources,
        scaling_cascade_energy,
        event_hit_info,
        event_dom_info,
        pegleg_stepsize,
    ):
        """Compute log likelihood for hypothesis sources given an event.

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
        pegleg_stepsize : int > 0
            Number of pegleg sources to add each time around the pegleg loop; ignored if
            pegleg procedure is not performed (i.e., if there are no `pegleg_sources`)

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
        return get_llh_(
            generic_sources=generic_sources,
            pegleg_sources=pegleg_sources,
            scaling_sources=scaling_sources,
            scaling_cascade_energy=scaling_cascade_energy,
            event_hit_info=event_hit_info,
            event_dom_info=event_dom_info,
            pegleg_stepsize=pegleg_stepsize,
            dom_tables=dom_tables,
            dom_table_norms=dom_table_norms,
            t_indep_dom_tables=t_indep_dom_tables,
            t_indep_dom_table_norms=t_indep_dom_table_norms,
            tdi_tables=tdi_tables,
        )
    get_llh.__doc__ = get_llh_.__doc__

    return get_llh
