#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating, too-many-locals

"""
Reco class for performing reconstructions
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "METHODS",
    "CRS_STOP_FLAGS",
    "REPORT_AFTER",
    "CART_DIMS",
    "Reco",
    "get_multinest_meta",
    "main",
]

__author__ = "J.L. Lanfranchi, P. Eller"
__license__ = """Copyright 2017-2018 Justin L. Lanfranchi and Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


from argparse import ArgumentParser
from collections import OrderedDict
from os.path import abspath, dirname, isdir, isfile, join
from shutil import rmtree
import sys
from tempfile import mkdtemp
import time
import traceback

import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import __version__, MissingOrInvalidPrefitError, init_obj
from retro.hypo.discrete_cascade_kernels import SCALING_CASCADE_ENERGY
from retro.hypo.discrete_muon_kernels import pegleg_eval
from retro.priors import (
    EXT_IC,
    PRI_COSINE,
    PRI_TIME_RANGE,
    PRI_UNIFORM,
    PRISPEC_OSCNEXT_PREFIT_TIGHT,
    PRISPEC_OSCNEXT_CRS_MN,
    Bound,
    get_prior_func,
)
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T, SPHER_T, FitStatus
from retro.tables.pexp_5d import generate_pexp_and_llh_functions
from retro.utils.geom import (
    rotate_points,
    add_vectors,
    fill_from_spher,
    fill_from_cart,
    reflect,
)
from retro.utils.get_arg_names import get_arg_names
from retro.utils.misc import sort_dict
from retro.utils.stats import estimate_from_llhp


LLH_FUDGE_SUMMAND = -1000

METHODS = set(
    [
        "multinest",
        "crs",
        "crs_prefit",
        "mn8d",
        "stopping_atm_muon_crs",
        "dn8d",
        "nlopt",
        "scipy",
        "skopt",
        "experimental_trackfit",
        "fast",
        "test",
        "truth",
    ]
)

CRS_STOP_FLAGS = {
    0: "max iterations reached",
    1: "llh stddev below threshold",
    2: "no improvement",
    3: "vertex stddev below threshold",
}

# TODO: make following args to `__init__` or `run`
REPORT_AFTER = 100

CART_DIMS = ("x", "y", "z", "time")


class Reco(object):
    """
    Setup tables, get events, run reconstructons on them, and optionally store
    results to disk.

    Note that "recipes" for different reconstructions are defined in the
    `Reco.run` method.

    Parameters
    ----------
    events_kw, dom_tables_kw, tdi_tables_kw : mappings
        As returned by `retro.init_obj.parse_args`

    debug : bool

    """

    def __init__(
        self,
        events_kw,
        dom_tables_kw,
        tdi_tables_kw,
        debug=False,
    ):
        self.debug = bool(debug)

        # We don't want to specify 'recos' so that new recos are automatically
        # found by `init_obj.get_events` function
        events_kw.pop("recos", None)
        self.events_kw = sort_dict(events_kw)

        self.dom_tables_kw = sort_dict(dom_tables_kw)
        self.tdi_tables_kw = sort_dict(tdi_tables_kw)
        self.attrs = OrderedDict(
            [
                ("events_kw", self.events_kw),
                ("dom_tables_kw", self.dom_tables_kw),
                ("tdi_tables_kw", self.tdi_tables_kw),
            ]
        )

        # Replace None values for `start` and `step` for fewer branches in
        # subsequent logic (i.e., these will always be integers)
        self.events_start = 0 if events_kw["start"] is None else events_kw["start"]
        self.events_step = 1 if events_kw["step"] is None else events_kw["step"]
        # Nothing we can do about None for `stop` since we don't know how many
        # events there are in total.
        self.events_stop = events_kw["stop"]

        self.dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
        self.tdi_tables, self.tdi_metas = init_obj.setup_tdi_tables(**tdi_tables_kw)
        self.pexp, self.get_llh, _ = generate_pexp_and_llh_functions(
            dom_tables=self.dom_tables,
            tdi_tables=self.tdi_tables,
            tdi_metas=self.tdi_metas,
        )
        self.event = None
        self.event_counter = 0
        self.successful_reco_counter = OrderedDict()
        self.hypo_handler = None
        self.prior = None
        self.priors_used = None
        self.loglike = None
        self.n_params = None
        self.n_opt_params = None

    @property
    def events(self):
        """Iterator over events.

        Yields
        ------
        event : OrderedDict
            Each event has attribute `.meta` (see retro.init_obj.get_events)
            but this gets populated with additional information within this
            method: "recodir" and "prefix".

        """
        # do initialization here so any new recos are automatically detected
        events = init_obj.get_events(**self.events_kw)
        for event in events:
            self.event = event
            self.event.meta["prefix"] = join(
                event.meta["events_root"],
                "recos",
                "evt{}.".format(event.meta["event_idx"]),
            )
            self.event_counter += 1
            print(
                'Reconstructing event #{} (index {} in dir "{}")'.format(
                    self.event_counter,
                    self.event.meta["event_idx"],
                    self.event.meta["events_root"],
                )
            )
            yield self.event

    def setup_hypo(self, **kwargs):
        """Setup hypothesis and record `n_params` and `n_opt_params`
        corresponding to the hypothesis.

        Parameters
        ----------
        **kwargs
            Passed to `retro.init_obj.setup_discrete_hypo`

        """
        self.hypo_handler = init_obj.setup_discrete_hypo(**kwargs)
        self.n_params = self.hypo_handler.n_params
        self.n_opt_params = self.hypo_handler.n_opt_params

    def _reco_event(self, method, save_llhp):
        """Recipes for performing different kinds of reconstructions.

        Parameters
        ----------
        method : str
        save_llhp : bool

        """
        # simple 1-stage recos
        if method in ("multinest", "test", "truth", "crs", "scipy", "nlopt", "skopt"):
            self.setup_hypo(
                cascade_kernel="scaling_aligned_one_dim",
                track_kernel="pegleg",
                track_time_step=1.0,
            )

            self.generate_prior_method(**PRISPEC_OSCNEXT_PREFIT_TIGHT)

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []
            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            if method == "test":
                run_info, fit_meta = self.run_test(seed=0)
            if method == "truth":
                run_info, fit_meta = self.run_with_truth()
            elif method == "crs":
                run_info, fit_meta = self.run_crs(
                    n_live=250,
                    max_iter=20000,
                    max_noimprovement=5000,
                    min_llh_std=0.1,
                    min_vertex_std=dict(x=1, y=1, z=1, time=3),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                )
            elif method == "multinest":
                run_info, fit_meta = self.run_multinest(
                    importance_sampling=True,
                    max_modes=1,
                    const_eff=True,
                    n_live=160,
                    evidence_tol=0.5,
                    sampling_eff=0.3,
                    max_iter=10000,
                    seed=0,
                )
            elif method == "scipy":
                run_info, fit_meta = self.run_scipy(
                    method="differential_evolution", eps=0.02
                )
            elif method == "nlopt":
                run_info, fit_meta = self.run_nlopt()
            elif method == "skopt":
                run_info, fit_meta = self.run_skopt()

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )
            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=True,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        elif method == "fast":
            self.setup_hypo(
                cascade_kernel="scaling_aligned_point_ckv",
                track_kernel="pegleg",
                track_time_step=3.0,
            )

            self.generate_prior_method(**PRISPEC_OSCNEXT_PREFIT_TIGHT)

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []

            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            run_info, fit_meta = self.run_crs(
                n_live=160,
                max_iter=10000,
                max_noimprovement=1000,
                min_llh_std=0.5,
                min_vertex_std=dict(x=5, y=5, z=5, time=15),
                use_priors=False,
                use_sobol=True,
                seed=0,
            )

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )

            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=False,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        elif method == "stopping_atm_muon_crs":
            self.setup_hypo(
                track_kernel="stopping_table_energy_loss", track_time_step=3.0
            )

            self.generate_prior_method(
                x=dict(kind=PRI_UNIFORM, extents=EXT_IC["x"]),
                y=dict(kind=PRI_UNIFORM, extents=EXT_IC["y"]),
                z=dict(kind=PRI_UNIFORM, extents=EXT_IC["z"]),
                time=dict(kind=PRI_TIME_RANGE),
                track_zenith=dict(
                    kind=PRI_COSINE, extents=((0, Bound.ABS), (np.pi / 2, Bound.ABS))
                ),
            )

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []

            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            run_info, fit_meta = self.run_crs(
                n_live=160,
                max_iter=10000,
                max_noimprovement=1000,
                min_llh_std=0.,
                min_vertex_std=dict(x=5, y=5, z=4, time=20),
                use_priors=False,
                use_sobol=True,
                seed=0,
            )

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )

            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=False,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        elif method == "crs_prefit":
            self.setup_hypo(
                cascade_kernel="scaling_aligned_point_ckv",
                track_kernel="pegleg",
                track_time_step=3.0,
            )

            self.generate_prior_method(**PRISPEC_OSCNEXT_PREFIT_TIGHT)

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []

            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            run_info, fit_meta = self.run_crs(
                n_live=160,
                max_iter=10000,
                max_noimprovement=1000,
                min_llh_std=0.5,
                min_vertex_std=dict(x=5, y=5, z=4, time=20),
                use_priors=False,
                use_sobol=True,
                seed=0,
            )

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )

            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=False,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        elif method == "mn8d":
            self.setup_hypo(
                cascade_kernel="scaling_aligned_one_dim",
                track_kernel="pegleg",
                track_time_step=1.0,
            )

            self.generate_prior_method(**PRISPEC_OSCNEXT_CRS_MN)

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []

            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            run_info, fit_meta = self.run_multinest(
                importance_sampling=True,
                max_modes=1,
                const_eff=True,
                n_live=250,
                evidence_tol=0.02,
                sampling_eff=0.5,
                max_iter=10000,
                seed=0,
            )

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )

            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=True,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        elif method == "dn8d":
            self.setup_hypo(
                cascade_kernel="scaling_aligned_one_dim",
                track_kernel="pegleg",
                track_time_step=1.0,
            )

            self.generate_prior_method(return_cube=True, **PRISPEC_OSCNEXT_CRS_MN)
            #self.generate_prior_method(return_cube=True, **PRISPEC_OSCNEXT_PREFIT_TIGHT)

            param_values = []
            log_likelihoods = []
            aux_values = []
            t_start = []

            self.generate_loglike_method(
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                aux_values=aux_values,
                t_start=t_start,
            )

            run_info, fit_meta = self.run_dynesty(
                n_live=100,
                maxiter=2000,
                maxcall=10000,
                dlogz=0.1,
            )

            llhp = self.make_llhp(
                method=method,
                log_likelihoods=log_likelihoods,
                param_values=param_values,
                aux_values=aux_values,
                save=save_llhp,
            )

            self.make_estimate(
                method=method,
                llhp=llhp,
                remove_priors=True,
                run_info=run_info,
                fit_meta=fit_meta,
            )

        else:
            raise ValueError("Unknown `Method` {}".format(method))

    def _print_non_fatal_exception(self, method):
        """Print to stderr a detailed message about a failure in reconstruction
        that is non-fatal.

        Parameters
        ----------
        method : str
            The name of the function, e.g. "run_crs" or "run_multinest"

        """
        id_fields = ["run_id", "sub_run_id", "event_id", "sub_event_id"]
        id_str = ", ".join(
            "{} {}".format(f, self.event["header"][f]) for f in id_fields
        )
        sys.stderr.write(
            "ERROR! Reco function {method} failed on event index {idx} ({id_str}) in"
            ' path "{fpath}". Recording reco failure and continuing to next event)'
            "\n{tbk}\n".format(
                method=method,
                idx=self.event.meta["event_idx"],
                fpath=self.event.meta["events_root"],
                id_str=id_str,
                tbk="".join(traceback.format_exc()),
            )
        )

    def run(
        self,
        methods,
        redo_failed=False,
        redo_all=False,
        save_llhp=False,
        filter=None,  # pylint: disable=redefined-builtin
    ):
        """Run reconstruction(s) on events.

        Parameters
        ----------
        methods : string or iterable thereof
            Each must be one of `METHODS`

        redo_failed : bool, optional
            If `True`, reconstruct each event that either hasn't been
            reconstructed with each method (as usual), but also re-reconstruct
            events that have `fit_status` indicating a failure (i.e., all
            events will be reconstructed using a given method unless they have
            for that method `fit_status == FitStatus.OK`). Default is False.

        redo_all : bool, optional
            If `True`, reconstruct all events with all `methods`, regardless if
            they have been reconstructed with these methods previously.

        save_llhp : bool, optional
            Save likelihood values & corresponding parameter values within a
            LLH range of the max LLH (this takes up a lot of disk space and
            creats a lot of files; use with caution if running jobs en masse)

        filter : str or None
            Filter to apply for selecting events to reconstruct. String is
            passed through `eval` and must produce a scalar value interpretable
            via `bool(eval(filter))`. Current event is accessible via the name
            `event` and numpy is named `np`. E.g. .. ::

                filter="event['header']['L5_oscNext_bool']"

        """
        start_time = time.time()
        if isinstance(methods, string_types):
            methods = [methods]

        for method in methods:
            if method not in METHODS:
                raise ValueError(
                    'Unrecognized `method` "{}"; must be one of {}'.format(
                        method, METHODS
                    )
                )

        if len(set(methods)) != len(methods):
            raise ValueError("Same reco specified multiple times")

        if filter is not None:
            assert isinstance(filter, string_types)
            filter = filter.strip()
            print("filter: '{}'".format(filter))

        print("Running {} reconstruction(s) on all specified events".format(methods))

        self.successful_reco_counter = OrderedDict([(method, 0) for method in methods])

        for event in self.events:  # pylint: disable=unused-variable
            if filter and not eval(filter):  # pylint: disable=eval-used
                print(
                    "filter evaluates to False; skipping event #{} (index {})".format(
                        self.event_counter, event.meta["event_idx"]
                    )
                )
                continue

            for method in methods:
                estimate_outf = join(
                    self.event.meta["events_root"],
                    "recos",
                    "retro_{}.npy".format(method),
                )
                if isfile(estimate_outf):
                    estimates = np.load(estimate_outf, mmap_mode="r")
                    fit_status = estimates[self.event.meta["event_idx"]]["fit_status"]
                    if fit_status != FitStatus.NotSet:
                        if redo_all:
                            print(
                                'Method "{}" already run on event; redoing'.format(
                                    method
                                )
                            )
                        elif redo_failed and fit_status != FitStatus.OK:
                            print(
                                'Method "{}" already run on event but failed'
                                " previously; retrying".format(method)
                            )
                        else:
                            print(
                                'Method "{}" already run on event; skipping'.format(
                                    method
                                )
                            )
                            continue

                print('Running "{}" reconstruction'.format(method))
                try:
                    self._reco_event(method=method, save_llhp=save_llhp)
                except MissingOrInvalidPrefitError as error:
                    print(
                        'ERROR: event idx {}, reco method {}: "{}"; ignoring'
                        " and moving to next event".format(
                            self.event.meta["event_idx"], method, error
                        )
                    )
                    estimates[self.event.meta["event_idx"]]["fit_status"] = (
                        FitStatus.MissingSeed
                    )
                else:
                    self.successful_reco_counter[method] += 1

        print("Total run time is {:.3f} s".format(time.time() - start_time))

    def generate_prior_method(self, return_cube=False, **kwargs):
        """Generate the prior transform method `self.prior` and info
        `self.priors_used` for a given event. Optionally, plots the priors to
        current working directory if `self.debug` is True.

        Call, e.g., via:

            self.generate_prior_method(
                x=dict(
                    kind=PRI_OSCNEXT_L5_V1_PREFIT,
                    extents=((-100, Bounds.REL),
                    (100, Bounds.REL)),
                ),
                y=dict(
                    kind=PRI_OSCNEXT_L5_V1_PREFIT,
                    extents=((-100, Bounds.REL),
                    (100, Bounds.REL)),
                ),
                z=dict(
                    kind=PRI_OSCNEXT_L5_V1_PREFIT,
                    extents=((-50, Bounds.REL),
                    (50, Bounds.REL)),
                ),
                time=dict(
                    kind=PRI_OSCNEXT_L5_V1_PREFIT,
                    extents=((-1000, Bounds.REL),
                    (1000, Bounds.REL)),
                ),
                azimuth=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
                zenith=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
            )

        Parameters
        ----------
        return_cube : bool
            if true, explicitly return the transformed cube
        **kwargs
            Prior definitions; anything unspecified falls back to a default
            (since all params must have priors, including ranges, for e.g.
            MultiNest and CRS).

        """
        prior_funcs = []
        self.priors_used = OrderedDict()

        miscellany = []
        for dim_num, dim_name in enumerate(self.hypo_handler.opt_param_names):
            spec = kwargs.get(dim_name, {})
            prior_func, prior_def, misc = get_prior_func(
                dim_num=dim_num, dim_name=dim_name, event=self.event, **spec
            )
            prior_funcs.append(prior_func)
            self.priors_used[dim_name] = prior_def
            miscellany.append(misc)

        def prior(cube, ndim=None, nparams=None):  # pylint: disable=unused-argument, inconsistent-return-statements
            """Apply `prior_funcs` to the hypercube to map values from the unit
            hypercube onto values in the physical parameter space.

            The result overwrites the values in `cube`.

            Parameters
            ----------
            cube
            ndim
            nparams

            """
            for prior_func in prior_funcs:
                prior_func(cube)

            if return_cube:
                return cube

        self.prior = prior

        if self.debug:
            # -- Plot priors and save to png's in current dir -- #
            import matplotlib as mpl
            mpl.use("agg", warn=False)
            import matplotlib.pyplot as plt

            n_opt_params = len(self.hypo_handler.opt_param_names)
            rand = np.random.RandomState(0)
            cube = rand.rand(n_opt_params, int(1e5))
            self.prior(cube)

            nx = int(np.ceil(np.sqrt(n_opt_params)))
            ny = int(np.ceil(n_opt_params / nx))
            fig, axes = plt.subplots(ny, nx, figsize=(6 * nx, 4 * ny))
            axit = iter(axes.flat)
            for dim_num, dim_name in enumerate(self.hypo_handler.opt_param_names):
                ax = next(axit)
                ax.hist(cube[dim_num], bins=100)
                misc = miscellany[dim_num]
                if "reco_val" in misc:
                    ylim = ax.get_ylim()
                    ax.plot([misc["reco_val"]] * 2, ylim, "k--", lw=1)
                    ax.set_ylim(ylim)

                misc_strs = []
                if "reco" in misc:
                    misc_strs.append(misc["reco"])
                if "reco_val" in misc:
                    misc_strs.append("{:.2f}".format(misc["reco_val"]))
                if (
                    "split_by_reco_param" in misc
                    and misc["split_by_reco_param"] is not None
                ):
                    misc_strs.append(
                        "split by {} = {:.2f}".format(
                            misc["split_by_reco_param"], misc["split_val"]
                        )
                    )
                misc_str = ", ".join(misc_strs)
                ax.set_title(
                    "{}: {} {}".format(
                        dim_name, self.priors_used[dim_name][0], misc_str
                    )
                )
            for ax in axit:
                ax.axis("off")
            fig.tight_layout()
            plt_fpath_base = self.event.meta["prefix"] + "priors"
            fig.savefig(plt_fpath_base + ".png", dpi=120)

    def generate_loglike_method(
        self, param_values, log_likelihoods, aux_values, t_start
    ):
        """Generate the LLH callback method `self.loglike` for a given event.

        Parameters
        ----------
        param_values : list
        log_likelihoods : list
        aux_values : list
        t_start : list
            Needs to be a list for `t_start` to be passed by reference (and
            therefore universally accessible within all methods that require
            knowing `t_start`).

        """
        # -- Variables to be captured by `loglike` closure -- #

        all_param_names = self.hypo_handler.all_param_names
        opt_param_names = self.hypo_handler.opt_param_names
        n_opt_params = self.hypo_handler.n_opt_params
        fixed_params = self.hypo_handler.fixed_params
        event = self.event
        hits = event["hits"]
        hits_indexer = event["hits_indexer"]
        hypo_handler = self.hypo_handler
        pegleg_muon_dt = hypo_handler.pegleg_kernel_kwargs.get("dt")
        pegleg_muon_const_e_loss = False
        dom_info = self.dom_tables.dom_info
        sd_idx_table_indexer = self.dom_tables.sd_idx_table_indexer
        if "truth" in event:
            truth = event["truth"]
            truth_info = OrderedDict(
                [
                    ("x", truth["x"]),
                    ("y", truth["y"]),
                    ("z", truth["z"]),
                    ("time", truth["time"]),
                    ("zenith", truth["zenith"]),
                    ("azimuth", truth["azimuth"]),
                    ("track_azimuth", truth["track_azimuth"]),
                    ("track_zenith", truth["track_zenith"]),
                    ("track_energy", truth["track_energy"]),
                    ("energy", truth["energy"]),
                    ("cascade_energy", truth['total_cascade_energy']),
                ]
            )
            optional = [
                ("cscd_az", "total_cascade_azimuth"),
                ("cscd_zen", "total_cascade_zenith"),
                ("cscd_em_equiv_en", "total_cascade_em_equiv_energy"),
            ]
            for label, key in optional:
                if key in truth:
                    truth_info[label] = truth[key]
        else:
            truth_info = None

        num_operational_doms = np.sum(dom_info["operational"])

        # Array containing only DOMs operational during the event & info
        # relevant to the hits these DOMs got (if any)
        event_dom_info = np.zeros(shape=num_operational_doms, dtype=EVT_DOM_INFO_T)

        # Array containing all relevant hit info for the event, including a
        # pointer back to the index of the DOM in the `event_dom_info` array
        event_hit_info = np.zeros(shape=hits.size, dtype=EVT_HIT_INFO_T)

        # Copy 'time' and 'charge' over directly; add 'event_dom_idx' below
        event_hit_info[["time", "charge"]] = hits[["time", "charge"]]

        # Must be a list, not tuple:
        copy_fields = [
            "sd_idx",
            "x",
            "y",
            "z",
            "quantum_efficiency",
            "noise_rate_per_ns",
        ]

        print("all noise rate %.5f" % np.nansum(dom_info["noise_rate_per_ns"]))
        print(
            "DOMs with zero or NaN noise %i"
            % np.count_nonzero(
                np.isnan(dom_info["noise_rate_per_ns"])
                | (dom_info["noise_rate_per_ns"] == 0)
            )
        )

        # Fill `event_{hit,dom}_info` arrays only for operational DOMs
        for dom_idx, this_dom_info in enumerate(dom_info[dom_info["operational"]]):
            this_event_dom_info = event_dom_info[dom_idx : dom_idx + 1]
            this_event_dom_info[copy_fields] = this_dom_info[copy_fields]
            sd_idx = this_dom_info["sd_idx"]
            this_event_dom_info["table_idx"] = sd_idx_table_indexer[sd_idx]

            # Copy any hit info from `hits_indexer` and total charge from
            # `hits` into `event_hit_info` and `event_dom_info` arrays
            this_hits_indexer = hits_indexer[hits_indexer["sd_idx"] == sd_idx]
            if len(this_hits_indexer) == 0:
                this_event_dom_info["hits_start_idx"] = 0
                this_event_dom_info["hits_stop_idx"] = 0
                this_event_dom_info["total_observed_charge"] = 0
                continue

            start = this_hits_indexer[0]["offset"]
            stop = start + this_hits_indexer[0]["num"]
            event_hit_info[start:stop]["event_dom_idx"] = dom_idx
            this_event_dom_info["hits_start_idx"] = start
            this_event_dom_info["hits_stop_idx"] = stop
            this_event_dom_info["total_observed_charge"] = np.sum(
                hits[start:stop]["charge"]
            )

        print("this evt. noise rate %.5f" % np.sum(event_dom_info["noise_rate_per_ns"]))
        print(
            "DOMs with zero noise: %i"
            % np.sum(event_dom_info["noise_rate_per_ns"] == 0)
        )
        # settings those to minimum noise
        noise = event_dom_info["noise_rate_per_ns"]
        mask = noise < 1e-7
        noise[mask] = 1e-7
        print("this evt. noise rate %.5f" % np.sum(event_dom_info["noise_rate_per_ns"]))
        print(
            "DOMs with zero noise: %i"
            % np.sum(event_dom_info["noise_rate_per_ns"] == 0)
        )
        print("min noise: ", np.min(noise))
        print("mean noise: ", np.mean(noise))

        assert np.sum(event_dom_info["quantum_efficiency"] <= 0) == 0, "negative QE"
        assert np.sum(event_dom_info["total_observed_charge"]) > 0, "no charge"
        assert np.isfinite(
            np.sum(event_dom_info["total_observed_charge"])
        ), "non-finite charge"

        def loglike(cube, ndim=None, nparams=None):  # pylint: disable=unused-argument
            """Get log likelihood values.

            Defined as a closure to capture particulars of the event and priors
            without having to pass these as parameters to the function.

            Note that this is called _after_ `prior` has been called, so `cube`
            already contains the parameter values scaled to be in their
            physical ranges.

            Parameters
            ----------
            cube
            ndim : int, optional
            nparams : int, optional

            Returns
            -------
            llh : float

            """
            t0 = time.time()
            if len(t_start) == 0:
                t_start.append(time.time())

            hypo = OrderedDict(list(zip(opt_param_names, cube)))

            generic_sources = hypo_handler.get_generic_sources(hypo)
            pegleg_sources = hypo_handler.get_pegleg_sources(hypo)
            scaling_sources = hypo_handler.get_scaling_sources(hypo)

            get_llh_retval = self.get_llh(
                generic_sources=generic_sources,
                pegleg_sources=pegleg_sources,
                scaling_sources=scaling_sources,
                event_hit_info=event_hit_info,
                event_dom_info=event_dom_info,
                pegleg_stepsize=1,
            )

            llh, pegleg_idx, scalefactor = get_llh_retval[:3]
            llh += LLH_FUDGE_SUMMAND
            aux_values.append(get_llh_retval[3:])

            assert np.isfinite(llh), "LLH not finite: {}".format(llh)
            # assert llh <= 0, "LLH positive: {}".format(llh)

            additional_results = []

            if self.hypo_handler.pegleg_kernel:
                pegleg_result = pegleg_eval(
                    pegleg_idx=pegleg_idx,
                    dt=pegleg_muon_dt,
                    const_e_loss=pegleg_muon_const_e_loss,
                    mmc=True,
                )
                additional_results.append(pegleg_result)

            if self.hypo_handler.scaling_kernel:
                additional_results.append(scalefactor * SCALING_CASCADE_ENERGY)

            result = (
                tuple(cube[:n_opt_params])
                + tuple(fixed_params.values())
                + tuple(additional_results)
            )
            param_values.append(result)

            log_likelihoods.append(llh)
            n_calls = len(log_likelihoods)
            t1 = time.time()

            if n_calls % REPORT_AFTER == 0:
                print("")
                if truth_info:
                    msg = "truth:                "
                    for key, val in zip(all_param_names, result):
                        try:
                            msg += " %s=%.1f" % (key, truth_info[key])
                        except KeyError:
                            pass
                    print(msg)
                t_now = time.time()
                best_idx = np.argmax(log_likelihoods)
                best_llh = log_likelihoods[best_idx]
                best_p = param_values[best_idx]
                msg = "best llh = {:.3f} @ ".format(best_llh)
                for key, val in zip(all_param_names, best_p):
                    msg += " %s=%.1f" % (key, val)
                print(msg)
                msg = "this llh = {:.3f} @ ".format(llh)
                for key, val in zip(all_param_names, result):
                    msg += " %s=%.1f" % (key, val)
                print(msg)
                print("{} LLH computed".format(n_calls))
                print(
                    "avg time per llh: {:.3f} ms".format(
                        (t_now - t_start[0]) / n_calls * 1000
                    )
                )
                print("this llh took:    {:.3f} ms".format((t1 - t0) * 1000))
                print("")

            return llh

        self.loglike = loglike

    def make_llhp(self, method, log_likelihoods, param_values, aux_values, save):
        """Create a structured numpy array containing the reco information;
        also add derived dimensions, and optionally save to disk.

        Parameters
        ----------
        method : str

        log_likelihoods : array

        param_values : array

        aux_values : array

        save : bool

        Returns
        -------
        llhp : length-n_llhp array of dtype llhp_t
            Note that llhp_t is derived from the defined parameter names.

        """
        # Setup LLHP dtype
        dim_names = list(self.hypo_handler.all_param_names)

        # add derived quantities
        derived_dim_names = ["energy", "azimuth", "zenith"]
        if "cascade_d_zenith" in dim_names and "cascade_d_azimuth" in dim_names:
            derived_dim_names += ["cascade_zenith", "cascade_azimuth"]

        aux_names = ["zero_dllh", "lower_dllh", "upper_dllh"]

        all_dim_names = dim_names + derived_dim_names + aux_names

        llhp_t = np.dtype([(field, np.float32) for field in ["llh"] + all_dim_names])

        # dump
        llhp = np.zeros(shape=len(param_values), dtype=llhp_t)
        llhp["llh"] = log_likelihoods
        llhp[dim_names] = param_values

        llhp[aux_names] = aux_values

        # create derived dimensions
        if "energy" in derived_dim_names:
            if "track_energy" in dim_names:
                llhp["energy"] += llhp["track_energy"]
            if "cascade_energy" in dim_names:
                llhp["energy"] += llhp["cascade_energy"]

        if "cascade_d_zenith" in dim_names and "cascade_d_azimuth" in dim_names:
            # create cascade angles from delta angles
            rotate_points(
                p_theta=llhp["cascade_d_zenith"],
                p_phi=llhp["cascade_d_azimuth"],
                rot_theta=llhp["track_zenith"],
                rot_phi=llhp["track_azimuth"],
                q_theta=llhp["cascade_zenith"],
                q_phi=llhp["cascade_azimuth"],
            )

        if "track_zenith" in all_dim_names and "track_azimuth" in all_dim_names:
            if "cascade_zenith" in all_dim_names and "cascade_azimuth" in all_dim_names:
                # this resulting radius we won't need, but need to supply an array to
                # the function
                r_out = np.empty(shape=llhp.shape, dtype=np.float32)
                # combine angles:
                add_vectors(
                    r1=llhp["track_energy"],
                    theta1=llhp["track_zenith"],
                    phi1=llhp["track_azimuth"],
                    r2=llhp["cascade_energy"],
                    theta2=llhp["cascade_zenith"],
                    phi2=llhp["cascade_azimuth"],
                    r3=r_out,
                    theta3=llhp["zenith"],
                    phi3=llhp["azimuth"],
                )
            else:
                # in this case there is no cascade angles
                llhp["zenith"] = llhp["track_zenith"]
                llhp["azimuth"] = llhp["track_azimuth"]

        elif "cascade_zenith" in all_dim_names and "cascade_azimuth" in all_dim_names:
            # in this case there are no track angles
            llhp["zenith"] = llhp["cascade_zenith"]
            llhp["azimuth"] = llhp["cascade_azimuth"]

        if save:
            fname = "retro_{}.llhp".format(method)
            # NOTE: since each array can have different length and numpy
            # doesn't handle "ragged" arrays nicely, forcing each llhp to be
            # saved to its own file
            llhp_outf = "{}{}.npy".format(self.event.meta["prefix"], fname)
            llh = llhp["llh"]
            cut_llhp = llhp[llh > np.max(llh) - 30]
            print(
                'Saving llhp within 30 LLH of max ({} llhp) to "{}"'.format(
                    len(cut_llhp), llhp_outf
                )
            )
            np.save(llhp_outf, cut_llhp)

        return llhp

    def make_estimate(
        self, method, llhp, remove_priors, run_info=None, fit_meta=None
    ):
        """Create estimate from llhp, attach result to `self.event`, and save to disk.

        Parameters
        ----------
        method : str
            Reconstruction method used
        llhp : length-n_llhp array of dtype llhp_t
        remove_priors : bool
            Remove effect of priors
        fit_meta : mapping, optional

        Returns
        -------
        estimate : numpy struct array

        """
        estimate, _ = estimate_from_llhp(
            llhp=llhp,
            treat_dims_independently=False,
            use_prob_weights=True,
            priors_used=self.priors_used if remove_priors else None,
            meta=fit_meta,
        )

        # Test if the LLH would be positive without LLH_FUDGE_SUMMAND
        if estimate["max_llh"] > LLH_FUDGE_SUMMAND:
            sys.stderr.write(
                "\nWARNING: Would be positive LLH w/o LLH_FUDGE_SUMMAND: {}\n".format(
                    estimate["max_llh"]
                )
            )
            if estimate.dtype.names and "fit_status" in estimate.dtype.names:
                if estimate["fit_status"] not in (FitStatus.OK, FitStatus.PositiveLLH):
                    raise ValueError(
                        "Postive LLH *and* fit failed with fit_status = {}".format(
                            estimate["fit_status"]
                        )
                    )
                estimate["fit_status"] = FitStatus.PositiveLLH

        # Place reco in current event in case another reco depends on it
        if "recos" not in self.event:
            self.event["recos"] = OrderedDict()
        self.event["recos"]["retro_" + method] = estimate

        estimate_outf = join(
            self.event.meta["events_root"],
            "recos",
            "retro_{}.npy".format(method),
        )
        if isfile(estimate_outf):
            estimates = np.load(estimate_outf, mmap_mode="r+")
            try:
                estimates[self.event.meta["event_idx"]] = estimate
            finally:
                # ensure file handle is not left open
                del estimates
        else:
            estimates = np.full(
                shape=self.event.meta["num_events"],
                fill_value=np.nan,
                dtype=estimate.dtype,
            )
            # Filling with nan doesn't set correct "fit_status"
            estimates["fit_status"] = FitStatus.NotSet
            estimates[self.event.meta["event_idx"]] = estimate
            np.save(estimate_outf, estimates)

        ## meta_outf = join(
        ##    self.outdir, '{}{}.pkl'.format('meta', fname)
        ## )
        ## meta_file_exists = isfile(meta_outf)
        # if self.successful_reco_counter[method] == 0:
        #    if est_file_exists:
        #        raise IOError('Est file already exists at "{}"'.format(estimate_outf))
        #    # if meta_file_exists:
        #    #    raise IOError('Meta file already exists at "{}"'.format(meta_outf))
        #    estimates = estimate
        # else:
        #    if not est_file_exists:
        #        raise IOError(
        #            'Est file with previous events does not exist at "{}"'.format(
        #                estimate_outf
        #            )
        #        )
        #    # if not meta_file_exists:
        #    #    raise IOError(
        #    #        'Metadata file does not exist at "{}"'
        #    #        .format(meta_outf)
        #    #    )
        #    previous_estimates = np.load(estimate_outf)
        #    estimates = np.concatenate([previous_estimates, estimate])

        #    # TODO: verify meta data hasn't changed?
        #    # existing_meta = pickle.load(open(meta_outf))

        # np.save(file=estimate_outf, arr=estimates)
        ## if not meta_file_exists:
        ##    pickle.dump(
        ##        obj=meta,
        ##        file=open(meta_outf, 'wb'),
        ##        protocol=pickle.HIGHEST_PROTOCOL,
        ##    )

    def run_test(self, seed):
        """Random sampling instead of an actual minimizer"""
        raise NotImplementedError("`run_test` not implemented")  # TODO
        t0 = time.time()

        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_test)[1:]:
            kwargs[arg_name] = locals()[arg_name]

        rand = np.random.RandomState(seed=seed)
        for i in range(100):
            param_vals = rand.uniform(0, 1, self.n_opt_params)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
        run_info = OrderedDict([("method", "run_test"), ("kwargs", kwargs)])
        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(FitStatus.OK)),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )
        return run_info, fit_meta

    def run_with_truth(self, rand_dims=None, n_samples=10000, seed=0):
        """Run with all params set to truth except for the dimensions defined,
        which will be randomized.

        Parameters
        ----------
        rand_dims : list, optional
            Dimensions to randomly sample; all not specified are set to truth

        n_samples : int
            Number of samples to draw

        """
        raise NotImplementedError("`run_with_truth` not implemented")  # TODO
        t0 = time.time()

        if rand_dims is None:
            rand_dims = []

        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_with_truth)[1:]:
            kwargs[arg_name] = locals()[arg_name]

        truth = self.event["truth"]
        true_params = np.zeros(self.n_opt_params)

        for i, name in enumerate(self.hypo_handler.opt_param_names):
            name = name.replace("cascade_", "total_cascade_")
            true_params[i] = truth[name]

        rand = np.random.RandomState(seed=seed)
        if len(rand_dims) > 1:
            for i in range(n_samples):
                rand_params = rand.uniform(0, 1, self.n_opt_params)
                self.prior(rand_params)
                param_vals = np.zeros(self.n_opt_params)
                param_vals[:] = true_params[:]
                param_vals[rand_dims] = rand_params[rand_dims]
                llh = self.loglike(param_vals)
        else:
            llh = self.loglike(true_params)

        run_info = OrderedDict([("method", "run_with_truth"), ("kwargs", kwargs)])
        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(FitStatus.OK)),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )

        return run_info, fit_meta

    def run_crs(
        self,
        n_live,
        max_iter,
        max_noimprovement,
        min_llh_std,
        min_vertex_std,
        use_priors,
        use_sobol,
        seed,
    ):
        """Implementation of the CRS2 algorithm, adapted to work with spherical
        coordinates (correct centroid calculation, reflection, and mutation).

        At the moment Cartesian (standard) parameters and spherical parameters
        are assumed to have particular names (i.e., spherical coordinates start
        with "az" and "zen"). Furthermore, all Cartesian coordinates must come
        first followed by the pairs of (azimuth, zenith) spherical coordinates;
        e.g., "az_1", "zen_1", "az_2", "zen_2", etc.

        Parameters
        ----------
        n_live : int
            Number of live points
        max_iter : int
            Maximum iterations
        max_noimprovement : int
            Maximum iterations with no improvement of best point
        min_llh_std : float
            Break if stddev of llh values across all livepoints drops below
            this threshold
        min_vertex_std : mapping
            Break condition on stddev of Cartesian dimension(s) (x, y, z, and
            time). Keys are dimension names and values are the standard
            deviations for each dimension. All specified dimensions must drop
            below the specified stddevs for this break condition to be met.
        use_priors : bool
            Use priors during minimization; if `False`, priors are only used
            for sampling the initial distributions. Even if set to `True`,
            angles (azimuth and zenith) do not use priors while operating (only
            for generating the initial distribution)
        use_sobol : bool
            Use a Sobol sequence instead of numpy pseudo-random numbers. Seems
            to do slightly better (but only small differences observed in tests
            so far)
        seed : int
            Random seed

        Returns
        -------
        run_info : OrderedDict

        Notes
        -----
        CRS2 [1] is a variant of controlled random search (CRS, a global
        optimizer) with faster convergence than CRS.

        Refrences
        ---------
        .. [1] P. Kaelo, M.M. Ali, "Some variants of the controlled random
           search algorithm for global optimization," J. Optim. Theory Appl.,
           130 (2) (2006), pp. 253-264.

        """
        t0 = time.time()

        if use_sobol:
            from sobol import i4_sobol

        rand = np.random.RandomState(seed=seed)

        # Record kwargs user supplied (after translation & standardization)
        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_crs)[1:]:
            kwargs[arg_name] = locals()[arg_name]

        run_info = OrderedDict(
            [
                ("method", "run_crs"),
                ("method_description", "CRS2spherical+lm+sampling"),
                ("kwargs", kwargs),
            ]
        )

        n_opt_params = self.n_opt_params
        # absolute minimum number of points necessary
        assert n_live > n_opt_params + 1

        # figure out which variables are Cartesian and which spherical
        opt_param_names = self.hypo_handler.opt_param_names
        cart_param_names = set(opt_param_names) & set(CART_DIMS)
        n_cart = len(cart_param_names)
        assert set(opt_param_names[:n_cart]) == cart_param_names
        n_spher_param_pairs = int((n_opt_params - n_cart) / 2)
        for sph_pair_idx in range(n_spher_param_pairs):
            az_param = opt_param_names[n_cart + sph_pair_idx * 2]
            zen_param = opt_param_names[n_cart + sph_pair_idx * 2 + 1]
            assert "az" in az_param, '"{}" not azimuth param'.format(az_param)
            assert "zen" in zen_param, '"{}" not zenith param'.format(zen_param)

        for dim in min_vertex_std.keys():
            if dim not in opt_param_names:
                raise ValueError('dim "{}" not being optimized'.format(dim))
            if dim not in cart_param_names:
                raise NotImplementedError(
                    'dim "{}" stddev not computed, as stddev currently only'
                    " computed for Cartesian parameters".format(dim)
                )

        # set standard reordering so subsequent calls with different input
        # ordering will create identical metadata
        min_vertex_std = OrderedDict(
            [(d, min_vertex_std[d]) for d in opt_param_names if d in min_vertex_std]
        )

        # storage for info about stddev, whether met, and when met; defaults
        # should indicate failure if not explicitly set elsewhere
        vertex_std = np.full(
            shape=1,
            fill_value=np.nan,
            dtype=[(d, np.float32) for d in min_vertex_std.keys()],
        )
        vertex_std_met = OrderedDict([(d, False) for d in min_vertex_std.keys()])
        vertex_std_met_at_iter = np.full(
            shape=1, fill_value=-1, dtype=[(d, np.int32) for d in min_vertex_std.keys()]
        )

        # default values (in case of failure and these don't get set elsewhere,
        # then these values will be returned)
        fit_status = FitStatus.GeneralFailure
        iter_num = 0
        stopping_flag = 0
        llh_std = np.nan
        no_improvement_counter = 0
        num_simplex_successes = 0
        num_mutation_successes = 0
        num_failures = 0

        # setup arrays to store points
        s_cart = np.zeros(shape=(n_live, n_cart))
        s_spher = np.zeros(shape=(n_live, n_spher_param_pairs), dtype=SPHER_T)
        llh = np.zeros(shape=(n_live,))

        def func(x):
            """Callable for minimizer"""
            if use_priors:
                param_vals = np.zeros_like(x)
                param_vals[:n_cart] = x[:n_cart]
                self.prior(param_vals)
                param_vals[n_cart:] = x[n_cart:]
            else:
                param_vals = x
            llh = self.loglike(param_vals)
            if np.isnan(llh):
                raise ValueError("llh is nan; params are {}".format(param_vals))
            if np.any(np.isnan(param_vals)):
                raise ValueError("params are nan: {}".format(param_vals))
            return -llh

        def create_x(x_cart, x_spher):
            """Patch Cartesian and spherical coordinates into one array"""
            # TODO: make proper
            x = np.empty(shape=n_opt_params)
            x[:n_cart] = x_cart
            x[n_cart + 1 :: 2] = x_spher["zen"]
            x[n_cart::2] = x_spher["az"]
            return x

        try:
            # generate initial population
            for i in range(n_live):
                # Sobol seems to do slightly better than pseudo-random numbers
                if use_sobol:
                    # Note we start at seed=1 since for n_live=1 this puts the
                    # first point in the middle of the range for all params (0.5),
                    # while seed=0 produces all zeros (the most extreme point
                    # possible, which will bias the distribution away from more
                    # likely values).
                    x, _ = i4_sobol(
                        dim_num=n_opt_params,  # number of dimensions
                        seed=i + 1,  # Sobol sequence number
                    )
                else:
                    x = rand.uniform(0, 1, n_opt_params)

                # Apply prior xforms to `param_vals` (contents are overwritten)
                param_vals = np.copy(x)
                self.prior(param_vals)

                # Always use prior-xformed angles
                x[n_cart:] = param_vals[n_cart:]

                # Only use xformed Cart params if NOT using priors during operation
                if not use_priors:
                    x[:n_cart] = param_vals[:n_cart]

                # Break up into Cartesian and spherical coordinates
                s_cart[i] = x[:n_cart]
                s_spher[i]["zen"] = x[n_cart + 1 :: 2]
                s_spher[i]["az"] = x[n_cart::2]
                fill_from_spher(s_spher[i])
                llh[i] = func(x)

            best_llh = np.min(llh)
            no_improvement_counter = -1

            # optional bookkeeping
            num_simplex_successes = 0
            num_mutation_successes = 0
            num_failures = 0
            stopping_flag = 0

            # minimizer loop
            for iter_num in range(max_iter):
                if iter_num % REPORT_AFTER == 0:
                    print(
                        "simplex: %i, mutation: %i, failed: %i"
                        % (num_simplex_successes, num_mutation_successes, num_failures)
                    )

                # compute value for break condition 1
                llh_std = np.std(llh)

                # compute value for break condition 3
                for dim, cond in min_vertex_std.items():
                    vertex_std[dim] = std = np.std(
                        s_cart[:, opt_param_names.index(dim)]
                    )
                    vertex_std_met[dim] = met = std < cond
                    if met:
                        if vertex_std_met_at_iter[dim] == -1:
                            vertex_std_met_at_iter[dim] = iter_num
                    else:
                        vertex_std_met_at_iter[dim] = -1

                # break condition 1
                if llh_std < min_llh_std:
                    stopping_flag = 1
                    break

                # break condition 2
                if no_improvement_counter > max_noimprovement:
                    stopping_flag = 2
                    break

                # break condition 3
                if len(min_vertex_std) > 0 and all(vertex_std_met.values()):
                    stopping_flag = 3
                    break

                new_best_llh = np.min(llh)

                if new_best_llh < best_llh:
                    best_llh = new_best_llh
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                worst_idx = np.argmax(llh)
                best_idx = np.argmin(llh)

                # choose n_opt_params random points but not best
                choice = rand.choice(n_live - 1, n_opt_params, replace=False)
                choice[choice >= best_idx] += 1

                # Cartesian centroid
                centroid_cart = (
                    np.sum(s_cart[choice[:-1]], axis=0) + s_cart[best_idx]
                ) / n_opt_params

                # reflect point
                new_x_cart = 2 * centroid_cart - s_cart[choice[-1]]

                # spherical centroid
                centroid_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
                centroid_spher["x"] = (
                    np.sum(s_spher["x"][choice[:-1]], axis=0) + s_spher["x"][best_idx]
                ) / n_opt_params
                centroid_spher["y"] = (
                    np.sum(s_spher["y"][choice[:-1]], axis=0) + s_spher["y"][best_idx]
                ) / n_opt_params
                centroid_spher["z"] = (
                    np.sum(s_spher["z"][choice[:-1]], axis=0) + s_spher["z"][best_idx]
                ) / n_opt_params
                fill_from_cart(centroid_spher)

                # reflect point
                new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
                reflect(s_spher[choice[-1]], centroid_spher, new_x_spher)

                if use_priors:
                    outside = np.any(new_x_cart < 0) or np.any(new_x_cart > 1)
                else:
                    outside = False

                if not outside:
                    new_llh = func(create_x(new_x_cart, new_x_spher))

                    if new_llh < llh[worst_idx]:
                        # found better point
                        s_cart[worst_idx] = new_x_cart
                        s_spher[worst_idx] = new_x_spher
                        llh[worst_idx] = new_llh
                        num_simplex_successes += 1
                        continue

                # mutation
                w = rand.uniform(0, 1, n_cart)
                new_x_cart2 = (1 + w) * s_cart[best_idx] - w * new_x_cart

                # first reflect at best point
                reflected_new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
                reflect(new_x_spher, s_spher[best_idx], reflected_new_x_spher)

                new_x_spher2 = np.zeros_like(new_x_spher)

                # now do a combination of best and reflected point with weight w
                for dim in ("x", "y", "z"):
                    w = rand.uniform(0, 1, n_spher_param_pairs)
                    new_x_spher2[dim] = (1 - w) * s_spher[best_idx][
                        dim
                    ] + w * reflected_new_x_spher[dim]
                fill_from_cart(new_x_spher2)

                if use_priors:
                    outside = np.any(new_x_cart2 < 0) or np.any(new_x_cart2 > 1)
                else:
                    outside = False

                if not outside:
                    new_llh = func(create_x(new_x_cart2, new_x_spher2))

                    if new_llh < llh[worst_idx]:
                        # found better point
                        s_cart[worst_idx] = new_x_cart2
                        s_spher[worst_idx] = new_x_spher2
                        llh[worst_idx] = new_llh
                        num_mutation_successes += 1
                        continue

                # if we get here no method was successful in replacing worst
                # point -> start over
                num_failures += 1

            print(CRS_STOP_FLAGS[stopping_flag])
            fit_status = FitStatus.OK

        except KeyboardInterrupt:
            raise

        except Exception:
            self._print_non_fatal_exception(method=run_info["method"])

        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(fit_status)),
                ("iterations", np.uint32(iter_num)),
                ("stopping_flag", np.int8(stopping_flag)),
                ("llh_std", np.float32(llh_std)),
                ("no_improvement_counter", np.uint32(no_improvement_counter)),
                ("vertex_std", vertex_std),  # already typed
                ("vertex_std_met_at_iter", vertex_std_met_at_iter),  # already typed
                ("num_simplex_successes", np.uint32(num_simplex_successes)),
                ("num_mutation_successes", np.uint32(num_mutation_successes)),
                ("num_failures", np.uint32(num_failures)),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )

        return run_info, fit_meta

    def run_scipy(self, method, eps):
        """Use an optimizer from scipy"""
        t0 = time.time()

        from scipy import optimize

        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_scipy)[1:]:
            kwargs[arg_name] = locals()[arg_name]

        run_info = OrderedDict([("method", "run_scipy"), ("kwargs", kwargs)])

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def func(x, *args):  # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(eps, 1 - eps)] * self.n_opt_params
        settings = OrderedDict()
        settings["eps"] = eps

        fit_status = FitStatus.GeneralFailure
        try:
            if method == "differential_evolution":
                optimize.differential_evolution(func, bounds=bounds, popsize=100)
            else:
                optimize.minimize(
                    func, x0, method=method, bounds=bounds, options=settings
                )
            fit_status = FitStatus.OK

        except KeyboardInterrupt:
            raise

        except Exception:
            self._print_non_fatal_exception(method=run_info["method"])

        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(fit_status)),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )

        return run_info, fit_meta

    def run_skopt(self):
        """Use an optimizer from scikit-optimize"""
        t0 = time.time()

        from skopt import gp_minimize  # , forest_minimize

        settings = OrderedDict(
            [
                ("acq_func", "EI"),  # acquisition function
                ("n_calls", 1000),  # number of evaluations of f
                ("n_random_starts", 5),  # number of random initialization
            ]
        )
        run_info = OrderedDict([("method", "run_skopt"), ("settings", settings)])

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def func(x, *args):  # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(0, 1)] * self.n_opt_params

        fit_status = FitStatus.GeneralFailure
        try:
            _ = gp_minimize(
                func,  # function to minimize
                bounds,  # bounds on each dimension of x
                x0=list(x0),
                **settings
            )
            fit_status = FitStatus.OK

        except KeyboardInterrupt:
            raise

        except Exception:
            self._print_non_fatal_exception(method=run_info["method"])

        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(fit_status)),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )

        return run_info, fit_meta

    def run_nlopt(self):
        """Use an optimizer from nlopt"""
        t0 = time.time()

        import nlopt

        def func(x, grad):  # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        # bounds
        lower_bounds = np.zeros(shape=self.n_opt_params)
        upper_bounds = np.ones(shape=self.n_opt_params)

        # for angles make bigger
        for i, name in enumerate(self.hypo_handler.opt_param_names):
            if "azimuth" in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5
            if "zenith" in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        # stepsize
        dx = np.zeros(shape=self.n_opt_params)
        for i in range(self.n_opt_params):
            if "azimuth" in self.hypo_handler.opt_param_names[i]:
                dx[i] = 0.001
            elif "zenith" in self.hypo_handler.opt_param_names[i]:
                dx[i] = 0.001
            elif self.hypo_handler.opt_param_names[i] in ("x", "y"):
                dx[i] = 0.005
            elif self.hypo_handler.opt_param_names[i] == "z":
                dx[i] = 0.002
            elif self.hypo_handler.opt_param_names[i] == "time":
                dx[i] = 0.01

        # seed from several angles
        # opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
        opt = nlopt.opt(nlopt.GN_CRS2_LM, self.n_opt_params)
        ftol_abs = 0.1
        # opt = nlopt.opt(nlopt.LN_PRAXIS, self.n_opt_params)
        opt.set_lower_bounds([0.0] * self.n_opt_params)
        opt.set_upper_bounds([1.0] * self.n_opt_params)
        opt.set_min_objective(func)
        opt.set_ftol_abs(ftol_abs)

        settings = OrderedDict(
            [("method", opt.get_algorithm_name()), ("ftol_abs", np.float32(ftol_abs))]
        )

        run_info = OrderedDict([("method", "run_nlopt"), ("settings", settings)])

        fit_status = FitStatus.GeneralFailure
        try:
            # initial guess

            angles = np.linspace(0, 1, 3)
            angles = 0.5 * (angles[1:] + angles[:-1])

            for zen in angles:
                for az in angles:
                    x0 = 0.5 * np.ones(shape=self.n_opt_params)

                    for i in range(self.n_opt_params):
                        if "az" in self.hypo_handler.opt_param_names[i]:
                            x0[i] = az
                        elif "zen" in self.hypo_handler.opt_param_names[i]:
                            x0[i] = zen
                    x = opt.optimize(x0)  # pylint: disable=unused-variable

            # local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
            # local_opt.set_lower_bounds([0.]*self.n_opt_params)
            # local_opt.set_upper_bounds([1.]*self.n_opt_params)
            # local_opt.set_min_objective(func)
            ##local_opt.set_ftol_abs(0.5)
            ##local_opt.set_ftol_abs(100)
            ##local_opt.set_xtol_rel(10)
            # local_opt.set_ftol_abs(1)
            # global
            # opt = nlopt.opt(nlopt.G_MLSL, self.n_opt_params)
            # opt.set_lower_bounds([0.]*self.n_opt_params)
            # opt.set_upper_bounds([1.]*self.n_opt_params)
            # opt.set_min_objective(func)
            # opt.set_local_optimizer(local_opt)
            # opt.set_ftol_abs(10)
            # opt.set_xtol_rel(1)
            # opt.set_maxeval(1111)

            # opt = nlopt.opt(nlopt.GN_ESCH, self.n_opt_params)
            # opt = nlopt.opt(nlopt.GN_ISRES, self.n_opt_params)
            # opt = nlopt.opt(nlopt.GN_CRS2_LM, self.n_opt_params)
            # opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND_NOSCAL, self.n_opt_params)
            # opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)

            # opt.set_lower_bounds(lower_bounds)
            # opt.set_upper_bounds(upper_bounds)
            # opt.set_min_objective(func)
            # opt.set_ftol_abs(0.1)
            # opt.set_population([x0])
            # opt.set_initial_step(dx)

            # local_opt.set_maxeval(10)

            # x = opt.optimize(x0) # pylint: disable=unused-variable

            # polish it up
            # print('***************** polishing ******************')

            # dx = np.ones(shape=self.n_opt_params) * 0.001
            # dx[0] = 0.1
            # dx[1] = 0.1

            # local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
            # lower_bounds = np.clip(np.copy(x) - 0.1, 0, 1)
            # upper_bounds = np.clip(np.copy(x) + 0.1, 0, 1)
            # lower_bounds[0] = 0
            # lower_bounds[1] = 0
            # upper_bounds[0] = 0
            # upper_bounds[1] = 0

            # local_opt.set_lower_bounds(lower_bounds)
            # local_opt.set_upper_bounds(upper_bounds)
            # local_opt.set_min_objective(func)
            # local_opt.set_ftol_abs(0.1)
            # local_opt.set_initial_step(dx)
            # x = opt.optimize(x)

            fit_status = FitStatus.OK

        except KeyboardInterrupt:
            raise

        except Exception:
            self._print_non_fatal_exception(method=run_info["method"])

        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(fit_status)),
                ("run_time", np.float32(time.time() - t0)),
                ("ftol_abs", np.float32(opt.get_ftol_abs())),
                ("ftol_rel", np.float32(opt.get_ftol_rel())),
                ("xtol_abs", np.float32(opt.get_xtol_abs())),
                ("xtol_rel", np.float32(opt.get_xtol_rel())),
                ("maxeval", np.float32(opt.get_maxeval())),
                ("maxtime", np.float32(opt.get_maxtime())),
                ("stopval", np.float32(opt.get_stopval())),
            ]
        )

        return run_info, fit_meta

    def run_dynesty(
        self,
        n_live,
        maxiter,
        maxcall,
        dlogz
    ):
        """Setup and run Dynesty on an event.

        Parameters
        ----------

        Returns
        -------
        run_info : OrderedDict
            Metadata dict containing dynesty settings used and extra info returned by
            dynesty

        fit_meta : OrderedDict

        """
        import dynesty

        t0 = time.time()

        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_dynesty)[1:]:
            kwargs[arg_name] = locals()[arg_name]


        dn_kwargs = OrderedDict(
            [
                ("ndim", self.n_opt_params),
                ('nlive', n_live),
                (
                    "periodic",
                    [i for i, p in enumerate(self.hypo_handler.all_param_names) if 'az' in p.lower()],
                ),
            ]
        )

        sampler_kwargs = OrderedDict(
            [
                ('maxiter', maxiter),
                ('maxcall', maxcall),
                ('dlogz', dlogz),
            ]
        )

        run_info = OrderedDict(
            [
                ("method", "run_dynesty"),
                ("kwargs", kwargs),
                ("dn_kwargs", dn_kwargs),
                ("sampler_kwargs", sampler_kwargs),
            ]
        )

        fit_meta = OrderedDict()
        fit_meta["fit_status"] = np.int8(FitStatus.NotSet)
        sampler = dynesty.NestedSampler(
            loglikelihood=self.loglike,
            prior_transform=self.prior,
            method='unif',
            bound='single',
            update_interval=1,
            **dn_kwargs
        )
        print('sampler instantiated')
        sampler.run_nested(**sampler_kwargs)

        fit_meta["fit_status"] = np.int8(FitStatus.OK)
        fit_meta["run_time"] = np.float32(time.time() - t0)

        print(fit_meta)

        return run_info, fit_meta


    def run_multinest(
        self,
        importance_sampling,
        max_modes,
        const_eff,
        n_live,
        evidence_tol,
        sampling_eff,
        max_iter,
        seed,
    ):
        """Setup and run MultiNest on an event.

        See the README file from MultiNest for greater detail on parameters
        specific to to MultiNest (parameters from `importance_sampling` on).

        Parameters
        ----------
        importance_sampling
        max_modes
        const_eff
        n_live
        evidence_tol
        sampling_eff
        max_iter
            Note that this limit is the maximum number of sample replacements
            and _not_ max number of likelihoods evaluated. A replacement only
            occurs when a likelihood is found that exceeds the minimum
            likelihood among the live points.
        seed

        Returns
        -------
        run_info : OrderedDict
            Metadata dict containing MultiNest settings used and extra info returned by
            MultiNest

        fit_meta : OrderedDict

        """
        t0 = time.time()

        # Import pymultinest here; it's a less common dependency, so other
        # functions/constants in this module will still be import-able w/o it.
        import pymultinest

        kwargs = OrderedDict()
        for arg_name in get_arg_names(self.run_multinest)[1:]:
            kwargs[arg_name] = locals()[arg_name]

        mn_kwargs = OrderedDict(
            [
                ("n_dims", self.n_opt_params),
                ("n_params", self.n_params),
                ("n_clustering_params", self.n_opt_params),
                (
                    "wrapped_params",
                    ["az" in p.lower() for p in self.hypo_handler.all_param_names],
                ),
                ("importance_nested_sampling", importance_sampling),
                ("multimodal", max_modes > 1),
                ("const_efficiency_mode", const_eff),
                ("n_live_points", n_live),
                ("evidence_tolerance", evidence_tol),
                ("sampling_efficiency", sampling_eff),
                ("null_log_evidence", -1e90),
                ("max_modes", max_modes),
                ("mode_tolerance", -1e90),
                ("seed", seed),
                ("log_zero", -1e100),
                ("max_iter", max_iter),
            ]
        )

        run_info = OrderedDict(
            [("method", "run_multinest"), ("kwargs", kwargs), ("mn_kwargs", mn_kwargs)]
        )

        fit_status = FitStatus.GeneralFailure
        tmpdir = mkdtemp()
        outputfiles_basename = join(tmpdir, "")
        mn_fit_meta = {}
        try:
            pymultinest.run(
                LogLikelihood=self.loglike,
                Prior=self.prior,
                verbose=True,
                outputfiles_basename=outputfiles_basename,
                resume=False,
                write_output=True,
                n_iter_before_update=REPORT_AFTER,
                **mn_kwargs
            )
            fit_status = FitStatus.OK
            mn_fit_meta = get_multinest_meta(outputfiles_basename=outputfiles_basename)

        except KeyboardInterrupt:
            raise

        except Exception:
            self._print_non_fatal_exception(method=run_info["method"])

        finally:
            rmtree(tmpdir)

        # TODO: If MultiNest fails in specific ways, set fit_status accordingly...

        fit_meta = OrderedDict(
            [
                ("fit_status", np.int8(fit_status)),
                ("logZ", np.float32(mn_fit_meta.pop("logZ", np.nan))),
                ("logZ_err", np.float32(mn_fit_meta.pop("logZ_err", np.nan))),
                ("ins_logZ", np.float32(mn_fit_meta.pop("ins_logZ", np.nan))),
                ("ins_logZ_err", np.float32(mn_fit_meta.pop("ins_logZ_err", np.nan))),
                ("run_time", np.float32(time.time() - t0)),
            ]
        )

        if mn_fit_meta:
            sys.stderr.write(
                "WARNING: Unrecorded MultiNest metadata: {}\n".format(
                    ", ".join("{} = {}".format(k, v) for k, v in mn_fit_meta.items())
                )
            )

        return run_info, fit_meta


def get_multinest_meta(outputfiles_basename):
    """Get metadata from files that MultiNest writes to disk.

    Parameters
    ----------
    outputfiles_basename : str

    Returns
    -------
    fit_meta : OrderedDict
        Contains "logZ", "logZ_err" and, if importance nested sampling was run,
        "ins_logZ" and "ins_logZ_err"

    """
    fit_meta = OrderedDict()
    if isdir(outputfiles_basename):
        stats_fpath = join(outputfiles_basename, "stats.dat")
    else:
        stats_fpath = outputfiles_basename + "stats.dat"

    with open(stats_fpath, "r") as stats_f:
        stats = stats_f.readlines()

    logZ, logZ_err = None, None
    ins_logZ, ins_logZ_err = None, None

    for line in stats:
        if logZ is None and line.startswith("Nested Sampling Global Log-Evidence"):
            logZ, logZ_err = [float(x) for x in line.split(":")[1].split("+/-")]
        elif ins_logZ is None and line.startswith(
            "Nested Importance Sampling Global Log-Evidence"
        ):
            ins_logZ, ins_logZ_err = [float(x) for x in line.split(":")[1].split("+/-")]

    if logZ is not None:
        fit_meta["logZ"] = np.float32(logZ)
        fit_meta["logZ_err"] = np.float32(logZ_err)
    if ins_logZ is not None:
        fit_meta["ins_logZ"] = np.float32(ins_logZ)
        fit_meta["ins_logZ_err"] = np.float32(ins_logZ_err)

    return fit_meta


def main(description=__doc__):
    """Script interface to Reco class and Reco.run(...) method"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        "--methods",
        required=True,
        choices=METHODS,
        nargs="+",
        help="""Method(s) to use for performing reconstructions; performed in
        order specified, so be sure to specify pre-fits / fits used as seeds
        first""",
    )
    parser.add_argument(
        "--redo-failed",
        action="store_true",
        help="""Whether to re-reconstruct events that have been reconstructed
        but have `fit_status` set to non-zero (i.e., not `FitStatus.OK`), in
        addition to reconstructing events with `fit_status` set to -1 (i.e.,
        `FitStatus.NotSet`)""",
    )
    parser.add_argument(
        "--redo-all",
        action="store_true",
        help="""Whether to reconstruct all events without existing
        reconstructions AND re-reconstruct all events that have existing
        reconstructions, regardless if their `fit_status` is OK or some form of
        failure""",
    )
    parser.add_argument(
        "--save-llhp",
        action="store_true",
        help="Whether to save LLHP within 30 LLH of max-LLH to disk",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="""Filter to apply for selecting events to reconstruct. String is
        passed through `eval` and must produce a scalar value interpretable via
        `bool(eval(filter))`. Current event is accessible via the name `event`
        and numpy is named `np`. E.g.,
        --filter='event["header"]["L5_oscNext_bool"]'"""
    )

    split_kwargs = init_obj.parse_args(
        dom_tables=True, tdi_tables=True, events=True, parser=parser
    )

    other_kw = split_kwargs.pop("other_kw")
    my_reco = Reco(**split_kwargs)
    my_reco.run(**other_kw)


if __name__ == "__main__":
    main()
