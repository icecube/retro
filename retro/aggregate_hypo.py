# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Aggregate together multiple hypotheses (objects of type :class:`retro.hypo.Hypo`).
"""

from __future__ import absolute_import, division, print_function

__all__ = ["AggregateHypo", "test_AggregateHypo"]

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2018 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from collections import OrderedDict
from copy import copy
from inspect import cleandoc
from os.path import abspath, dirname
import sys

import numba
import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit
from retro.const import (
    EMPTY_SOURCES, SPEED_OF_LIGHT_M_PER_NS, SrcHandling, dummy_pegleg_gens
)
from retro.hypo import Hypo
from retro.utils.misc import make_valid_python_name


class AggregateHypo(Hypo):
    """
    Aggregate together multiple hypotheses (objects of type :class:`retro.hypo.Hypo`).

    Parameters
    ----------
    hypos : Hypo or Iterable thereof
    hypo_names : str or Iterable thereof, same number as there are `hypos`

    """
    def __init__(self, hypos, hypo_names):
        # -- Convert `hypos` to a tuple of one or more hypos -- #

        if isinstance(hypos, Hypo):
            hypos = (hypos,)
        else:
            hypos = tuple(hypos)

        # -- Convert `hypo_names` to tuple of strs and validate -- #

        if isinstance(hypo_names, string_types):
            hypo_names = (hypo_names,)
        else:
            hypo_names = tuple(hypo_names)

        if len(hypo_names) != len(hypos):
            raise ValueError(
                "got {} `hypo_names` but there are {} `hypos`"
                .format(len(hypo_names), len(hypos))
            )

        invalid_names = []
        for name in hypo_names:
            valid_name = make_valid_python_name(name).lower()
            if name != valid_name:
                invalid_names.append(name)

        if invalid_names:
            raise ValueError(
                'Invalid hypo_name(s) "{}"; must be lower-case and formatted as a'
                ' valid Python identifier'
                .format(invalid_names)
            )

        # -- Make a trivial mapping from ext -> ext param names -- #

        external_param_names = set()
        max_num_scalefactors_per_hypo = []
        for hypo in hypos:
            external_param_names.update(hypo.external_param_names)
            max_num_scalefactors_per_hypo.append(hypo.max_num_scalefactors)
        external_param_names = tuple(sorted(external_param_names))
        dummy_internal_param_names = copy(external_param_names)
        param_mapping = {pn: pn for pn in external_param_names}

        # -- Initialize base class -- #

        super(AggregateHypo, self).__init__(
            internal_param_names=dummy_internal_param_names,
            param_mapping=param_mapping,
        )

        # -- Store attrs unique to AggregateHypo -- #

        self.hypos = hypos
        self.hypo_names = hypo_names
        self.max_num_scalefactors_per_hypo = tuple(max_num_scalefactors_per_hypo)
        self.max_num_scalefactors = np.max(max_num_scalefactors_per_hypo)
        self.num_hypos_with_scalefactors = (
            np.count_nonzero(max_num_scalefactors_per_hypo)
        )
        self.energy_per_hypo = None
        self.num_pegleg_generators_per_hypo = None
        self.num_scalefactors_per_hypo = None

        # -- Record configuration items unique to an aggregate hypo -- #

        self.config["hypo_configs"] = tuple(hypo.config for hypo in hypos)
        self.config["max_num_scalefactors_per_hypo"] = max_num_scalefactors_per_hypo

    def get_sources(self, **kwargs):
        """Call hypotheses' `get_sources` methods individually and aggregate the
        results.

        Overrides `Hypo.get_sources` method.

        Parameters
        ----------
        **kwargs
            Passed to each `hypo.get_sources` method

        Returns
        -------
        sources
        sources_handling
        num_pegleg_generators
        pegleg_generators

        """
        self.num_calls += 1

        for external_param_name in self.external_param_names:
            try:
                self.external_params[external_param_name] = kwargs[external_param_name]
            except KeyError:
                print(
                    'Missing param "{}" in passed kwargs {}'
                    .format(external_param_name, kwargs)
                )
                raise

        # Map external param names/values onto internal param names/values
        self.internal_params = self.param_mapping(**self.external_params)

        # -- Create generator to call `pegleg_generators` from each hypo -- #

        sources = [] # all scaling and nonscaling sources
        sources_handling = [] # all handling designations (scaling or nonscaling)
        num_pl_gens = [] # number of pegleg generators per hypo kernel
        pegleg_generators = [] # pegleg generator per hypo kernel
        num_scalefactors_per_hypo = [] # number of scalefactors per hypo kernel

        for hypo in self.hypos:
            # Get all sources & any pegleg generators
            sources_, sources_handling_, n_pl, pl_gens = hypo.get_sources(**kwargs)

            # Aggregate the sources, sources_handling & generators into separate lists
            # and record how many pegleg generators & scalefactors apply to the hypo so
            # that pegleg indices and scalefactors can be divvied up appropriately to
            # each component hypo later
            num_sf = 0
            for srcs, srcs_hndl in zip(sources_, sources_handling_):
                if srcs_hndl == SrcHandling.none:
                    continue
                elif srcs_hndl == SrcHandling.nonscaling:
                    sources.append(srcs)
                    sources_handling.append(srcs_hndl)
                elif srcs_hndl == SrcHandling.scaling:
                    num_sf += 1
                    sources.append(srcs)
                    sources_handling.append(srcs_hndl)
                else:
                    raise ValueError("Invalid sources handling {}".format(srcs_hndl))

            num_pl_gens.append(n_pl)
            pegleg_generators.append(pl_gens)
            num_scalefactors_per_hypo.append(num_sf)

        num_pegleg_generators = np.sum(num_pl_gens)

        # We need at least one value of each type to pass to Numba or it will choke
        # (since it can't type the elements of an empty list or tuple)
        if len(sources) == 0:
            sources.append(EMPTY_SOURCES)
            sources_handling.append(SrcHandling.none)

        if num_pegleg_generators == 0:
            pegleg_generators = dummy_pegleg_gens
        else:
            # Ensure all generators are jit-compiled
            for gens_num, gens in enumerate(pegleg_generators):
                if not isinstance(gens, numba.targets.registry.CPUDispatcher):
                    try:
                        gens = numba.njit(cache=True, fastmath=False, nogil=True)(gens)
                    except:
                        print("failed to numba-jit-compile generator {}".format(gens))
                        raise
                pegleg_generators[gens_num] = gens

            # -- Construct super pegleg generator to call hypos' pegleg generators -- #
            #    ("dirty hack" since Numba can't be passed a list of functions)

            conditional_lines = []
            lcls = locals()

            gens_idx = 0
            start_gen_idx = 0
            for this_num_pl_gens, this_pl_gens in zip(num_pl_gens, pegleg_generators):
                if this_num_pl_gens == 0:
                    continue

                stop_gen_idx = start_gen_idx + this_num_pl_gens

                # Define a variable (gens0, gens1, etc.) in local scope that is the callee
                gens_name = "gens{:d}".format(gens_idx)
                lcls[gens_name] = this_pl_gens

                # Define the conditional
                conditional = "if" if len(conditional_lines) == 0 else "elif"
                conditional_lines.append(
                    "    {} {} <= gen_idx < {}:".format(
                        conditional, start_gen_idx, stop_gen_idx
                    )
                )

                # Define the execution statement, calling the internal generator with the
                # index applicable to that generator
                conditional_lines.append(
                    "        for val in {gen_name}(gen_idx - {start_gen_idx}):"
                    .format(gen_name=gens_name, start_gen_idx=start_gen_idx)
                )
                conditional_lines.append(
                    "            yield val"
                )

                start_gen_idx = stop_gen_idx
                gens_idx += 1

            lcls["EMPTY_SOURCES"] = EMPTY_SOURCES
            lcls["SrcHandling"] = SrcHandling
            default_yield = "yield (EMPTY_SOURCES,), (SrcHandling.none,)"
            if len(conditional_lines) == 0:
                final_condit = "    {}".format(default_yield)
            else:
                final_condit = "\n".join([
                    "    else:",
                    "        {}".format(default_yield)
                ])

            # Remove leading spaces uniformly in string
            py_pegleg_generators_str = cleandoc(
                """
                def py_pegleg_generators(gen_idx):
                {body}
                {final_condit}
                """
            ).format(
                body="\n".join(conditional_lines),
                final_condit=final_condit,
            )

            try:
                exec py_pegleg_generators_str in locals() # pylint: disable=exec-used
            except:
                print(py_pegleg_generators_str)
                raise

            pegleg_generators = numba_jit(fastmath=False, nogil=True, nopython=True)(
                py_pegleg_generators # pylint: disable=undefined-variable
            )

        # -- Store aggregated values & new "super" pegleg generator as attrs -- #

        self.sources = tuple(sources)
        self.sources_handling = tuple(sources_handling)
        self.num_pegleg_generators = num_pegleg_generators
        self.pegleg_generators = pegleg_generators
        self.num_pegleg_generators_per_hypo = num_pl_gens
        self.num_scalefactors_per_hypo = num_scalefactors_per_hypo

        return (
            self.sources,
            self.sources_handling,
            self.num_pegleg_generators,
            self.pegleg_generators
        )

    def get_energy(self, pegleg_indices=None, scalefactors=None):
        """Get energy from each hypothesis, returning the total and storing each
        individual hypothesis's energy to `self.energy_per_hypo`.

        Parameters
        ----------
        pegleg_indices
        scalefactors

        Returns
        -------
        total_energy
            Energy summed from all component hypotheses, in units of GeV

        """
        pl_start_idx = 0
        sf_hypo_idx = 0
        energy_per_hypo = []
        for n_pl, n_sf, hypo in zip(
            self.num_pegleg_generators_per_hypo,
            self.num_scalefactors_per_hypo,
            self.hypos,
        ):
            if n_pl > 0:
                pl_stop_idx = pl_start_idx + n_pl
                hypo_pegleg_indices = pegleg_indices[pl_start_idx:pl_stop_idx]
            else:
                hypo_pegleg_indices = None

            if n_sf > 0:
                hypo_scalefactors = scalefactors[sf_hypo_idx, 0:n_sf]
            else:
                hypo_scalefactors = None

            energy_per_hypo.append(
                hypo.get_energy(
                    pegleg_indices=hypo_pegleg_indices,
                    scalefactors=hypo_scalefactors,
                )
            )

            pl_start_idx = pl_stop_idx

        self.energy_per_hypo = tuple(energy_per_hypo)

        return np.sum(energy_per_hypo)

    def get_derived_params(self, pegleg_indices=None, scalefactors=None):
        """Retrieve any derived params from component hypotheses.

        Parameters
        ----------
        pegleg_indices : required if a component hypothesis is pegleg
        scalefactors : required if a component hypothesis is scaling

        Returns
        -------
        derived_params : OrderedDict
            Keys are strings concatenating the name of the hypo with the
            derived param name (as returned by the hypo), separated by an underscore

        """
        derived_params = OrderedDict()

        pl_start_idx = 0
        sf_hypo_idx = 0
        for name, n_pl, n_sf, hypo in zip(
            self.hypo_names,
            self.num_pegleg_generators_per_hypo,
            self.num_scalefactors_per_hypo,
            self.hypos,
        ):
            if n_pl > 0:
                assert pegleg_indices is not None
                pl_stop_idx = pl_start_idx + n_pl
                hypo_pegleg_indices = pegleg_indices[pl_start_idx:pl_stop_idx]
            else:
                hypo_pegleg_indices = None

            if n_sf > 0:
                assert scalefactors is not None
                hypo_scalefactors = scalefactors[sf_hypo_idx, 0:n_sf]
            else:
                hypo_scalefactors = None

            this_derived_params = hypo.get_derived_params(
                pegleg_indices=hypo_pegleg_indices,
                scalefactors=hypo_scalefactors,
            )
            for key, val in this_derived_params.items():
                derived_params[name + "_" + key] = val

            pl_start_idx = pl_stop_idx

        return derived_params


def test_AggregateHypo():
    """Unit tests for :class:`AggregateHypo`"""
    # Import here to avoid unnecessary imports under non-testing conditions
    from cascade_hypo import CascadeHypo
    from muon_hypo import MuonHypo

    pl_track_lengths = [1000, 100]

    # -- Pegleg muon -- #

    mu0_mapping = dict(
        x="x", y="y", z="z", time="time", mu0_azimuth="azimuth", mu0_zenith="zenith"
    )
    muon0 = MuonHypo(
        fixed_track_length=pl_track_lengths[0],
        pegleg_step_size=10,
        param_mapping=mu0_mapping,
        continuous_loss_model="all_avg_gms_table",
        stochastic_loss_model=None,
        continuous_loss_model_kwargs=dict(time_step=1),
    )

    # -- Non-pegleg muon -- #

    mu1_mapping = dict(
        x="x", y="y", z="z", time="time", mu1_length="track_length",
        mu1_azimuth="azimuth", mu1_zenith="zenith"
    )
    muon1 = MuonHypo(
        param_mapping=mu1_mapping,
        continuous_loss_model="all_avg_leera2",
        stochastic_loss_model=None,
        continuous_loss_model_kwargs=dict(time_step=1),
    )

    # -- Another pegleg muon -- #

    mu2_mapping = dict(
        x="x", y="y", z="z", time="time", mu2_azimuth="azimuth",
        mu2_zenith="zenith"
    )
    muon2 = MuonHypo(
        fixed_track_length=pl_track_lengths[1],
        pegleg_step_size=1,
        param_mapping=mu2_mapping,
        continuous_loss_model="all_avg_const",
        stochastic_loss_model=None,
        continuous_loss_model_kwargs=dict(time_step=1),
    )

    # -- Scaling cascade -- #

    cscd0_mapping = dict(
        x="x", y="y", z="z", time="time", cscd0_azimuth="azimuth",
        cscd0_zenith="zenith",
    )
    cscd0 = CascadeHypo(
        param_mapping=cscd0_mapping,
        model="one_dim_v1",
        num_sources=-1,
        scaling_proto_energy=100,
    )

    # -- Non-scaling cascade -- #

    def cscd1_param_mapping(
        x, y, z, time, mu1_azimuth, mu1_zenith, cscd1_offset, cscd1_energy,
    ):
        """Put cascade collinear but translated along the track"""
        dt = SPEED_OF_LIGHT_M_PER_NS * cscd1_offset
        cos_zen = np.cos(mu1_zenith)
        sin_zen = np.sin(mu1_zenith)
        sin_az = np.sin(mu1_azimuth)
        cos_az = np.cos(mu1_azimuth)
        dz = cos_zen * cscd1_offset
        rho = sin_zen * cscd1_offset
        dx = rho * cos_az
        dy = rho * sin_az
        return dict(
            x=x + dx,
            y=y + dy,
            z=z + dz,
            time=time + dt,
            azimuth=mu1_azimuth,
            zenith=mu1_zenith,
            energy=cscd1_energy,
        )

    cscd1 = CascadeHypo(
        param_mapping=cscd1_param_mapping,
        model="one_dim_v1",
        num_sources=1,
    )

    # -- Aggregate all hypos -- #

    hypos = [muon0, muon1, muon2, cscd0, cscd1]

    external_param_names = set()
    for hypo in hypos:
        print(hypo.external_param_names)
        external_param_names.update(hypo.external_param_names)

    rand = np.random.RandomState(0)
    params_kw = {k: rand.uniform(1, 3) for k in external_param_names}

    # Make sure each hypo works on its own
    for hypo in hypos:
        _, _, _, _ = hypo.get_sources(**params_kw)

    #sources, handling, n_pl, pl_gen = aggregate_sources(hypos=hypos, **params_kw)
    agg_hypo = AggregateHypo(hypos, hypo_names=["mu0", "mu1", "mu2", "cscd0", "cscd1"])
    sources, handling, n_pl, pl_gen = agg_hypo.get_sources(**params_kw)
    derived_params = agg_hypo.get_derived_params(
        pegleg_indices=[50, 100],
        scalefactors=np.array([[20]]),
    )
    print("len(sources):", len(sources))
    print("handling:", handling)
    print("n_pl:", n_pl)
    print("pl_gen:", pl_gen)
    print("params:        ", agg_hypo.external_params)
    print("derived_params:", derived_params)

    for pl_gen_num in range(n_pl):
        all_pl_sources = []
        for out in pl_gen(pl_gen_num):
            pl_sources, _ = out
            all_pl_sources.extend(pl_sources)
        all_pl_sources = np.concatenate(all_pl_sources)
        #n_pl_sources = len(all_pl_sources)
        #n_pl_sources *

    print("<< PASS : test_AggregateHypo >>")


if __name__ == "__main__":
    test_AggregateHypo()
