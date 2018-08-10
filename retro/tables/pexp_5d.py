# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals, consider-using-enumerate

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['MACHINE_EPS', 'generate_pexp_function']

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
from retro.const import SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI, SRC_CKV_BETA1
from retro.utils.geom import generate_digitizer


MACHINE_EPS = 1e-10


def generate_pexp_function(
    table_meta,
    table_kind,
    t_is_residual_time,
    tdi_table_meta=None,
    template_library=None,
):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function to use.

    Parameters
    ----------
    table_meta : mapping
        As returned by `load_clsim_table_minimal`

    table_kind : str in {'raw_uncompr', 'ckv_uncompr', 'ckv_templ_compr'}

    t_is_residual_time : bool
        Whether time axis is actually time residual, defined to be
        (actual time) - (fastest time a photon could get to spatial coordinate)

    tdi_table_meta : sequence of two mappings, optional
        If provided, sequence _must_ contain two mappings, where the first corresponds
        to the finest-binned TDI table and the second corresponds to the coarsest-binned
        table (the first table takes priority over the second for looking up sources).
        Each mapping must contain keys "x_bin_edges", "y_bin_edges", "z_bin_edges",
        "costhetadir_bin_edges", and "phidir_bin_edges"; values are arrays of the bin
        edges in each of these dimensions

    template_library : shape-(n_templates, n_dir_theta, n_dir_deltaphi) array
        Containing the directionality templates for compressed tables

    Returns
    -------
    pexp : callable
        Function usable to extract photon expectations from a table of
        `table_kind` and with the binning of `table_meta` (and optionally
        `tdi_table_meta`).

    get_llh : callable

    meta : OrderedDict
        Parameters, including the binning, that uniquely identify what the
        capabilities of the returned `pexp`. (Use this to eliminate
        redundant pexp functions.)

    """
    # pylint: disable=missing-docstring
    tbl_is_ckv = table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    tbl_is_templ_compr = table_kind in ['raw_templ_compr', 'ckv_templ_compr']
    if not tbl_is_ckv:
        raise NotImplementedError('Only Ckv tables are implemented.')

    meta = OrderedDict()
    meta['table_kind'] = table_kind
    meta['table_binning'] = OrderedDict()
    for key in (
        'r_bin_edges', 'costhetadir_bin_edges', 't_bin_edges', 'costhetadir_bin_edges',
        'deltaphidir_bin_edges'
    ):
        meta['table_binning'][key] = table_meta[key]

    if tdi_table_meta is not None:
        assert tdi_table_meta['phidir_bin_edges'][0] == -np.pi
        assert tdi_table_meta['phidir_bin_edges'][-1] == np.pi

        meta['tdi_table_binning'] = OrderedDict()
        for key in (
            'x_bin_edges', 'y_bin_edges', 'z_bin_edges', 'costhetadir_bin_edges',
            'phidir_bin_edges'
        ):
            meta['tdi_table_binning'][key] = tdi_table_meta[key]

    # NOTE: For now, we only support absolute value of deltaphidir (which
    # assumes azimuthal symmetry). In future, this could be revisited (and then
    # the abs(...) applied before binning in the pexp code will have to be
    # removed or replaced with behavior that depend on the range of the
    # deltaphidir_bin_edges).
    assert table_meta['deltaphidir_bin_edges'][0] == 0, 'only abs(deltaphidir) supported'
    assert table_meta['deltaphidir_bin_edges'][-1] == np.pi

    # -- Define things used by `pexp*` closures defined below -- #

    # Constants
    rsquared_max = np.max(table_meta['r_bin_edges'])**2
    t_max = np.max(table_meta['t_bin_edges'])
    recip_max_group_vel = table_meta['group_refractive_index'] / SPEED_OF_LIGHT_M_PER_NS

    # Digitization functions for each binning dimension
    digitize_r = generate_digitizer(table_meta['r_bin_edges'], clip=True)
    digitize_costheta = generate_digitizer(table_meta['costheta_bin_edges'], clip=True)
    digitize_t = generate_digitizer(table_meta['t_bin_edges'], clip=True)
    digitize_costhetadir = generate_digitizer(
        table_meta['costhetadir_bin_edges'], clip=True
    )
    digitize_deltaphidir = generate_digitizer(
        np.sin(table_meta['deltaphidir_bin_edges']), clip=True
    )

    x_edges = tdi_table_meta[0]['x_bin_edges']
    y_edges = tdi_table_meta[0]['y_bin_edges']
    z_edges = tdi_table_meta[0]['z_bin_edges']
    tdi0_xmin, tdi0_xmax = x_edges[0, -1]
    tdi0_ymin, tdi0_ymax = y_edges[0, -1]
    tdi0_zmin, tdi0_zmax = z_edges[0, -1]
    digitize_tdi0_x = generate_digitizer(x_edges, clip=True)
    digitize_tdi0_y = generate_digitizer(y_edges, clip=True)
    digitize_tdi0_z = generate_digitizer(z_edges, clip=True)
    digitize_tdi0_costhetadir = generate_digitizer(
        tdi_table_meta[0]['costhetadir_bin_edges'], clip=True
    )
    digitize_tdi0_sinphidir = generate_digitizer(
        np.sin(tdi_table_meta[0]['phidir_bin_edges']), clip=True
    )

    x_edges = tdi_table_meta[1]['x_bin_edges']
    y_edges = tdi_table_meta[1]['y_bin_edges']
    z_edges = tdi_table_meta[1]['z_bin_edges']
    tdi1_xmin, tdi1_xmax = x_edges[0, -1]
    tdi1_ymin, tdi1_ymax = y_edges[0, -1]
    tdi1_zmin, tdi1_zmax = z_edges[0, -1]
    digitize_tdi1_x = generate_digitizer(x_edges, clip=True)
    digitize_tdi1_y = generate_digitizer(y_edges, clip=True)
    digitize_tdi1_z = generate_digitizer(z_edges, clip=True)
    digitize_tdi1_costhetadir = generate_digitizer(
        tdi_table_meta[1]['costhetadir_bin_edges'], clip=True
    )
    digitize_tdi1_sinphidir = generate_digitizer(
        np.sin(tdi_table_meta[1]['phidir_bin_edges']), clip=True
    )

    # Model clock jitter and transit time spread (TTS)
    jitter_timeshifts = np.arange(-10, 11, 2)

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

    # -- Define `pexp*` closures (functions that use things defined above) -- #

    # Note in the following that we need to invert the costheta
    # direction of the sources to match the directions that
    # Retro simulation comes up with. Thus the angles
    # associated with this that we want to work with are
    #   cos(pi - thetadir) = -cos(thetadir),
    #   sin(pi - thetadir) = sin(thetadir),
    #   cos(phidir) = cos(-phidir),
    #   sin(phidir) = -sin(-phidir)
    #
    # This should be seen as a bug, but not sure how to address
    # it without modifying existing tables, so sticking with it
    # for now.
    #
    # We bin cos(pi - thetadir), so need to simply bin the
    # quantity `-src_dir_costheta`.
    #
    # We want to bin abs(deltaphidir), which is described now:
    # Just look at vectors in the xy-plane, since we want
    # difference of angle in this plane. Use dot product:
    #   dot(dir_vec_xy, pos_vec_xy) = |dir_vec_xy| |pos_vec_xy| cos(deltaphidir)
    # where the length of the directionality vector in the xy-plane is
    #   |dir_vec_xy| = rhodir = rdir * sin(pi - thetadir)
    # and since rdir = 1 and the inversion of the angle above
    #   |dir_vec_xy| = rhodir = sin(thetadir).
    # The length of the position vector in the xy-plane is
    #   |pos_vec_xy| = rho = sqrt(dx^2 + dy^2)
    # where dx and dy are src_x - dom_x and src_y - dom_y.
    # Solving for cos(deltaphidir):
    #   cos(deltaphidir) = dot(dir_vec_xy, pos_vec_xy) / (rhodir * rho)
    # we just need to write out the components of the dot
    # product in terms of quantites we have:
    #   dir_vec_x = rhodir * cos(phidir)
    #   dir_vec_y = rhodir * sin(phidir)
    #   pos_vec_x = dx
    #   pos_vec_y = dy
    # giving
    #   cos(deltaphidir) = -(rhodir*cos(phidir)*dx + rhodir*sin(phidir)*dy)/(rhodir*rho)
    # (where we use the negative to account for the inverted
    # costhetadir in the tables); cancel rhodir out
    #   cos(deltaphidir) = (cos(phidirpi)*dx + sin(phidirpi)*dy) / rho
    # and substitute the identities above
    #   cos(deltaphidir) = (cos(phidir)*dx + sin(phidir)*dy) / rho
    # Finally, solve for deltaphidir
    #   deltaphidir = acos((cos(phidir)*dx + sin(phidir)*dy) / rho)

    # A photon that starts immediately in the past (before the
    # DOM was hit) will show up in the Retro DOM tables in bin
    # 0; the further in the past the photon started, the
    # higher the time bin index. Therefore, subract source
    # time from hit time.

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_sensor_dep_t_indep_tables(
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
                dom_info = event_dom_info[op_dom_idx]
                dom_tbl_idx = dom_info['table_idx']
                dom_qe = dom_info['quantum_efficiency']
                dom_hits_start_idx = dom_info['hits_start_idx']
                dom_hits_stop_idx = dom_info['hits_stop_idx']

                dx = src_x - dom_info['x']
                dy = src_y - dom_info['y']
                dz = src_z - dom_info['z']

                rhosquared = max(MACHINE_EPS, dx**2 + dy**2)
                rsquared = rhosquared + dz**2

                if rsquared > rsquared_max:
                    continue

                r = max(MACHINE_EPS, math.sqrt(rsquared))
                r_bin_idx = digitize_r(r)

                costheta_bin_idx = digitize_costheta(dz/r)

                if src_kind == SRC_OMNI:
                    t_indep_surv_prob = np.mean(
                        t_indep_tables[dom_tbl_idx][r_bin_idx, costheta_bin_idx, :, :]
                    )

                else: # SRC_CKV_BETA1:
                    rho = math.sqrt(rhosquared)

                    if rho <= MACHINE_EPS:
                        absdeltaphidir = 0.
                    else:
                        absdeltaphidir = abs(math.acos(
                            min(1., max(-1., -(src_dir_cosphi*dx + src_dir_sinphi*dy) / rho))
                        ))

                    costhetadir_bin_idx = digitize_costhetadir(src_dir_costheta)
                    deltaphidir_bin_idx = digitize_deltaphidir(absdeltaphidir)

                    t_indep_surv_prob = t_indep_tables[dom_tbl_idx][
                        r_bin_idx,
                        costheta_bin_idx,
                        costhetadir_bin_idx,
                        deltaphidir_bin_idx
                    ]

                ti_norm = t_indep_table_norms[dom_tbl_idx][r_bin_idx]
                t_indep_exp += src_photons * ti_norm * t_indep_surv_prob * dom_qe

                for hit_idx in range(dom_hits_start_idx, dom_hits_stop_idx):
                    hit_info = event_hit_info[hit_idx]
                    if t_is_residual_time:
                        nominal_dt = hit_info['time'] - src_time - r * recip_max_group_vel
                    else:
                        nominal_dt = hit_info['time'] - src_time

                    surv_prob_at_hit_t = 0.
                    for timeshift in jitter_timeshifts:
                        shifted_dt = nominal_dt + timeshift

                        # Note the comparison is written such that it will evaluate
                        # to True if hit_time is NaN.
                        if (not shifted_dt >= 0) or shifted_dt > t_max:
                            continue

                        t_bin_idx = digitize_t(shifted_dt)

                        if src_kind == SRC_OMNI:
                            surv_prob_at_hit_t = max(
                                surv_prob_at_hit_t,
                                table_lookup_mean(
                                    tables,
                                    dom_tbl_idx,
                                    r_bin_idx,
                                    costheta_bin_idx,
                                    t_bin_idx,
                                )
                            )

                        else: # SRC_CKV_BETA1
                            surv_prob_at_hit_t = max(
                                surv_prob_at_hit_t,
                                table_lookup(
                                    tables,
                                    dom_tbl_idx,
                                    r_bin_idx,
                                    costheta_bin_idx,
                                    t_bin_idx,
                                    costhetadir_bin_idx,
                                    deltaphidir_bin_idx,
                                )
                            )

                        r_t_bin_norm = table_norms[dom_tbl_idx][r_bin_idx, t_bin_idx]
                        hit_exp[hit_idx] += (
                            src_photons * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                        )

        return t_indep_exp

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def pexp_tdi_tables(
        sources,
        sources_start,
        sources_stop,
        event_dom_info,
        event_hit_info,
        tables,
        table_norms,
        tdi_tables,
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

        tables : array or list of arrays
            Time-dependent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir)
            while if you use a template-compressed table, this will have shape
                (n_templates, n_costhetadir, n_deltaphidir)

        table_norms : shape (n_tables, n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions.

        tdi_tables : length n_tdi_tables list of arrays
            Time- and DOM-independent photon survival probability tables.

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

            if (
                tdi0_xmin <= src_x <= tdi0_xmax
                and tdi0_ymin <= src_y <= tdi0_ymax
                and tdi0_zmin <= src_z <= tdi0_zmax
            ):
                t_indep_exp += src_photons * tdi_tables[0][
                    digitize_tdi0_x(src_x),
                    digitize_tdi0_y(src_x),
                    digitize_tdi0_z(src_x),
                    digitize_tdi0_costhetadir(src_dir_costheta),
                    digitize_tdi0_sinphidir(src_dir_sinphi),
                ]
            elif (
                tdi1_xmin <= src_x <= tdi1_xmax
                and tdi1_ymin <= src_y <= tdi1_ymax
                and tdi1_zmin <= src_z <= tdi1_zmax
            ):
                t_indep_exp += src_photons * tdi_tables[1][
                    digitize_tdi1_x(src_x),
                    digitize_tdi1_y(src_x),
                    digitize_tdi1_z(src_x),
                    digitize_tdi1_costhetadir(src_dir_costheta),
                    digitize_tdi1_sinphidir(src_dir_sinphi),
                ]
            else:
                continue

            for hit_idx, hit_info in enumerate(event_hit_info):
                dom_info = event_dom_info[hit_info['event_dom_idx']]
                dom_tbl_idx = dom_info['table_idx']
                dom_qe = dom_info['quantum_efficiency']

                dx = src_x - dom_info['x']
                dy = src_y - dom_info['y']
                dz = src_z - dom_info['z']

                rhosquared = max(MACHINE_EPS, dx**2 + dy**2)
                rsquared = rhosquared + dz**2

                if rsquared > rsquared_max:
                    continue

                r = max(MACHINE_EPS, math.sqrt(rsquared))
                r_bin_idx = digitize_r(r)

                costheta_bin_idx = digitize_costheta(dz/r)

                if src_kind == SRC_CKV_BETA1:
                    rho = math.sqrt(rhosquared)

                    if rho <= MACHINE_EPS:
                        absdeltaphidir = 0.
                    else:
                        absdeltaphidir = abs(math.acos(
                            min(1., max(-1., -(src_dir_cosphi*dx + src_dir_sinphi*dy) / rho))
                        ))

                    costhetadir_bin_idx = digitize_costhetadir(src_dir_costheta)
                    deltaphidir_bin_idx = digitize_deltaphidir(absdeltaphidir)

                if t_is_residual_time:
                    nominal_dt = hit_info['time'] - src_time - r * recip_max_group_vel
                else:
                    nominal_dt = hit_info['time'] - src_time

                surv_prob_at_hit_t = 0.
                for timeshift in jitter_timeshifts:
                    shifted_dt = nominal_dt + timeshift

                    # Note the comparison is written such that it will evaluate
                    # to True if hit_time is NaN.
                    if (not shifted_dt >= 0) or shifted_dt > t_max:
                        continue

                    t_bin_idx = digitize_t(shifted_dt)

                    if src_kind == SRC_OMNI:
                        surv_prob_at_hit_t = max(
                            surv_prob_at_hit_t,
                            table_lookup_mean(
                                tables,
                                dom_tbl_idx,
                                r_bin_idx,
                                costheta_bin_idx,
                                t_bin_idx,
                            )
                        )

                    else: # SRC_CKV_BETA1
                        surv_prob_at_hit_t = max(
                            surv_prob_at_hit_t,
                            table_lookup(
                                tables,
                                dom_tbl_idx,
                                r_bin_idx,
                                costheta_bin_idx,
                                t_bin_idx,
                                costhetadir_bin_idx,
                                deltaphidir_bin_idx,
                            )
                        )

                    r_t_bin_norm = table_norms[dom_tbl_idx][r_bin_idx, t_bin_idx]
                    hit_exp[hit_idx] += (
                        src_photons * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                    )

        return t_indep_exp

    if tdi_table_meta:
        pexp = pexp_tdi_tables
    else:
        pexp = pexp_sensor_dep_t_indep_tables

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
        num_pegleg_sources = len(pegleg_sources)
        num_pegleg_llhs = 1 + int(num_pegleg_sources / pegleg_stepsize)
        num_hits = len(event_hit_info)

        # -- Expectations due to nominal (`scalefactor = 1`) scaling sources -- #

        nominal_scaling_t_indep_exp = 0.
        nominal_scaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

        nominal_scaling_t_indep_exp += pexp(
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
        nonscaling_t_indep_exp += pexp(
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

        # -- Loop initialization -- #

        llhs = np.full(shape=num_pegleg_llhs, fill_value=-np.inf, dtype=np.float64)
        llhs[0] = llh

        scalefactors = np.zeros(shape=num_pegleg_llhs, dtype=np.float64)
        scalefactors[0] = scalefactor

        best_llh = llh
        best_llh_idx = 0

        # -- Pegleg loop -- #

        for llh_idx in range(1, num_pegleg_llhs):
            pegleg_stop_idx = llh_idx * pegleg_stepsize
            pegleg_start_idx = pegleg_stop_idx - pegleg_stepsize

            # Add to expectations by including another "batch" of pegleg sources
            nonscaling_t_indep_exp += pexp(
                sources=pegleg_sources,
                sources_start=pegleg_start_idx,
                sources_stop=pegleg_stop_idx,
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

            # Store this pegleg step's LLH and best scalefactor
            llhs[llh_idx] = llh
            scalefactors[llh_idx] = scalefactor

            if llh > best_llh:
                best_llh = llh
                best_llh_idx = llh_idx

            # TODO: make this more general, less hacky continue/stop condition
            if pegleg_start_idx > 300 and llh - llhs[pegleg_start_idx - 300] < 0.5:
                break

        return best_llh, best_llh_idx * pegleg_stepsize, scalefactors[best_llh_idx]

    return pexp, get_llh, meta
