# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-locals, consider-using-enumerate

"""
Function to generate the funciton for finding expected number of photons to
survive from a 5D CLSim table.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'Minimizer',
    'StepSpacing',
    'LLHChoice',
    'MACHINE_EPS',
    'MAX_RAD_SQ',
    'SCALE_FACTOR_MINIMIZER',
    'PEGLEG_SPACING',
    'PEGLEG_BEST_DELTA_LLH_THRESHOLD',
    'USE_JITTER',
    'generate_pexp_and_llh_functions',
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
import enum
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
from retro.const import SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI, SRC_CKV_BETA1
from retro.utils.geom import generate_digitizer
from retro.hypo.discrete_cascade_kernels import SCALING_CASCADE_ENERGY


class Minimizer(enum.IntEnum):
    """Minimizer to use for scale factor"""
    GRADIENT_DESCENT = 0
    NEWTON = 1
    BINARY_SEARCH = 2

class TrackType(enum.IntEnum):
    """How to treat track energy depositions"""
    CONST = 0
    STOCHASTIC = 1

class StepSpacing(enum.IntEnum):
    """Pegleg step spacing"""
    LINEAR = 0
    LOG = 1

MACHINE_EPS = 1e-10

MAX_RAD_SQ = 500**2
"""Maximum radius to consider, squared (units of m^2)"""

SCALE_FACTOR_MINIMIZER = Minimizer.BINARY_SEARCH
"""Choice of which minimizer to use for computing scaling factor for scaling sources"""

PEGLEG_SPACING = StepSpacing.LINEAR
"""Pegleg adds segments either linearly (same number of segments independent of energy)
or logarithmically (more segments are added the longer the track"""

TRACK_TYPE = TrackType.CONST
"""`CONST` is the standard treatement, `STOCHASTIC` performs conjugate gradient minimization"""

PEGLEG_BEST_DELTA_LLH_THRESHOLD = 0.1
"""For Pegleg `LLHChoice` that require a range of LLH and average (mean, median, etc.),
take all LLH that are within this threshold of the maximum LLH"""

PEGLEG_BREAK_COUNTER = 100
"""After how many steps without improving the llh to exit the pegleg loop adding more track segments"""

MAX_CASCADE_ENERGY = 1000.
"""Maximum cascade energy (for LowEn events 1000. is a good value, for HESE 10000000.)"""

# TODO: a "proper" jitter (and transit time spread) implementation should treat each DOM
# independently and pick the time offset for each DOM that maximizes LLH (_not_ expected
# photon detections)

USE_JITTER = True
"""Whether to use a crude jitter implementation"""


def generate_pexp_and_llh_functions(
    dom_tables,
    tdi_tables=None,
    tdi_metas=None,
):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function to use.

    Parameters
    ----------
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
    pexp : callable
        Function to find detected-photon expectations given a hypothesis

    get_llh : callable

    meta : OrderedDict
        Parameters, including the binning, that uniquely identify what the
        capabilities of the returned `pexp`. (Use this to eliminate
        redundant pexp functions.)

    """
    if tdi_tables is None:
        tdi_tables = ()
    if tdi_metas is None:
        tdi_metas = ()

    tbl_is_ckv = dom_tables.table_kind in ['ckv_uncompr', 'ckv_templ_compr']
    tbl_is_templ_compr = dom_tables.table_kind in ['raw_templ_compr', 'ckv_templ_compr']
    if not tbl_is_ckv:
        raise NotImplementedError('Only Ckv tables are implemented.')

    # TODO: sanity checks that all TDI metadata is compatible with DOM tables
    for tdi_meta in tdi_metas:
        assert tdi_meta['bin_edges']['phidir'][0] == -np.pi
        assert tdi_meta['bin_edges']['phidir'][-1] == np.pi

    meta = OrderedDict()
    meta['table_kind'] = dom_tables.table_kind
    meta['table_binning'] = OrderedDict()
    for key in (
        'r_bin_edges', 'costhetadir_bin_edges', 't_bin_edges', 'costhetadir_bin_edges',
        'deltaphidir_bin_edges'
    ):
        meta['table_binning'][key] = dom_tables.table_meta[key]

    meta['tdi'] = tdi_metas
    if len(tdi_tables) == 1:
        tdi_tables = (tdi_tables[0], tdi_tables[0])

    # NOTE: For now, we only support absolute value of deltaphidir (which
    # assumes azimuthal symmetry). In future, this could be revisited (and then
    # the abs(...) applied before binning in the pexp code will have to be
    # removed or replaced with behavior that depend on the range of the
    # deltaphidir_bin_edges).
    assert dom_tables.table_meta['deltaphidir_bin_edges'][0] == 0, 'only abs(deltaphidir) supported'
    assert dom_tables.table_meta['deltaphidir_bin_edges'][-1] == np.pi

    # -- Define things used by `pexp*` closures defined below -- #

    # Constants
    rsquared_max = np.max(dom_tables.table_meta['r_bin_edges'])**2
    t_max = np.max(dom_tables.table_meta['t_bin_edges'])
    recip_max_group_vel = dom_tables.table_meta['group_refractive_index'] / SPEED_OF_LIGHT_M_PER_NS

    # Digitization functions for each binning dimension
    digitize_r = generate_digitizer(
        dom_tables.table_meta['r_bin_edges'],
        clip=True
    )
    digitize_costheta = generate_digitizer(
        dom_tables.table_meta['costheta_bin_edges'],
        clip=True
    )
    digitize_t = generate_digitizer(
        dom_tables.table_meta['t_bin_edges'],
        clip=True
    )
    digitize_costhetadir = generate_digitizer(
        dom_tables.table_meta['costhetadir_bin_edges'],
        clip=True
    )
    digitize_deltaphidir = generate_digitizer(
        dom_tables.table_meta['deltaphidir_bin_edges'],
        clip=True
    )

    num_tdi_tables = len(tdi_metas)
    if num_tdi_tables == 0:
        # Numba needs an object that it can determine type of
        tdi_tables = 0
    else:
        x_edges = tdi_metas[0]['bin_edges']['x']
        y_edges = tdi_metas[0]['bin_edges']['y']
        z_edges = tdi_metas[0]['bin_edges']['z']
        tdi0_xmin, tdi0_xmax = x_edges[[0, -1]]
        tdi0_ymin, tdi0_ymax = y_edges[[0, -1]]
        tdi0_zmin, tdi0_zmax = z_edges[[0, -1]]
        digitize_tdi0_x = generate_digitizer(x_edges, clip=True)
        digitize_tdi0_y = generate_digitizer(y_edges, clip=True)
        digitize_tdi0_z = generate_digitizer(z_edges, clip=True)
        digitize_tdi0_costhetadir = generate_digitizer(
            tdi_metas[0]['bin_edges']['costhetadir'], clip=True
        )
        digitize_tdi0_phidir = generate_digitizer(
            tdi_metas[0]['bin_edges']['phidir'], clip=True
        )

        if num_tdi_tables == 1:
            idx = 0
        elif num_tdi_tables == 2:
            idx = 1
        else:
            raise ValueError(
                'Can only handle 0, 1, or 2 TDI tables; got {}'
                .format(num_tdi_tables)
            )

        x_edges = tdi_metas[idx]['bin_edges']['x']
        y_edges = tdi_metas[idx]['bin_edges']['y']
        z_edges = tdi_metas[idx]['bin_edges']['z']
        tdi1_xmin, tdi1_xmax = x_edges[[0, -1]]
        tdi1_ymin, tdi1_ymax = y_edges[[0, -1]]
        tdi1_zmin, tdi1_zmax = z_edges[[0, -1]]
        digitize_tdi1_x = generate_digitizer(x_edges, clip=True)
        digitize_tdi1_y = generate_digitizer(y_edges, clip=True)
        digitize_tdi1_z = generate_digitizer(z_edges, clip=True)
        digitize_tdi1_costhetadir = generate_digitizer(
            tdi_metas[idx]['bin_edges']['costhetadir'], clip=True
        )
        digitize_tdi1_phidir = generate_digitizer(
            tdi_metas[idx]['bin_edges']['phidir'], clip=True
        )

    dom_tables_ = dom_tables

    dom_tables = dom_tables_.tables
    dom_table_norms = dom_tables_.table_norms
    dom_tables_template_library = dom_tables_.template_library
    t_indep_dom_tables = dom_tables_.t_indep_tables
    t_indep_dom_table_norms = dom_tables_.t_indep_table_norms
    t_is_residual_time = dom_tables_.t_is_residual_time

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

    if USE_JITTER:
        # Time offsets to sample for DOM jitter
        jitter_dt = np.arange(-10, 11, 2)

        # Weight at each time offset
        jitter_weights = stats.norm.pdf(jitter_dt, 0, 5)
        jitter_weights /= np.sum(jitter_weights)
    else:
        jitter_dt = np.array([0.])
        jitter_weights = np.array([1.])
    num_jitter_time_offsets = len(jitter_dt)

    # Indexing functions for table types omni / directional lookups
    if tbl_is_templ_compr:
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx
        ): # pylint: disable=missing-docstring
            templ = tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx]
            return templ['weight'] / dom_tables_template_library[templ['index']].size

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
            costhetadir_bin_idx, deltaphidir_bin_idx
        ): # pylint: disable=missing-docstring
            templ = tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx]
            return (
                templ['weight'] * dom_tables_template_library[
                    templ['index'],
                    costhetadir_bin_idx,
                    deltaphidir_bin_idx,
                ]
            )

    else: # table is not template-compressed
        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup_mean(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx
        ): # pylint: disable=missing-docstring
            return np.mean(
                tables[table_idx][r_bin_idx, costheta_bin_idx, t_bin_idx]
            )

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def table_lookup(
            tables, table_idx, r_bin_idx, costheta_bin_idx, t_bin_idx,
            costhetadir_bin_idx, deltaphidir_bin_idx
        ): # pylint: disable=missing-docstring
            return tables[table_idx][
                r_bin_idx,
                costheta_bin_idx,
                t_bin_idx,
                costhetadir_bin_idx,
                deltaphidir_bin_idx,
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

    # TODO: integrate tdi into same pexp function?

    pexp_docstr = (
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

        sources_start, sources_stop : int
            Starting and stopping indices for the part of the array on which to
            work. Note that the latter is exclusive, i.e., following Python
            range / slice syntax. Hence, the following section of `sources` will
            be operated upon: .. ::

                sources[sources_start:sources_stop]

        event_dom_info : shape (n_operational_doms,) array of dtype EVT_DOM_INFO_T

        event_hit_info : shape (n_hits,) array of dtype EVT_HIT_INFO_T

        hit_exp : shape (n_hits,) array of floats
            Time-dependent hit expectation at each (actual) hit time;
            initialize outside of this function, as values are incremented
            within this function. Values in `hit_exp` correspond to the values
            in `event_hit_info`.

        dom_tables : array
            DOM time-dependent photon survival probability tables. If using an
            uncompressed table, these will have shape
                (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir)
            while if you use a template-compressed table, this will have shape
                (n_templates, n_costhetadir, n_deltaphidir)

        dom_table_norms : shape (n_tables, n_r, n_t) array
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions.

        t_indep_dom_tables : array
            Time-independent photon survival probability table. If using an
            uncompressed table, this will have shape
                (n_r, n_costheta, n_costhetadir, n_deltaphidir)
            while if using a

        t_indep_dom_dom_table_norms : shape (n_tables, n_r) array
            r-dependent normalization (any t-dep normalization is assumed to
            already have been applied to generate the t_indep_table).

        tdi_tables : {type}
            {text}

        Returns
        -------
        t_indep_exp : float
            Expectation of total hits for all operational DOMs

        Out
        ---
        hit_exp
            `hit_exp` is modified by the function; see Parameters section for
            detailed explanation of parameter `hit_exp`

        """
    )

    if num_tdi_tables == 0:

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def pexp_(
            sources,
            sources_start,
            sources_stop,
            event_dom_info,
            event_hit_info,
            hit_exp,
            dom_tables,
            dom_table_norms,
            t_indep_dom_tables,
            t_indep_dom_table_norms,
            tdi_tables, # pylint: disable=unused-argument
        ): # pylint: disable=missing-docstring, too-many-arguments
            num_operational_doms = len(event_dom_info)
            t_indep_exp = 0.
            for source_idx in range(sources_start, sources_stop):
                src = sources[source_idx]

                for op_dom_idx in range(num_operational_doms):
                    dom_info = event_dom_info[op_dom_idx]
                    dom_tbl_idx = dom_info['table_idx']
                    dom_qe = dom_info['quantum_efficiency']
                    dom_hits_start_idx = dom_info['hits_start_idx']
                    dom_hits_stop_idx = dom_info['hits_stop_idx']

                    dx = src['x'] - dom_info['x']
                    dy = src['y'] - dom_info['y']
                    dz = src['z'] - dom_info['z']

                    rhosquared = max(MACHINE_EPS, dx**2 + dy**2)
                    rsquared = rhosquared + dz**2

                    # first thing to check if this DOM is out of range and we can skip it
                    if rsquared > rsquared_max:
                        continue

                    r = max(MACHINE_EPS, math.sqrt(rsquared))
                    r_bin_idx = digitize_r(r)

                    costheta_bin_idx = digitize_costheta(dz/r)

                    if src['kind'] == SRC_OMNI:
                        t_indep_surv_prob = np.mean(
                            t_indep_dom_tables[dom_tbl_idx][r_bin_idx, costheta_bin_idx, :, :]
                        )

                    else: # SRC_CKV_BETA1:
                        rho = math.sqrt(rhosquared)

                        if rho <= MACHINE_EPS:
                            absdeltaphidir = 0.
                        else:
                            absdeltaphidir = abs(math.acos(
                                max(-1., min(1., -(src['dir_cosphi']*dx + src['dir_sinphi']*dy) / rho))
                            ))

                        costhetadir_bin_idx = digitize_costhetadir(src['dir_costheta'])
                        deltaphidir_bin_idx = digitize_deltaphidir(absdeltaphidir)

                        t_indep_surv_prob = t_indep_dom_tables[dom_tbl_idx][
                            r_bin_idx,
                            costheta_bin_idx,
                            costhetadir_bin_idx,
                            deltaphidir_bin_idx
                        ]

                    ti_norm = t_indep_dom_table_norms[dom_tbl_idx][r_bin_idx]
                    t_indep_exp += src['photons'] * ti_norm * t_indep_surv_prob * dom_qe

                    for hit_idx in range(dom_hits_start_idx, dom_hits_stop_idx):
                        hit_info = event_hit_info[hit_idx]
                        if t_is_residual_time:
                            nominal_dt = hit_info['time'] - src['time'] - r * recip_max_group_vel
                        else:
                            nominal_dt = hit_info['time'] - src['time']

                        for jitter_idx in range(num_jitter_time_offsets):
                            dt = nominal_dt + jitter_dt[jitter_idx]

                            # Note the comparison is written such that it will evaluate
                            # to True if `dt` is NaN or less than zero.
                            if (not dt >= 0) or dt > t_max:
                                continue

                            t_bin_idx = digitize_t(dt)

                            if src['kind'] == SRC_OMNI:
                                surv_prob_at_hit_t = table_lookup_mean(
                                    tables=dom_tables,
                                    table_idx=dom_tbl_idx,
                                    r_bin_idx=r_bin_idx,
                                    costheta_bin_idx=costheta_bin_idx,
                                    t_bin_idx=t_bin_idx,
                                )

                            else: # SRC_CKV_BETA1
                                surv_prob_at_hit_t = table_lookup(
                                    tables=dom_tables,
                                    table_idx=dom_tbl_idx,
                                    r_bin_idx=r_bin_idx,
                                    costheta_bin_idx=costheta_bin_idx,
                                    t_bin_idx=t_bin_idx,
                                    costhetadir_bin_idx=costhetadir_bin_idx,
                                    deltaphidir_bin_idx=deltaphidir_bin_idx,
                                )

                            r_t_bin_norm = dom_table_norms[dom_tbl_idx][r_bin_idx, t_bin_idx]
                            hit_exp[hit_idx] += jitter_weights[jitter_idx] * (
                                src['photons'] * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                            )

            return t_indep_exp

        pexp_.__doc__ = pexp_docstr.format(
            type='int',
            text="""Dummy argument for this version of `pexp` since it doesn't use TDI
            tables (but this argument needs to be present to maintain same
            interface)"""
        )

    else: # pexp function given we are using TDI tables

        @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
        def pexp_(
            sources,
            sources_start,
            sources_stop,
            event_dom_info,
            event_hit_info,
            hit_exp,
            dom_tables,
            dom_table_norms,
            t_indep_dom_tables, # pylint: disable=unused-argument
            t_indep_dom_table_norms, # pylint: disable=unused-argument
            tdi_tables,
        ): # pylint: disable=missing-docstring, too-many-arguments
            # -- Time- and DOM-independent photon-detection expectation -- #

            t_indep_exp = 0.
            for source_idx in range(sources_start, sources_stop):
                src = sources[source_idx]
                src_opposite_dir_costheta = -src['dir_costheta']
                src_opposite_dir_phi = ((src['dir_phi'] + 2*np.pi) % (2*np.pi)) - np.pi

                if (
                    tdi0_xmin <= src['x'] <= tdi0_xmax
                    and tdi0_ymin <= src['y'] <= tdi0_ymax
                    and tdi0_zmin <= src['z'] <= tdi0_zmax
                ):
                    t_indep_exp += 0.45 * src['photons'] * tdi_tables[0][
                        digitize_tdi0_x(src['x']),
                        digitize_tdi0_y(src['y']),
                        digitize_tdi0_z(src['z']),
                        digitize_tdi0_costhetadir(src_opposite_dir_costheta),
                        digitize_tdi0_phidir(src_opposite_dir_phi),
                    ]
                elif num_tdi_tables >= 2 and (
                    tdi1_xmin <= src['x'] <= tdi1_xmax
                    and tdi1_ymin <= src['y'] <= tdi1_ymax
                    and tdi1_zmin <= src['z'] <= tdi1_zmax
                ):
                    t_indep_exp += 0.45 * src['photons'] * tdi_tables[1][
                        digitize_tdi1_x(src['x']),
                        digitize_tdi1_y(src['y']),
                        digitize_tdi1_z(src['z']),
                        digitize_tdi1_costhetadir(src_opposite_dir_costheta),
                        digitize_tdi1_phidir(src_opposite_dir_phi),
                    ]
                else:
                    continue

            # -- Time-dependent photon-det expectation for each hit DOM -- #

            for hit_idx, hit_info in enumerate(event_hit_info):
                dom_info = event_dom_info[hit_info['event_dom_idx']]
                dom_tbl_idx = dom_info['table_idx']
                dom_qe = dom_info['quantum_efficiency']

                for source_idx in range(sources_start, sources_stop):
                    src = sources[source_idx]

                    dx = src['x'] - dom_info['x']
                    dy = src['y'] - dom_info['y']
                    dz = src['z'] - dom_info['z']

                    rhosquared = max(MACHINE_EPS, dx**2 + dy**2)
                    rsquared = rhosquared + dz**2

                    if rsquared > rsquared_max:
                        continue

                    r = max(MACHINE_EPS, math.sqrt(rsquared))
                    r_bin_idx = digitize_r(r)

                    costheta_bin_idx = digitize_costheta(dz/r)

                    if src['kind'] == SRC_CKV_BETA1:
                        rho = math.sqrt(rhosquared)

                        if rho <= MACHINE_EPS:
                            absdeltaphidir = 0.
                        else:
                            absdeltaphidir = abs(math.acos(
                                max(-1., min(1., -(src['dir_cosphi']*dx + src['dir_sinphi']*dy) / rho))
                            ))

                        costhetadir_bin_idx = digitize_costhetadir(src['dir_costheta'])
                        deltaphidir_bin_idx = digitize_deltaphidir(absdeltaphidir)

                    if t_is_residual_time:
                        nominal_dt = hit_info['time'] - src['time'] - r * recip_max_group_vel
                    else:
                        nominal_dt = hit_info['time'] - src['time']

                    # Note: caching last `t_bin_idx`, `r_t_bin_norm`, and
                    # `surv_prob_at_hit_t` and checking for identical `t_bin_idx` seems
                    # to take about the same time as not caching these values, so
                    # choosing the simpler way

                    for jitter_idx in range(num_jitter_time_offsets):
                        dt = nominal_dt + jitter_dt[jitter_idx]

                        # Note the comparison is written such that it will evaluate to
                        # True if `dt` is NaN or less than zero.
                        if (not dt >= 0) or dt > t_max:
                            continue

                        t_bin_idx = digitize_t(dt)

                        if src['kind'] == SRC_OMNI:
                            surv_prob_at_hit_t = table_lookup_mean(
                                tables=dom_tables,
                                table_idx=dom_tbl_idx,
                                r_bin_idx=r_bin_idx,
                                costheta_bin_idx=costheta_bin_idx,
                                t_bin_idx=t_bin_idx,
                            )

                        else: # SRC_CKV_BETA1
                            surv_prob_at_hit_t = table_lookup(
                                tables=dom_tables,
                                table_idx=dom_tbl_idx,
                                r_bin_idx=r_bin_idx,
                                costheta_bin_idx=costheta_bin_idx,
                                t_bin_idx=t_bin_idx,
                                costhetadir_bin_idx=costhetadir_bin_idx,
                                deltaphidir_bin_idx=deltaphidir_bin_idx,
                            )

                        r_t_bin_norm = dom_table_norms[dom_tbl_idx][r_bin_idx, t_bin_idx]
                        hit_exp[hit_idx] += jitter_weights[jitter_idx] * (
                            src['photons'] * r_t_bin_norm * surv_prob_at_hit_t * dom_qe
                        )

            return t_indep_exp

        pexp_.__doc__ = pexp_docstr.format(
            type='tuple of 1 or 2 arrays',
            text="""TDI tables"""
        )

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
            containing all relevant event per DOM info
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
    ):
        """Find optimal (highest-likelihood) `scalefactor` for scaling sources.

        Parameters:
        -----------
        event_dom_info : array of dtype EVT_DOM_INFO_T
            containing all relevant event per DOM info
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
        if SCALE_FACTOR_MINIMIZER in (Minimizer.GRADIENT_DESCENT, Minimizer.BINARY_SEARCH):

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
            scalefactor = max(0., min(MAX_CASCADE_ENERGY / SCALING_CASCADE_ENERGY, scalefactor))

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
                last = MAX_CASCADE_ENERGY/SCALING_CASCADE_ENERGY
                last_grad = get_grad_neg_llh_wrt_scalefactor(last)
                if last_grad < 0 or abs(last_grad) < epsilon:
                    scalefactor = last
                    done = True
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

        elif SCALE_FACTOR_MINIMIZER is Minimizer.NEWTON:

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
                    s = (
                        hit_info['charge'] * nominal_scaling_hit_exp[hit_idx]
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
                return numerator / denominator

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
            scalefactor = max(0., min(MAX_CASCADE_ENERGY/SCALING_CASCADE_ENERGY, scalefactor))

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
    def grad(
        sfs,
        event_dom_info,
        event_hit_info,
        nominal_scaling_hit_exps,
        nominal_scaling_t_indep_exps,
        idx,
    ):
        """same as get_grad_neg_llh_wrt_scalefactor, just otput as array"""
        g = np.zeros(shape=(idx,))
        #g = np.zeros(1)
        # Time- and DOM-independent part of grad(-LLH)
        g += nominal_scaling_t_indep_exps[:idx]

        # Time-dependent part of grad(-LLH) (i.e., at hit times)
        for hit_idx, hit_info in enumerate(event_hit_info):
            norm = (
                event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                + np.sum(sfs[:idx] * nominal_scaling_hit_exps[:idx,hit_idx])
            )
            g -= hit_info['charge'] * nominal_scaling_hit_exps[:idx,hit_idx] / norm
        return g

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def fun(
        sfs,
        event_dom_info,
        event_hit_info,
        nominal_scaling_hit_exps,
        nominal_scaling_t_indep_exps,
        idx,
    ):
        """llh function"""
        # Time- and DOM-independent part of LLH
        llh = - np.sum(sfs[:idx] * nominal_scaling_t_indep_exps[:idx])

        # Time-dependent part of LLH (i.e., at hit times)
        for hit_idx, hit_info in enumerate(event_hit_info):
            llh += hit_info['charge'] * math.log(
                event_dom_info[hit_info['event_dom_idx']]['noise_rate_per_ns']
                + np.sum(sfs[:idx] * nominal_scaling_hit_exps[:idx,hit_idx])
            )
        return -llh

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_optimal_scalefactors(
        event_dom_info,
        event_hit_info,
        nominal_scaling_hit_exps,
        nominal_scaling_t_indep_exps,
        scalefacots,
        idx,
    ):
        """Find optimal (highest-likelihood) `scalefacots` for n scaling sources.

        Parameters:
        -----------
        event_dom_info : array of dtype EVT_DOM_INFO_T
            containing all relevant event per DOM info
        event_hit_info : array of dtype EVT_HIT_INFO_T
        nominal_scaling_hit_exps : shape (n_sources, n_hits, 2) array of dtype float
            Detected-charge-rate expectation at each hit time due to scaling sources at
            nominal values (i.e., with `scalefactor = 1`); this quantity is
            lambda_d^s(t_{k_d}) in `likelihood_function_derivation.ipynb`
        nominal_scaling_t_indep_exps : shape (n_sources, )
            Total charge expected across the detector due to nominal scaling sources
            (Lambda^s in `likelihood_function_derivation.ipynb`)
        scalefacots : shape (n_sources)
            Starting point for minimizer
        idx : int
            up to which index to consider sources and scalefacors

        Returns
        -------
        llh

        """


        def line_search_interpolation(g, h, a0, p0):
            """perform line search using interpolation strategy

            Parameters:
            -----------
            g : array
                gradient vector
            h : array
                search vector
            a0 : float
                starting distance
            p : array
                starting position

            Returns:
            --------
            float, optimal a value

            Original License:
            -------
            MIT License

            Copyright (c) 2018 Ivo Filot

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.

            """
            delta = 1e-4
            deriv = np.dot(g,h)

            p0[p0 < 1] = 1.
            e0 = fun(
                p0,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )

            p1 = p0 + a0 * h
            p1[p1 < 1] = 1.
            e1 = fun(
                p1,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )

            if e1 < e0 + delta * a0 * deriv:
                return a0

            a1 = -deriv * a0**2 / (2. * (e1 - e0 - deriv * a0))
            p2 = p0 + a1 * h
            p2[p2 < 1] = 1.
            e2 = fun(
                p2,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )

            if e2 < e0 + delta * a1 * deriv:
                return a1

            aa = 1. / (a0**2 * a1**2 * (a1 - a0)) * (a0**2 * (e2 - e0 - deriv * a1) - a1**2 * (e1 - e0 - deriv * a0))
            bb = 1. / (a0**2 * a1**2 * (a1 - a0)) * (-a0**3 * (e2 - e0 - deriv * a1) + a1**3 * (e1 - e0 - deriv * a0))
            a2 = -bb + math.sqrt(bb**2 - 3. * aa * deriv) / (3. * aa)
            if a2 < 0:
                a2 = a1 / 2.
            p3 = p0 + a2 * h
            p3[p3 < 1] = 1.
            e3 = fun(
                p3,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )

            if e3 < e0 + delta * a2 * deriv:
                return a2

            return 0.

        # -- Conjugate gradient optimization -- #

        iter_num = 0
        g = grad(
            scalefacots,
            event_dom_info,
            event_hit_info,
            nominal_scaling_hit_exps,
            nominal_scaling_t_indep_exps,
            idx,
        )
        llh_old = fun(
            scalefacots,
            event_dom_info,
            event_hit_info,
            nominal_scaling_hit_exps,
            nominal_scaling_t_indep_exps,
            idx,
        )
        norm_g = np.linalg.norm(g)
        if norm_g == 0:
            return -llh_old
        h = -g / norm_g
        maxlinesearch = 2.0

        while iter_num < 300:
            p0 = np.copy(scalefacots[:idx])
            A = line_search_interpolation(g, h, maxlinesearch, p0)

            #for s in range(len(scalefacots)):
            #    scalefacots[s] += A * h[s]
            scalefacots[:idx] += A * h[:idx]
            scalefacots[scalefacots < 1] = 1.

            g1 = grad(
                scalefacots,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )
            llh_new = fun(
                scalefacots,
                event_dom_info,
                event_hit_info,
                nominal_scaling_hit_exps,
                nominal_scaling_t_indep_exps,
                idx,
            )

            if np.fabs(llh_new - llh_old) < 1e-3:
                break

            llh_old = llh_new

            # check angle between two vectors
            norm = np.linalg.norm(g1) * np.linalg.norm(h)
            if norm > 0:
                angle = np.arccos(np.dot(-g1, h) / norm)

                if angle > (math.pi / 4.):
                    oldg = np.copy(g)
                    g = np.copy(g1)
                    beta = np.dot(g, g - oldg) / np.dot(oldg, oldg)
                    h = -g + beta * h

            iter_num += 1

        #print(scalefacots)
        #print(iter_num)
        return -llh_new

    @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
    def get_llh_(
        generic_sources,
        pegleg_sources,
        scaling_sources,
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
            `pegleg_stop_idx` is designed to be fed to
            :func:`retro.hypo.discrete_muon_kernels.pegleg_eval`
        scalefactor : float
            Best scale factor for `scaling_sources` at best pegleg hypo
        zero_dllh : float >=0
            delta LLH of best fit pegleg LLH to LLH of zero length track
        lower_dllh : float >= 0
            delta LLH of best fit pegleg LLH to LLH `PEGLEG_BREAK_COUNTER` track steps before best LLH
        upper_dllh : float >= 0
            delta LLH of best fit pegleg LLH to LLH `PEGLEG_BREAK_COUNTER` track steps after best LLH

        """
        if TRACK_TYPE == TrackType.CONST:
            num_pegleg_sources = len(pegleg_sources)
            num_pegleg_steps = 1 + int(num_pegleg_sources / pegleg_stepsize)
            num_scaling_sources = len(scaling_sources)
            num_hits = len(event_hit_info)

            if num_scaling_sources > 0:
                # -- Storage for exp due to nominal (`scalefactor = 1`) scaling sources -- #
                nominal_scaling_t_indep_exp = 0.
                nominal_scaling_hit_exp = np.zeros(shape=num_hits, dtype=np.float64)

                nominal_scaling_t_indep_exp += pexp_(
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
                nonscaling_t_indep_exp += pexp_(
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
                    0.,
                    0.,
                    0.,
                )

            # -- Pegleg loop -- #
            if PEGLEG_SPACING is StepSpacing.LOG:
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

            # -- Loop initialization -- #

            num_llhs = num_pegleg_steps + 1
            llhs = np.full(shape=num_llhs, fill_value=-np.inf, dtype=np.float64)
            llhs[0] = llh

            all_scalefactors = np.zeros(shape=num_llhs, dtype=np.float64)
            all_scalefactors[0] = scalefactor

            best_llh = llh
            previous_llh = best_llh - 100
            pegleg_max_llh_step = 0
            getting_worse_counter = 0

            for pegleg_step in range(1, num_pegleg_steps):
                pegleg_stop_idx = pegleg_step * pegleg_stepsize
                pegleg_start_idx = pegleg_stop_idx - pegleg_stepsize

                # Add to expectations by including another "batch" or segment of pegleg
                # sources
                nonscaling_t_indep_exp += pexp_(
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
                all_scalefactors[pegleg_step] = scalefactor

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
                if getting_worse_counter > PEGLEG_BREAK_COUNTER:
                    #for idx in range(pegleg_idx+1,n_pegleg_steps):
                    #    # fill up with bad llhs. just to make sure they're not used
                    #    llhs[idx] = best_llh - 100
                    #print('break at step ',pegleg_idx)
                    break

            lower_idx = max(0, pegleg_max_llh_step - PEGLEG_BREAK_COUNTER)
            upper_idx = min(num_llhs, pegleg_max_llh_step + PEGLEG_BREAK_COUNTER)
            return (
                llhs[pegleg_max_llh_step],
                pegleg_max_llh_step * pegleg_stepsize,
                all_scalefactors[pegleg_max_llh_step],
                llhs[pegleg_max_llh_step] - llhs[0],
                llhs[pegleg_max_llh_step] - llhs[lower_idx],
                llhs[pegleg_max_llh_step] - llhs[upper_idx],
            )

        else:
            # let's do CGD
            num_hits = len(event_hit_info)

            n_opt_segments = 100

            nominal_scaling_t_indep_exps = np.zeros(n_opt_segments, dtype=np.float64)
            nominal_scaling_hit_exps = np.zeros(shape=(n_opt_segments, num_hits), dtype=np.float64)

            scalefacots = np.zeros(shape=(n_opt_segments,))

            llhs = np.full(shape=n_opt_segments, fill_value=-np.inf, dtype=np.float64)
            mean_scalefactor = np.zeros(shape=n_opt_segments)

            best_llh = -np.inf
            getting_worse_counter = 0
            sources_per_segment = 3

            for n in range(n_opt_segments):
                # fill up exps
                start = sources_per_segment*n
                stop = sources_per_segment*(n+1)
                nominal_scaling_t_indep_exps[n] = pexp_(
                    sources=pegleg_sources,
                    sources_start=start,
                    sources_stop=stop,
                    event_dom_info=event_dom_info,
                    event_hit_info=event_hit_info,
                    hit_exp=nominal_scaling_hit_exps[n],
                    dom_tables=dom_tables,
                    dom_table_norms=dom_table_norms,
                    t_indep_dom_tables=t_indep_dom_tables,
                    t_indep_dom_table_norms=t_indep_dom_table_norms,
                    tdi_tables=tdi_tables,
                )

                llh = get_optimal_scalefactors(
                    event_dom_info=event_dom_info,
                    event_hit_info=event_hit_info,
                    nominal_scaling_hit_exps=nominal_scaling_hit_exps,
                    nominal_scaling_t_indep_exps=nominal_scaling_t_indep_exps,
                    scalefacots=scalefacots,
                    idx=n+1,
                )

                llhs[n] = llh
                mean_scalefactor[n] = np.sum(scalefacots[:n+1])/(n+1)

                if llh > best_llh:
                    best_llh = llh
                    getting_worse_counter = 0
                else:
                    getting_worse_counter += 1
                if getting_worse_counter == 3:
                    break

                #print(n, llh, scalefacots[:n+1])

            #print(n, np.sum(scalefacots[:n+1])/n)

            best_idx = n - getting_worse_counter

            return (
                llhs[best_idx],
                sources_per_segment*best_idx,
                mean_scalefactor[best_idx],
                0.,
                0.,
                0.,
            )


    # -- Define pexp and get_llh closures, baking-in the tables -- #

    # Note: faster to _not_ jit-compile this function (why, though?)
    def pexp(
        sources,
        sources_start,
        sources_stop,
        event_dom_info,
        event_hit_info,
        hit_exp,
    ):
        return pexp_(
            sources=sources,
            sources_start=sources_start,
            sources_stop=sources_stop,
            event_dom_info=event_dom_info,
            event_hit_info=event_hit_info,
            hit_exp=hit_exp,
            dom_tables=dom_tables,
            dom_table_norms=dom_table_norms,
            t_indep_dom_tables=t_indep_dom_tables,
            t_indep_dom_table_norms=t_indep_dom_table_norms,
            tdi_tables=tdi_tables,
        )

    # Note: numba fails w/ TDI tables if this is set to be jit-compiled (why?)
    def get_llh(
        generic_sources,
        pegleg_sources,
        scaling_sources,
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
        zero_dllh : float >=0
            delta LLH of best fit pegleg LLH to LLH of zero length track
        lower_dllh : float >= 0
            delta LLH of best fit pegleg LLH to LLH `PEGLEG_BREAK_COUNTER` track steps before best LLH
        upper_dllh : float >= 0
            delta LLH of best fit pegleg LLH to LLH `PEGLEG_BREAK_COUNTER` track steps after best LLH

        """
        return get_llh_(
            generic_sources=generic_sources,
            pegleg_sources=pegleg_sources,
            scaling_sources=scaling_sources,
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

    return pexp, get_llh, meta
