#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then tales of icecube hdf5 files with events (hits
series) in them as inputs and calculates lilkelihoods.

At the moment, these likelihoods can be single points or 1d or 2d scans.
"""


from __future__ import absolute_import, division, print_function


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


from argparse import ArgumentParser
from collections import OrderedDict, Sequence
from copy import deepcopy
import cPickle as pickle
from itertools import izip, product
from os import makedirs
from os.path import abspath, dirname, isdir, join
import sys
import time

import numba # pylint: disable=unused-import
import numpy as np
from pyswarm import pso

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import DC_DOM_JITTER_NS, IC_DOM_JITTER_NS # pylint: disable=unused-import
from retro import FTYPE, HYPO_PARAMS_T, HypoParams10D
from retro import DETECTOR_GEOM_FILE
from retro import (event_to_hypo_params, expand, poisson_llh,
                   get_primary_interaction_str)
from retro.events import Events
from retro.discrete_hypo import DiscreteHypo
from retro.discrete_muon_kernels import const_energy_loss_muon # pylint: disable=unused-import
from retro.discrete_cascade_kernels import point_cascade # pylint: disable=unused-import
from retro.plot_1d_scan import plot_1d_scan
from retro.table_readers import DOMTimePolarTables, TDICartTable # pylint: disable=unused-import


DFLT_EVENTS_FPATH = (
    '/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600'
    '/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5'
)

# (Almost) completely arbitrary first guesses
EPS = 1.0
EPS_STAT = EPS
EPS_CLSIM_LENGTH_BINNING = EPS
EPS_CLSIM_ANGLE_BINNING = EPS
NOISE_CHARGE = 0.00000025 * 100
CASCADE_E_SCALE = 1
TRACK_E_SCALE = 1
NUM_JITTER_SAMPLES = 1
JITTER_SIGMA = 5

NUM_SCAN_POINTS = 100
HypoClass = DiscreteHypo
HYPOCLASS_KWARGS = dict(time_increment=1)
LLH_USE_AVGPHOT = False
LLH_USE_NOHIT = False

RESULTS_DIR = (
    'results_avgphot%d_nohit%d%s_cesc%d_tesc%d_noise%.1e_jitsamp%d%s'
    % (LLH_USE_AVGPHOT,
       LLH_USE_NOHIT,
       '_eps%.1f' % EPS if LLH_USE_AVGPHOT else '',
       CASCADE_E_SCALE,
       TRACK_E_SCALE,
       NOISE_CHARGE,
       NUM_JITTER_SAMPLES,
       '_jitsig%d' % JITTER_SIGMA if NUM_JITTER_SAMPLES > 1 else '')
)
if not isdir(RESULTS_DIR):
    makedirs(expand(RESULTS_DIR), mode=0o2777)

ABS_BOUNDS = HypoParams10D(
    t=(-1000, 1e6),
    x=(-700, 700),
    y=(-700, 700),
    z=(-800, 600),
    track_azimuth=(0, 2*np.pi),
    track_zenith=(-np.pi, np.pi),
    track_energy=(0, 100),
    cascade_azimuth=(0, 2*np.pi),
    cascade_zenith=(-np.pi, np.pi),
    cascade_energy=(0, 100)
)
"""Absolute bounds for scanning / minimizer to work within"""

REL_BOUNDS = HypoParams10D(
    t=(-300, 300),
    #t=(-1000, 1000),
    x=(-100, 100),
    y=(-100, 100),
    z=(-100, 100),
    track_azimuth=None,
    track_zenith=None,
    track_energy=None,
    cascade_azimuth=None,
    cascade_zenith=None,
    cascade_energy=None
)
"""Relative bounds defined about "truth" values for scanning / minimization"""

MIN_USE_RELATIVE_BOUNDS = False
"""Whether minimizer is to use relative bounds (where defined)"""

SCAN_USE_RELATIVE_BOUNDS = True
"""Whether scanning is to use relative bounds (where defined)"""

MIN_DIMS = [] #('t x y z track_zenith track_azimuth track_energy'.split())
"""Which dimensions to plug into minimizer (dims not fixed to truth)"""

SCAN_DIM_SETS = (
    't',
    'x',
    'y',
    'z',
    'track_zenith',
    'track_azimuth',
    'track_energy',
    'cascade_energy',
    #('t', 'x'),
    #('t', 'y'),
    #('t', 'z'),
    #('x', 'y'),
    #('x', 'z'),
    #('track_zenith', 'track_azimuth'),
    #('track_zenith', 'z'),
)
"""Which dimensions to scan. Tuples specify 2+ dimension scans"""


#@profile
def get_neg_llh(pinfo_gen, event, dom_tables, tdi_table=None,
                detailed_info_list=None):
    """Get log likelihood.

    Parameters
    ----------
    pinfo_gen

    event : retro.Event namedtuple or convertible thereto

    dom_tables

    tdi_table

    detailed_info_list : None or appendable sequence
        If a list is provided, it is appended with a dict containing detailed
        info from the calculation useful, e.g., for debugging. If ``None`` is
        passed, no detailed info is made available.

    Returns
    -------
    llh : float
        Negative of the log likelihood

    """
    t0 = time.time()

    neg_llh = 0
    noise_counts = 0

    eps_angle = EPS_STAT + EPS_CLSIM_ANGLE_BINNING
    eps_length = EPS_STAT + EPS_CLSIM_LENGTH_BINNING

    if tdi_table is not None:
        total_expected_q = tdi_table.get_photon_expectation(pinfo_gen=pinfo_gen)
    else:
        total_expected_q = 0

    expected_q_accounted_for = 0

    # Loop over pulses (aka hits) to get likelihood of those hits coming from
    # the hypo
    for string, depth_idx, pulse_time, pulse_charge in izip(*event.pulses):
        expected_charge = dom_tables.get_photon_expectation(
            pinfo_gen=pinfo_gen,
            hit_time=pulse_time,
            string=string,
            depth_idx=depth_idx
        )

        expected_charge_excluding_noise = expected_charge

        if expected_charge < NOISE_CHARGE:
            noise_counts += 1
            # "Add" in noise (i.e.: expected charge must be at least as
            # large as noise level)
            expected_charge = NOISE_CHARGE

        # Poisson log likelihood (take negative to interface w/ minimizers)
        pulse_neg_llh = -poisson_llh(expected=expected_charge,
                                     observed=pulse_charge)

        neg_llh += pulse_neg_llh
        expected_q_accounted_for += expected_charge

    # Penalize the likelihood (_add_ to neg_llh) by expected charge that
    # would be seen by DOMs other than those hit (by the physics event itself,
    # i.e. non-noise hits). This is the unaccounted-for excess predicted by the
    # hypothesis.
    unaccounted_excess_expected_q = total_expected_q - expected_q_accounted_for
    if tdi_table is not None:
        if unaccounted_excess_expected_q > 0:
            print('neg_llh before correction    :', neg_llh)
            print('unaccounted_excess_expected_q:',
                  unaccounted_excess_expected_q)
            neg_llh += unaccounted_excess_expected_q
            print('neg_llh after correction     :', neg_llh)
            print('')
        else:
            print('WARNING!!!! DOM tables account for %e expected charge, which'
                  ' exceeds total expected from TDI tables of %e'
                  % (expected_q_accounted_for, total_expected_q))
            #raise ValueError()

    # Record details if user passed a list for storing them
    if detailed_info_list is not None:
        detailed_info = dict(
            noise_counts=noise_counts,
            total_expected_q=total_expected_q,
            expected_q_accounted_for=expected_q_accounted_for,
        )
        detailed_info_list.append(detailed_info)

    #print('time to compute likelihood: %.5f ms' % ((time.time() - t0) * 1000))
    return neg_llh


def scan(hypo_obj, event, neg_llh_func, dims, scan_values, nominal_params=None,
         neg_llh_func_kwargs=None):
    """Scan likelihoods for hypotheses changing one parameter dimension.

    Parameters
    ----------
    hypo_obj

    event : Event
        Event for which to get likelihoods

    neg_llh_func : callable
        Function used to compute a likelihood. Must take ``pinfo_gen`` and
        ``event`` as first two arguments, where ``pinfo_gen`` is (...) and
        ``event`` is the argument passed here. Function must return just one
        value (the ``llh``)

    dims : string or iterable thereof
        One of 't', 'x', 'y', 'z', 'azimuth', 'zenith', 'cascade_energy',
        or 'track_energy'.

    scan_values : iterable of floats, or iterable thereof
        Values to set for the dimension being scanned.

    nominal_params : None or HYPO_PARAMS_T namedtuple
        Nominal values for all param values. The value for the params being
        scanned are irrelevant, as this is replaced with each value from
        `scan_values`. Therefore this is optional if _all_ parameters are
        subject to the scan.

    neg_llh_func_kwargs : mapping or None
        Keyword arguments to pass to `get_neg_llh` function

    Returns
    -------
    all_llh : numpy.ndarray (len(scan_values[0]) x len(scan_values[1]) x ...)
        Likelihoods corresponding to each value in product(*scan_values).

    """
    if neg_llh_func_kwargs is None:
        neg_llh_func_kwargs = {}

    all_params = HYPO_PARAMS_T._fields

    # Need list of strings (dim names). If we just have a string, make it the
    # first element of a single-element tuple.
    if isinstance(dims, basestring):
        dims = (dims,)

    # Need iterable-of-iterables-of-floats. If we have just an iterable of
    # floats (e.g. for 1D scan), then make it the first element of a
    # single-element tuple.
    if np.isscalar(next(iter(scan_values))):
        scan_values = (scan_values,)

    scan_sequences = []
    shape = []
    for sv in scan_values:
        if not isinstance(sv, Sequence):
            sv = list(sv)
        scan_sequences.append(sv)
        shape.append(len(sv))

    if nominal_params is None:
        #assert len(dims) == len(all_params)
        nominal_params = HYPO_PARAMS_T(*([np.nan]*len(all_params)))

    # Make nominal into a list so we can mutate its values as we scan
    params = list(nominal_params)

    # Get indices for each param that we'll be changing, in the order they will
    # be specified
    param_indices = []
    for dim in dims:
        param_indices.append(all_params.index(dim))

    all_neg_llh = []
    for param_values in product(*scan_sequences):
        for pidx, pval in izip(param_indices, param_values):
            params[pidx] = pval

        hypo_params = HYPO_PARAMS_T(*params)

        pinfo_gen = hypo_obj.get_pinfo_gen(hypo_params=hypo_params)
        neg_llh = neg_llh_func(pinfo_gen, event, **neg_llh_func_kwargs)
        all_neg_llh.append(neg_llh)

    all_neg_llh = np.array(all_neg_llh, dtype=FTYPE)
    all_neg_llh.reshape(shape)

    return all_neg_llh


def main(events_fpath, tables_dir, geom_file=None, start_index=None,
         stop_index=None):
    """Perform scans and minimization for events.

    Parameters
    ----------
    events_fpath : string
        Path to events HDF5 file

    tables_dir : string
        Path to directory containing the retro tables

    geom_file : string
        File containing detector geometry

    start_index : None or int
        Event index (as ordered in events file) to start on. Specify 0 or
        `None` to start with the first event. I.e., iterate over
        `range(start_index, stop_index)`.

    stop_index : None or int
        Event index (as ordered in events file) to stop before. I.e., iterate
        over `range(start_index, stop_index)`.

    """
    # pylint: disable=no-member

    print('Instantiate a hypoo class...')
    discrete_hypo = DiscreteHypo(
        hypo_kernels=[const_energy_loss_muon, point_cascade]
    )

    # Load events
    print('Loading events...')
    events = Events(events_fpath)
    print('  %d events found' % len(events))

    # Load detector geometry array
    print('Loading detector geometry from "%s"...' % expand(geom_file))
    detector_geometry = np.load(expand(geom_file))

    # Load tables
    print('Loading DOM tables...')
    dom_tables = DOMTimePolarTables(
        tables_dir=tables_dir,
        hash_val=None,
        geom=detector_geometry,
        use_directionality=False
    )
    dom_tables.load_tables()

    tdi_table = None
    #print('Loading TDI table...')
    #tdi_table = TDICartTable(
    #    tables_dir=tables_dir,
    #    use_directionality=False,
    #    #proto_tile_hash='0e28683a74ebea92', # 14^3 tiles, 1 m gridsize, +/- 700 m in x and y, -800 to +600 in z
    #    #proto_tile_hash='8c4770c8371a4025', # single tile, 10 m gridsize +/- 700 m in x and y, -800 to +600 in z
    #    proto_tile_hash='fd29bc306d29bc83', # single tile, QE used; 10 m gridsize +/- 700 m in x and y, -800 to +600 in z
    #    scale=1,
    #)

    neg_llh_func_kwargs = dict(dom_tables=dom_tables, tdi_table=tdi_table)

    # Iterate through events
    for idx, event in enumerate(events[start_index:stop_index]):
        primary_interaction_str = get_primary_interaction_str(event)

        print('Working on event #%i / event UID %d' % (idx, event.uid))
        print('  %s, %.1f GeV' % (primary_interaction_str,
                                  event.neutrino.energy))
        print('    track   :', event.track)
        print('    cascade :', event.cascade)

        truth_params = event_to_hypo_params(event)
        pinfo_gen_truth = discrete_hypo.get_pinfo_gen(truth_params)
        neg_llh_truth = get_neg_llh(pinfo_gen=pinfo_gen_truth, event=event,
                                    **neg_llh_func_kwargs)
        print('llh at truth = %.2f' % neg_llh_truth)

        if MIN_DIMS:
            print('Will minimize following dimension(s): %s' % MIN_DIMS)

            min_results = None

            variable_dims = []
            fixed_dims = []
            for dim in HYPO_PARAMS_T._fields:
                if dim in MIN_DIMS:
                    variable_dims.append(dim)
                else:
                    fixed_dims.append(dim)

            lower_bounds = []
            upper_bounds = []
            for dim in MIN_DIMS:
                if (MIN_USE_RELATIVE_BOUNDS
                        and getattr(REL_BOUNDS, dim) is not None):
                    nom_val = getattr(truth_params, dim)
                    lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                    upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                else:
                    lower, upper = getattr(ABS_BOUNDS, dim)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

            def get_neg_llh_partial(args):
                """Minimizer callable that only includes free params as args"""
                pass

            xopt1, fopt1 = pso(get_neg_llh_partial, lower_bounds, upper_bounds,
                               kwargs=neg_llh_func_kwargs,
                               minstep=1e-5,
                               minfunc=1e-1,
                               debug=True)

        if SCAN_DIM_SETS:
            print('Will scan following sets of dimensions: %s'
                  % str(SCAN_DIM_SETS))
            time_to_scan = 0
            num_likelihoods = 0
            t0 = time.time()
            for dims in SCAN_DIM_SETS:
                print('Scanning dimension(s): %s...' % str(dims))

                if isinstance(dims, basestring):
                    dims = [dims]

                nominal_params = deepcopy(truth_params)
                scan_values = []
                for dim in dims:
                    if (SCAN_USE_RELATIVE_BOUNDS
                            and getattr(REL_BOUNDS, dim) is not None):
                        nom_val = getattr(nominal_params, dim)
                        lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                        upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                    else:
                        lower, upper = getattr(ABS_BOUNDS, dim)
                    scan_values.append(
                        np.linspace(lower, upper, NUM_SCAN_POINTS)
                    )

                ts0 = time.time()
                neg_llh = scan(
                    hypo_obj=discrete_hypo,
                    event=event,
                    neg_llh_func=get_neg_llh,
                    dims=dims,
                    scan_values=scan_values,
                    nominal_params=nominal_params,
                    neg_llh_func_kwargs=neg_llh_func_kwargs
                )
                ts1 = time.time()
                time_to_scan += ts1 - ts0

                num_likelihoods += len(neg_llh)

                datadump = OrderedDict([
                    ('filename', event.filename),
                    ('event', event.event),
                    ('uid', event.uid),
                    ('neutrino_energy', event.neutrino.energy),
                    ('primary_interaction', primary_interaction_str),
                    ('dims', dims),
                    ('scan_values', scan_values),
                    ('neg_llh', neg_llh),
                    ('truth', tuple(getattr(truth_params, d) for d in dims)),
                    ('LLH_USE_AVGPHOT', LLH_USE_AVGPHOT),
                    ('LLH_USE_NOHIT', LLH_USE_NOHIT),
                    ('JITTER_SIGMA', JITTER_SIGMA),
                    ('NUM_JITTER_SAMPLES', NUM_JITTER_SAMPLES),
                    ('CASCADE_E_SCALE', CASCADE_E_SCALE),
                    ('TRACK_E_SCALE', TRACK_E_SCALE),
                    ('EPS_ANGLE', EPS_STAT + EPS_CLSIM_ANGLE_BINNING),
                    ('EPS_LENGTH', EPS_STAT + EPS_CLSIM_LENGTH_BINNING),
                    ('NOISE_CHARGE', NOISE_CHARGE),
                ])
                fname = ('scan_results_event_%d_uid_%d_dims_%s.pkl'
                         % (event.event, event.uid, '_'.join(dims)))
                fpath = join(RESULTS_DIR, fname)
                pickle.dump(
                    datadump,
                    file(fpath, 'wb'),
                    pickle.HIGHEST_PROTOCOL
                )
                print('saved scan to "%s"' % fpath)

            plot_1d_scan(dir=RESULTS_DIR, event=event.event, uid=event.uid)
            print('')
            print('Time to scan ({:d} likelihoods): {} s'
                  .format(num_likelihoods, np.round(time_to_scan, 3)))
            print('Time to scan, dump, and plot: {} s'
                  .format(np.round(time.time() - t0, 3)))
            print('')


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-f', '--file', metavar='H5_FILE', type=str, required=True,
        help='''Events HDF5 file, containing (string, DOM, charge, time per
        event)''',
    )
    parser.add_argument(
        '--start-index', default=None, type=int,
        help='''Event index offset for event to start with (0-indexed)'''
    )
    parser.add_argument(
        '--stop-index', default=None, type=int,
        help='''Event index offset for event to start with (0-indexed)'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/fastio/icecube/retro_tables/full1000',
        help='''Directory containing retro tables''',
    )
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''NPY file containing DOM locations as (string, dom, x, y, z)
        entries'''
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(events_fpath=ARGS.file, geom_file=ARGS.geom_file,
         start_index=ARGS.start_index, stop_index=ARGS.stop_index,
         tables_dir=ARGS.tables_dir)
