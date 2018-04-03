#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, take icecube hdf5 files with events (hits
series) in them as inputs, and calculate lilkelihoods.

At the moment, these likelihoods can be single points, 1d scan, or 2d scan.
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
from collections import OrderedDict
from copy import deepcopy
import cPickle as pickle
from os import makedirs
from os.path import abspath, dirname, isdir, join
import sys
import time

import numpy as np
from pyswarm import pso

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro
from retro.retro_types import HypoParams10D
from retro.utils.misc import expand
from retro.hypo.discrete_cascade_kernels import point_cascade # pylint: disable=unused-import
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo.discrete_muon_kernels import const_energy_loss_muon, table_energy_loss_muon # pylint: disable=unused-import
from retro.events import Events
from retro.likelihood import get_neg_llh
from retro.plot.plot_1d_scan import plot_1d_scan
from retro.scan import scan
from retro.tables.dom_time_polar_tables import DOMTimePolarTables # pylint: disable=unused-import
from retro.tables.tdi_cart_tables import TDICartTable # pylint: disable=unused-import


DFLT_EVENTS_FPATH = (
    '/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600'
    '/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5'
)

# (Almost) completely arbitrary first guesses
NOISE_CHARGE = 0.00000025 * 100
CASCADE_E_SCALE = 1
TRACK_E_SCALE = 1

NUM_SCAN_POINTS = 100
HypoClass = DiscreteHypo
HYPOCLASS_KWARGS = dict(time_increment=1)
LLH_USE_NOHIT = False

RESULTS_DIR = (
    'results_nohit%d_cesc%d_tesc%d_noise%.1e'
    % (LLH_USE_NOHIT,
       CASCADE_E_SCALE,
       TRACK_E_SCALE,
       NOISE_CHARGE)
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
    'time',
    'x',
    'y',
    'z',
    'track_zenith',
    'track_azimuth',
    'track_energy',
    'cascade_energy',
    #('time', 'x'),
    #('time', 'y'),
    #('time', 'z'),
    #('x', 'y'),
    #('x', 'z'),
    #('track_zenith', 'track_azimuth'),
    #('track_zenith', 'z'),
)
"""Which dimensions to scan. Tuples specify 2+ dimension scans"""


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

    print('Instantiate a hypo class...')
    discrete_hypo = DiscreteHypo(
        hypo_kernels=[const_energy_loss_muon, point_cascade]
    )

    # Load events
    print('Loading events...')
    events = Events(events_fpath)
    print('  %d events found' % len(events))

    # Load detector geometry array
    print('Loading detector geometry from "%s"...' % retro.expand(geom_file))
    detector_geometry = np.load(retro.expand(geom_file))

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

    neg_llh_func_kwargs = dict(
        dom_tables=dom_tables,
        tdi_table=tdi_table,
        noise_charge=NOISE_CHARGE
    )

    # Iterate through events
    for idx, event in enumerate(events[start_index:stop_index]):
        primary_interaction_str = retro.get_primary_interaction_str(event)

        print('Working on event #%i / event UID %d' % (idx, event.uid))
        print('  %s, %.1f GeV' % (primary_interaction_str,
                                  event.neutrino.energy))
        print('    track   :', event.track)
        print('    cascade :', event.cascade)

        truth_params = retro.event_to_hypo_params(event)
        pinfo_gen_truth = discrete_hypo.get_pinfo_gen(truth_params)
        neg_llh_truth = get_neg_llh(pinfo_gen=pinfo_gen_truth, event=event,
                                    **neg_llh_func_kwargs)
        print('llh at truth = %.2f' % neg_llh_truth)

        if MIN_DIMS:
            print('Will minimize following dimension(s): %s' % MIN_DIMS)

            min_results = None

            variable_dims = []
            fixed_dims = []
            for dim in retro.HYPO_PARAMS_T._fields:
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
                    ('LLH_USE_NOHIT', LLH_USE_NOHIT),
                    ('CASCADE_E_SCALE', CASCADE_E_SCALE),
                    ('TRACK_E_SCALE', TRACK_E_SCALE),
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
        default=retro.DETECTOR_GEOM_FILE,
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
