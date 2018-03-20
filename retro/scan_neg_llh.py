#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Instantiate Retro tables and scan the negative log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['scan_neg_llh', 'parse_args']

__author__ = 'J.L. Lanfranchi'
__license__ = '''Copyright 2017 Justin L. Lanfranchi

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
from itertools import product
from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T
from retro.utils.misc import expand, mkdir
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo.discrete_cascade_kernels import (
    point_cascade
)
from retro.hypo.discrete_muon_kernels import (
    const_energy_loss_muon, table_energy_loss_muon
)
from retro.i3info.extract_gcd import extract_gcd
from retro.likelihood import get_neg_llh
from retro.scan import scan
from retro.tables.retro_5d_tables import (
    NORM_VERSIONS, TABLE_KINDS, Retro5DTables
)


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )
    parser.add_argument(
        '--n-events', type=int, default=None
    )
    parser.add_argument(
        '--start-event-idx', type=int, default=0
    )

    for dim in HYPO_PARAMS_T._fields:
        parser.add_argument(
            '--{}'.format(dim.replace('_', '-')), nargs='+', required=True,
            help='''Hypothses will take this(these) value(s) for dimension
            {dim_hr}. Specify a single value to not scan over this dimension;
            specify a human-readable string of values, e.g. '0, 0.5, 1-10:0.2'
            scans 0, 0.5, and from 1 to 10 (inclusive of both endpoints) with
            stepsize of 0.2.'''.format(dim_hr=dim.replace('_', ' '))
        )

    parser.add_argument(
        '--hits', required=True,
    )
    parser.add_argument(
        '--hits-are-photons', action='store_true',
    )
    parser.add_argument(
        '--time-window', type=float, required=True,
    )
    parser.add_argument(
        '--angsens-model',
        choices='nominal  h1-100cm  h2-50cm  h3-30cm'.split()
    )

    parser.add_argument(
        '--cascade-kernel', choices=['point', 'one_dim'], required=True,
    )
    parser.add_argument(
        '--cascade-samples', type=int, default=1,
    )
    parser.add_argument(
        '--track-kernel', required=True,
        choices=['const_e_loss', 'nonconst_e_loss'],
    )
    parser.add_argument(
        '--track-time-step', type=float, required=True,
    )

    parser.add_argument(
        '--dom-tables-fname-proto', required=True,
        help='''Must have one of the brace-enclosed fields "{string}" or
        "{subdet}", and must have one of "{dom}" or "{depth_idx}". E.g.:
        "my_tables_{subdet}_{depth_idx}"'''
    )
    parser.add_argument(
        '--step-length', type=float, default=1.0,
        help='''Step length used in the CLSim table generator.'''
    )
    parser.add_argument(
        '--force-no-mmap', action='store_true',
        help='''Specify to NOT memory map the tables. If not specified, a
        sensible default is chosen for the type of tables being used.'''
    )
    parser.add_argument(
        '--dom-table-kind', choices=TABLE_KINDS, required=True,
        help='''Kind of single-DOM table to use.'''
    )
    parser.add_argument(
        '--gcd', required=True,
        help='''IceCube GCD file; can either specify an i3 file, or the
        extracted pkl file used in Retro.'''
    )
    parser.add_argument(
        '--norm-version', choices=NORM_VERSIONS, required=True,
        help='''Norm version.'''
    )
    parser.add_argument(
        '--no-dir', action='store_true',
        help='''Do NOT use source photon directionality'''
    )
    parser.add_argument(
        '--num-phi-samples', type=int, default=None,
    )
    parser.add_argument(
        '--ckv-sigma-deg', type=float, default=None,
    )
    parser.add_argument(
        '--tdi-table', default=None
    )
    parser.add_argument(
        '--template-library', default=None
    )

    return parser.parse_args()


def scan_neg_llh():
    """Script "main" function"""
    t00 = time.time()

    args = parse_args()
    kwargs = vars(args)
    orig_kwargs = deepcopy(kwargs)

    scan_values = []
    for dim in HYPO_PARAMS_T._fields:
        val_str = ''.join(kwargs.pop(dim))
        val_str.replace('pi', format(np.pi, '.17e'))
        val_str.replace('e', format(np.exp(1), '.17e'))
        scan_values.append(hrlist2list(val_str))

    # -- Instantiate hypo class with kernels -- #
    print('Instantiating hypo object & kernels')
    t0 = time.time()

    hypo_kernels = []
    kernel_kwargs = []

    cascade_kernel = kwargs.pop('cascade_kernel')
    cascade_samples = kwargs.pop('cascade_samples')
    if cascade_kernel == 'point':
        hypo_kernels.append(point_cascade)
        kernel_kwargs.append(dict())
    else:
        raise NotImplementedError('{} cascade not implemented yet.'
                                  .format(cascade_kernel))
        #hypo_kernels.append(one_dim_cascade)
        #kernel_kwargs.append(dict(num_samples=cascade_samples))

    track_kernel = kwargs.pop('track_kernel')
    if track_kernel == 'const_e_loss':
        hypo_kernels.append(const_energy_loss_muon)
    else:
        hypo_kernels.append(table_energy_loss_muon)
    kernel_kwargs.append(dict(dt=kwargs.pop('track_time_step')))

    hypo_handler = DiscreteHypo(
        hypo_kernels=hypo_kernels,
        kernel_kwargs=kernel_kwargs
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # -- Instantiate tables class and load tables -- #
    print('Instantiating tables object')
    t0 = time.time()

    dom_table_kind = kwargs.pop('dom_table_kind')
    angsens_model = kwargs.pop('angsens_model')
    norm_version = kwargs.pop('norm_version')
    num_phi_samples = kwargs.pop('num_phi_samples')
    ckv_sigma_deg = kwargs.pop('ckv_sigma_deg')
    dom_tables_fname_proto = kwargs.pop('dom_tables_fname_proto')
    time_window = kwargs.pop('time_window')
    step_length = kwargs.pop('step_length')
    force_no_mmap = kwargs.pop('force_no_mmap')
    if force_no_mmap:
        mmap = False
    else:
        mmap = 'uncompr' in dom_table_kind

    use_directionality = not kwargs.pop('no_dir')

    tdi_table = kwargs.pop('tdi_table')
    if tdi_table is not None:
        raise NotImplementedError('TDI table not handled yet')


    if dom_table_kind in ['raw_templ_compr', 'ckv_templ_compr']:
        template_library = np.load(args.template_library)
    else:
        template_library = None

    compute_t_indep_exp = tdi_table is None

    gcd = extract_gcd(kwargs.pop('gcd'))

    # Instantiate single-DOM tables
    dom_tables = Retro5DTables(
        table_kind=dom_table_kind,
        geom=gcd['geo'],
        rde=gcd['rde'],
        noise_rate_hz=gcd['noise'],
        angsens_model=angsens_model,
        compute_t_indep_exp=compute_t_indep_exp,
        use_directionality=use_directionality,
        norm_version=norm_version,
        num_phi_samples=num_phi_samples,
        ckv_sigma_deg=ckv_sigma_deg,
        template_library=template_library,
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # Load single-DOM tables
    print('Loading single-DOM tables')
    t0 = time.time()

    common_kw = dict(step_length=step_length, mmap=mmap)

    if '{subdet' in dom_tables_fname_proto:
        for subdet, dom in list(product(['ic', 'dc'], range(1, 60+1))):
            if subdet == 'ic' and dom < 25:
                continue
            if subdet == 'dc' and dom < 11:
                continue

            fpath = dom_tables_fname_proto.format(
                subdet=subdet, dom=dom, depth_idx=dom-1
            )
            dom_tables.load_table(
                fpath=fpath,
                string=subdet,
                dom=dom,
                **common_kw
            )
    elif '{string' in dom_tables_fname_proto:
        for string, dom in product(range(1, 86+1), range(1, 60+1)):
            fpath = dom_tables_fname_proto.format(
                string=string, string_idx=string - 1,
                dom=dom, depth_idx=dom - 1
            )
            dom_tables.load_table(
                fpath=fpath,
                string=string,
                dom=dom,
                **common_kw
            )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # -- Load hits -- #
    print('Loading hits')
    t0 = time.time()

    hits_are_photons = kwargs.pop('hits_are_photons')

    hits_file = expand(kwargs['hits'])
    with open(hits_file, 'rb') as f:
        hits = pickle.load(f)

    outdir = kwargs.pop('outdir')
    mkdir(outdir)

    start_event_idx = kwargs.pop('start_event_idx')
    n_events = kwargs.pop('n_events')
    stop_event_idx = None if n_events is None else start_event_idx + n_events
    events_slice = slice(start_event_idx, stop_event_idx)

    # Keyword args for the `metric` callable (get_neg_llh)
    metric_kw = dict(
        time_window=time_window,
        hypo_handler=hypo_handler,
        dom_tables=dom_tables,
        tdi_table=tdi_table
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Scanning paramters')
    t0 = time.time()

    metrics = []
    for event_ofst, event_hits in enumerate(hits[events_slice]):
        event_idx = start_event_idx + event_ofst
        t1 = time.time()
        if hits_are_photons:
            # For photons, we assign a "charge" from their weight, which comes
            # from angsens model.
            event_photons = event_hits
            event_hits = OrderedDict()
            for str_dom, pinfo in event_photons.items():
                t = pinfo[0, :]
                coszen = pinfo[4, :]
                weight = np.float32(dom_tables.angsens_poly(coszen))
                event_hits[str_dom] = np.concatenate(
                    (t[np.newaxis, :], weight[np.newaxis, :]),
                    axis=0
                )

        metric_kw['hits'] = event_hits

        # Perform the actual scan
        metric_vals = scan(
            scan_values=scan_values,
            metric=get_neg_llh,
            metric_kw=metric_kw
        )

        metrics.append(metric_vals)

        dt = time.time() - t1
        n_points = metric_vals.size
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt/n_points*1e3))

    kwargs.pop('hits')
    info = OrderedDict([
        ('hypo_params', HYPO_PARAMS_T._fields),
        ('scan_values', scan_values),
        ('kwargs', OrderedDict([(k, orig_kwargs[k]) for k in sorted(orig_kwargs.keys())])),
        ('metric_name', 'neg_llh'),
        ('metrics', metrics)
    ])

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    t0 = time.time()
    outfpath = join(outdir, 'scan.pkl')
    print('Saving results to pickle file at "{}"'.format(outfpath))
    pickle.dump(info, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Total script run time is {:.3f} s'.format(time.time() - t00))

    return metrics, orig_kwargs


if __name__ == '__main__':
    metric_vals, orig_kwargs = scan_neg_llh() # pylint: disable=invalid-name
