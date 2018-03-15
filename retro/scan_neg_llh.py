#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Instantiate Retro tables and scan the negative log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['scan_neg_llh', 'main', 'parse_args']

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
from itertools import product
from os.path import abspath, dirname
import pickle
import sys

import numpy as np

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T
from retro.utils.misc import expand
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


def scan_neg_llh(
        # Scan params
        scan_values,

        # Hit params
        hits, time_window,

        # Hypo computation params
        hypo_kernels, kernel_kwargs,

        # DOM tables' params
        dom_tables_fname_proto, angular_acceptance_fract, step_length, mmap,
        dom_table_kind, gcd, norm_version, use_directionality, num_phi_samples,
        ckv_sigma_deg,

        # TDI table params
        tdi_table
    ):
    """Instantiate Retro tables and perform a likelihood scan.

    """
    if tdi_table is not None:
        raise NotImplementedError('TDI table not handled yet')

    compute_t_indep_exp = tdi_table is None
    if isinstance(gcd, basestring):
        gcd = extract_gcd(gcd)

    # Instantiate single-DOM tables
    dom_tables = Retro5DTables(
        table_kind=dom_table_kind,
        geom=gcd['geo'],
        rde=gcd['rde'],
        noise_rate_hz=gcd['noise'],
        compute_t_indep_exp=compute_t_indep_exp,
        use_directionality=use_directionality,
        norm_version=norm_version,
        num_phi_samples=num_phi_samples,
        ckv_sigma_deg=ckv_sigma_deg
    )

    # Load single-DOM tables
    common_kw = dict(
        step_length=step_length,
        angular_acceptance_fract=angular_acceptance_fract,
        mmap=mmap
    )

    if '{subdet' in dom_tables_fname_proto:
        for subdet, dom in list(product(['ic', 'dc'], range(1, 60+1))):
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

    hypo_handler = DiscreteHypo(
        hypo_kernels=hypo_kernels,
        kernel_kwargs=kernel_kwargs
    )

    # Keyword args for the `metric` callable
    metric_kw = dict(
        hits=hits, time_window=time_window, hypo_handler=hypo_handler,
        dom_tables=dom_tables, tdi_table=tdi_table
    )

    # Perform the actual scan
    metric_vals = scan(
        scan_values=scan_values,
        metric=get_neg_llh,
        metric_kw=metric_kw
    )

    return metric_vals


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

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
        '--time-window', type=float, required=True,
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
        '--angular-acceptance-fract', type=float, default=0.338019664877,
        help='''Comes from the angular acceptance model chosen.'''
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

    return parser.parse_args()


def main():
    """Script "main" function"""
    args = parse_args()
    kwargs = vars(args)

    scan_values = []
    for dim in HYPO_PARAMS_T._fields:
        val_str = ''.join(kwargs.pop(dim))
        val_str.replace('pi', format(np.pi, '.17e'))
        val_str.replace('e', format(np.exp(1), '.17e'))
        scan_values.append(hrlist2list(val_str))

    hits_file = expand(kwargs['hits'])
    with open(hits_file, 'rb') as f:
        kwargs['hits'] = pickle.load(f)

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

    track_kernel = kwargs.popp('track_kernel')
    if track_kernel == 'const_e_loss':
        hypo_kernels.append(const_energy_loss_muon)
    else:
        hypo_kernels.append(table_energy_loss_muon)
    kernel_kwargs.append(dict(dt=kwargs.pop('track_time_step')))

    force_no_mmap = kwargs.pop('force_no_mmap')
    if force_no_mmap:
        kwargs['mmap'] = False
    else:
        kwargs['mmap'] = 'uncompr' in kwargs['dom_table_kind']

    kwargs['use_directionality'] = not kwargs.pop('no_dir')

    return scan_neg_llh(**kwargs)


if __name__ == '__main__':
    metric_vals = main() # pylint: disable=invalid-name
