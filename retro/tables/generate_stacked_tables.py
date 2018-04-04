#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Stack template-compressed Cherenkov tables and time-independent Cherenkov
tables to make two monolithic table files, making acces to all tables fast (and
possible from Numba).
"""

from __future__ import absolute_import, division, print_function

__all__ = ['generate_stacked_tables']

__author__ = 'J.L. Lanfranchi, P. Eller'
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
from os.path import abspath, dirname, join
import cPickle as pickle
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import init_obj
from retro.utils.misc import expand, mkdir


def generate_stacked_tables(outdir, dom_tables_kw):
    """Stack a set of tables into a single numpy array for use of all tables in
    numba.

    Currently, only ckv_templ_compr tables are supported.

    Parameters
    ----------
    outdir : string
        Path ot directory into which the three resulting files will be stored.

    dom_tables_kw : mapping
        As returned by retro.init_obj.parse_args

    """
    if dom_tables_kw['dom_tables_kind'] != 'ckv_templ_compr':
        raise NotImplementedError(
            '"{}" tables not supported; only "ckv_templ_compr"'
            .format(dom_tables_kw['dom_tables_kind'])
        )

    # Use the conveneience function to load the single-DOM tables into a
    # retro_5d_tables.Retro5DTables object, and then we can use the loaded
    # tables from there.
    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)

    assert np.all(dom_tables.sd_idx_table_indexer >= 0)

    table_meta = OrderedDict()
    table_meta['table_kind'] = dom_tables.table_kind
    table_meta['sd_idx_table_indexer'] = dom_tables.sd_idx_table_indexer
    table_meta.update(dom_tables.table_meta)
    table_meta['n_photons'] = 1.0
    table_meta['n_photons_per_table'] = np.array(dom_tables.n_photons_per_table)

    outdir = expand(outdir)
    mkdir(outdir)

    fpath = join(outdir, 'stacked_{}_meta.pkl'.format(dom_tables.table_name))
    sys.stdout.write('Writing metadata to "{}" ...'.format(fpath))
    sys.stdout.flush()
    pickle.dump(
        table_meta,
        file(fpath, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    sys.stdout.write(' done.\n')
    sys.stdout.flush()

    if dom_tables.compute_t_indep_exp:
        # Renormalize to 1 photon
        stacked_t_indep_tables = np.stack(
            [tbl/n for tbl, n in zip(dom_tables.t_indep_tables, dom_tables.n_photons_per_table)]
        )
        fpath = join(
            outdir,
            'stacked_{}.npy'.format(dom_tables.t_indep_table_name)
        )
        sys.stdout.write('Writing stacked t_indep tables to "{}" ...'
                         .format(fpath))
        sys.stdout.flush()
        np.save(fpath, stacked_t_indep_tables)
        sys.stdout.write(' done.\n')
        sys.stdout.flush()

    # Renormalize to 1 photon
    for template_map, n_photons in zip(dom_tables.tables, dom_tables.n_photons_per_table):
        template_map['weight'] /= n_photons

    stacked_tables = np.stack(dom_tables.tables)
    fpath = join(outdir, 'stacked_{}.npy'.format(dom_tables.table_name))
    sys.stdout.write('Writing stacked tables to "{}" ...'.format(fpath))
    sys.stdout.flush()
    np.save(fpath, stacked_tables)
    sys.stdout.write(' done.\n')
    sys.stdout.flush()


def main(description=__doc__):
    """Script main function"""
    parser = ArgumentParser(description=description)
    parser.add_argument('--outdir', required=True)

    split_kwargs = init_obj.parse_args(dom_tables=True, parser=parser)
    generate_stacked_tables(dom_tables_kw=split_kwargs['dom_tables_kw'],
                            **split_kwargs['other_kw'])


if __name__ == '__main__':
    main()
