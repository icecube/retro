#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Plot likelihood scan results
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    FNAME_TEMPLATE
    plot_1d_scan
    parse_args
'''.split()

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
import cPickle as pickle
from os.path import abspath, dirname, join
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import HypoParams8D
from retro.utils.misc import expand, get_primary_interaction_tex


FNAME_TEMPLATE = 'scan_results_event_{event}_uid_{uid}_dims_{param}.pkl'


def plot_1d_scan(dir, event, uid): # pylint: disable=redefined-builtin
    """main"""
    #scan_files = glob(expand(dirpath) + '/*_uid_%d_*' % uid)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 8), dpi=72)
    axiter = iter(axes.flatten())

    for pnum, param in enumerate(HypoParams8D._fields):
        fname = FNAME_TEMPLATE.format(event=event, uid=uid, param=param)
        fpath = expand(join(dir, fname))
        scan = pickle.load(file(fpath, 'rb'))
        scan_values = scan['scan_values'][0]
        truth = scan['truth'][0]
        llh = -scan['neg_llh']
        err_at_max_llh = scan_values[llh == llh.max()][0] - truth

        if param == 'time':
            units = 'ns'
        elif param in ['x', 'y', 'z']:
            units = 'm'
        elif param in ['track_zenith', 'track_azimuth']:
            units = 'deg'
            scan_values *= 180 / np.pi
            err_at_max_llh *= 180/np.pi
            err_at_max_llh = ((err_at_max_llh + 180) % 360) - 180
            truth *= 180/np.pi
        elif param in ['track_energy', 'cascade_energ']:
            units = 'GeV'

        ax = next(axiter)
        ax.plot(
            [0]*2, [llh.min(), llh.max()],
            color='C1',
            label='truth'
        )
        ax.plot(
            scan_values - truth, llh,
            color='C0',
            label='LLH scan'
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axis('tight')

        ax.set_xlabel(r'%s$_{\rm reco} -$%s$_{\rm true}$ (%s)'
                      % (param, param, units))
        ax.set_title(r'Error at LLH$_{\rm max}$: %.2f %s'
                     % (err_at_max_llh, units))

        if pnum == 0:
            ax.legend(loc='best')

    if scan['LLH_USE_AVGPHOT']:
        llhname = r'LLH from counts including avg. photon'
        eps = (r', $\epsilon_{\rm ang}$=%.1f, $\epsilon_{\rm len}$=%.1f'
               % (scan['EPS_ANGLE'], scan['EPS_LENGTH']))
    else:
        llhname = 'LLH from simple counts'
        eps = ''
    if scan['NUM_JITTER_SAMPLES'] > 1:
        jitter_sigmas = r' $\sigma_{\rm jitter}$=%d,' % scan['JITTER_SIGMA']
    else:
        jitter_sigmas = ''

    prim_int_tex = get_primary_interaction_tex(scan['primary_interaction'])

    fig.suptitle(
        r'Event %s: %.1f GeV $%s$; %s%s'
        r'$q_{\rm noise}$=%.1e,'
        '%s'
        r' $N_{\rm samp,jitter}=$%d,'
        r' escale$_{\rm cscd}$=%d,'
        r' escale$_{\rm trck}$=%d'
        '%s'
        % (scan['uid'], scan['neutrino_energy'], prim_int_tex, llhname, '\n',
           scan['NOISE_CHARGE'],
           jitter_sigmas,
           scan['NUM_JITTER_SAMPLES'],
           scan['CASCADE_E_SCALE'],
           scan['TRACK_E_SCALE'],
           eps),
        fontsize=14
    )

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fbasename = 'scan_results_event_%d_uid_%d_1d' % (event, uid)
    fbasepath = join(dir, fbasename)
    fpath = fbasepath + '.png'
    fig.savefig(fpath, dpi=120)
    print('saved plot to "%s"' % fpath)
    #fpath = fbasepath + '.pdf'
    #fig.savefig(fpath)
    #print('saved plot to "%s"' % fpath)
    #plt.draw()
    #plt.show()


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-d', '--dir', metavar='DIR', type=str, required=True,
        help='''Directory containing retro tables''',
    )
    parser.add_argument(
        '-e', '--event', type=int, required=True,
        help='''Event ID from original I3 / HDF5 file'''
    )
    parser.add_argument(
        '-u', '--uid', type=int, required=True,
        help='''Unique event ID'''
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    plot_1d_scan(**vars(parse_args()))
