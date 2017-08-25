#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Plot likelihood scan results
"""

from __future__ import absolute_import, division

from argparse import ArgumentParser
import cPickle as pickle
import os
from os.path import abspath, dirname, join

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import HypoParams8D
from retro import expand, get_primary_interaction_tex


__all__ = ['FNAME_TEMPLATE', 'parse_args', 'plot_1d_scan']


FNAME_TEMPLATE = 'scan_results_event_{event}_uid_{uid}_dims_{param}.pkl'


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


def plot_1d_scan(dir, event, uid):
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

        if param == 't':
            units = 'ns'
        elif param in ['x', 'y', 'z']:
            units = 'm'
        elif param in ['track_zenith', 'track_azimuth']:
            units = 'deg'
            scan_values *= 180 / np.pi
            err_at_max_llh *= 180/np.pi
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
    fig.savefig(join(dir, fbasename + '.png'), dpi=300)
    fig.savefig(join(dir, fbasename + '.pdf'))
    #plt.draw()
    #plt.show()


if __name__ == '__main__':
    plot_1d_scan(**vars(parse_args()))
