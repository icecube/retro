#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Plot marginal distributions saved to JSON files by `summarize_clsim_table.py`
for one or more tables.
"""


from __future__ import absolute_import, division, print_function

__all__ = ['plot_clsim_table_summary', 'parse_args', 'main']

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
from collections import Mapping, OrderedDict
from copy import deepcopy
from glob import glob
from itertools import product
from os.path import abspath, dirname, join
import sys

import matplotlib as mpl
mpl.use('agg', warn=False)
import matplotlib.pyplot as plt
import numpy as np

from pisa.utils.jsons import from_json

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import COLOR_CYCLE_ORTHOG
from retro import expand, mkdir


def plot_clsim_table_summary(summaries, save_formats=None, outdir=None):
    """Plot the table summary produced by `summarize_clsim_table`.

    Parameters
    ----------
    summaries : string, summary, or iterable thereof
        If string(s) are provided, each is glob-expanded. See
        :method:`glob.glob` for valid syntax.

    save_formats : None, string, or iterable of strings in {'pdf', 'png'}
        If no formats are provided, the plot will not be saved.

    outdir : None or string
        If `save_formats` is specified and `outdir` is None, the plots are
        saved to the present working directory.

    Returns
    -------
    figs : list of two :class:`matplotlib.figure.Figure`

    axes : list of to lists of :class:`matplotlib.axes.Axes`

    summaries : list of :class:`collections.OrderedDict`
        List of all summaries loaded

    """
    orig_summaries = deepcopy(summaries)

    if isinstance(summaries, (basestring, Mapping)):
        summaries = [summaries]

    tmp_summaries = []
    for summary in summaries:
        if isinstance(summary, Mapping):
            tmp_summaries.append(summary)
        elif isinstance(summary, basestring):
            tmp_summaries.extend(glob(expand(summary)))
    summaries = tmp_summaries

    for summary_n, summary in enumerate(summaries):
        if isinstance(summary, basestring):
            summary = from_json(summary)
            summaries[summary_n] = summary

    if save_formats is None:
        save_formats = []
    elif isinstance(save_formats, basestring):
        save_formats = [save_formats]

    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)

    n_summaries = len(summaries)

    if n_summaries == 0:
        raise ValueError(
            'No summaries found based on argument `summaries`={}'
            .format(orig_summaries)
        )

    for n, save_format in enumerate(save_formats):
        save_format = save_format.strip().lower()
        assert save_format in ('pdf', 'png'), save_format
        save_formats[n] = save_format

    all_items = OrderedDict()
    for summary in summaries:
        for key, value in summary.items():
            if key == 'dimensions':
                continue
            if not all_items.has_key(key):
                all_items[key] = []
            all_items[key].append(value)

    same_items = OrderedDict()
    different_items = OrderedDict()
    for key, values in all_items.items():
        all_same = True
        ref_value = values[0]
        for value in values[1:]:
            if np.any(value != ref_value):
                all_same = False

        if all_same:
            same_items[key] = values[0]
        else:
            different_items[key] = values

    if n_summaries > 1:
        if same_items:
            print('Same for all:\n{}'.format(same_items.keys()))
        if different_items:
            print('Different for some or all:\n{}'
                  .format(different_items.keys()))

    def formatter(mapping, key_only=False, fname=False):
        """Formatter for labels to go in plots and and filenames.

        Parameters
        ----------
        mapping : Mapping
        key_only : bool
        fname : bool

        """
        order = [
            'hash_val', 'string', 'depth_idx', 'seed', 'table_shape',
            'n_events', 'n_photons', 'ice_model', 'tilt'
        ]

        label_strs = []
        for key in order:
            if key not in mapping:
                continue

            if key_only:
                label_strs.append(key)
                continue

            if fname:
                sep = '_'
            else:
                sep = '='

            value = mapping[key]

            if key == 'n_photons':
                label_strs.append('n_photons{}{:.3e}'.format(sep, value))
            elif key in ['depth_idx', 'seed', 'string', 'n_events', 'ice_model', 'tilt']:
                label_strs.append('{}{}{}'.format(key, sep, value))
            elif key == 'phase_refractive_index':
                label_strs.append('n{}{:.3f}'.format(sep, value))
            elif key == 'table_shape':
                if fname:
                    shape = '_'.join(str(x) for x in value)
                    fmt = '{}'
                else:
                    shape = ', '.join(str(x) for x in value)
                    fmt = '({})'
                label_strs.append(('shape{}%s' % fmt).format(sep, shape))
            elif key == 'hash_val':
                label_strs.append('hash{}{}'.format(sep, value))
        if fname:
            return '__'.join(label_strs)
        return ', '.join(label_strs)

    same_label = formatter(same_items)

    strings = sorted(set(all_items['string']))
    depths = sorted(set(all_items['depth_idx']))
    seeds = sorted(set(all_items['seed']))

    plot_kinds = ('mean', 'max')
    dim_names = summaries[0]['dimensions'].keys()
    n_dims = len(dim_names)

    fig_x = 10 # inches
    fig_header_y = 0.25 # inches
    fig_one_axis_y = 5 # inches
    fig_all_axes_y = n_dims * fig_one_axis_y
    fig_y = fig_header_y + fig_all_axes_y # inches

    fig1, axes1 = plt.subplots(nrows=n_dims, ncols=1, squeeze=False,
                               figsize=(fig_x, fig_y))
    fig2, axes2 = plt.subplots(nrows=n_dims, ncols=1, squeeze=False,
                               figsize=(fig_x, fig_y))
    figs = [fig1, fig2]
    axes1 = list(axes1.flat)
    axes2 = list(axes2.flat)
    axes = [axes1, axes2]

    for ax_set in axes:
        for ax in ax_set:
            ax.set_prop_cycle('color', COLOR_CYCLE_ORTHOG)

    n_lines = 0
    xlims = [[np.inf, -np.inf]] * n_dims

    labels_assigned = set()
    for string, depth_idx, seed in product(strings, depths, seeds):
        for summary_n, summary in enumerate(summaries):
            if (summary['string'] != string
                    or summary['depth_idx'] != depth_idx
                    or summary['seed'] != seed):
                continue

            different_label = formatter({k: v[summary_n] for k, v in different_items.items()})

            if different_label:
                label = different_label
                if label in labels_assigned:
                    label = None
                else:
                    labels_assigned.add(label)
            else:
                label = None

            for dim_num, dim_name in enumerate(dim_names):
                dim_info = summary['dimensions'][dim_name]
                dim_axes = (axes1[dim_num], axes2[dim_num])
                bin_edges = summary[dim_name + '_bin_edges']
                if dim_name == 'deltaphidir':
                    bin_edges /= np.pi
                xlims[dim_num] = [
                    min(xlims[dim_num][0], np.min(bin_edges)),
                    max(xlims[dim_num][1], np.max(bin_edges))
                ]
                for ax, plot_kind in zip(dim_axes, plot_kinds):
                    vals = dim_info[plot_kind]
                    ax.step(bin_edges, [vals[0]] + list(vals),
                            linewidth=1, clip_on=False,
                            label=label)
                    n_lines += 1

    dim_labels = dict(
        r=r'$r$',
        costheta=r'$\cos\theta$',
        t=r'$t$',
        costhetadir=r'$\cos\theta_{\rm dir}$',
        deltaphidir=r'$\Delta\phi_{\rm dir}$'
    )
    units = dict(r='m', t='ns', deltaphidir=r'rad/$\pi$')

    logx_dims = []
    logy_dims = ['r', 't', 'deltaphidir']

    flabel = ''
    same_flabel = formatter(same_items, fname=True)
    different_flabel = formatter(different_items, key_only=True, fname=True)
    if same_flabel:
        flabel += '__same__' + same_flabel
    if different_flabel:
        flabel += '__differ__' + different_flabel

    for kind_idx, (plot_kind, fig) in enumerate(zip(plot_kinds, figs)):
        for dim_num, (dim_name, ax) in enumerate(zip(dim_names, axes[kind_idx])):
            #if dim_num == 0 and different_items:
            if different_items:
                ax.legend(loc='best', frameon=False,
                          prop=dict(size=7, family='monospace'))

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()

            ax.set_xlim(xlims[dim_num])

            xlabel = dim_labels[dim_name]
            if dim_name in units:
                xlabel += ' ({})'.format(units[dim_name])
            ax.set_xlabel(xlabel)
            if dim_name in logx_dims:
                ax.set_xscale('log')
            if dim_name in logy_dims:
                ax.set_yscale('log')

        fig.tight_layout(rect=(0, 0, 1, fig_all_axes_y/fig_y))
        suptitle = (
            'Marginalized distributions (taking {} over all other axes)'
            .format(plot_kind)
        )
        if same_label:
            suptitle += '\n' + same_label
        fig.suptitle(suptitle, y=(fig_all_axes_y + fig_header_y*0.8) / fig_y,
                     fontsize=9)

        for save_format in save_formats:
            outfpath = ('clsim_table_summaries{}__{}.{}'
                        .format(flabel, plot_kind, save_format))
            if outdir:
                outfpath = join(outdir, outfpath)
            fig.savefig(outfpath, dpi=300)
            print('Saved image to "{}"'.format(outfpath))

    return [fig1, fig2], [axes1, axes2], summaries


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--summaries', nargs='+', required=True,
        help='''Path(s) to summary JSON files to plot. Note that literal
        strings are glob-expanded.'''
    )
    parser.add_argument(
        '--save-formats', choices=('pdf', 'png'), nargs='+', default='pdf',
        help='''Save plots to chosen format(s). Choices are "pdf" and "png".'''
    )
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory to which to save the plot(s). Defaults to same
        directory as the present working directory.'''
    )
    return parser.parse_args()


def main():
    """Main function for calling plot_clsim_table_summary as a script"""
    args = parse_args()
    kwargs = vars(args)
    plot_clsim_table_summary(**kwargs)


if __name__ == '__main__':
    main()
