#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Plot marginal distributions saved to JSON files by `summarize_clsim_table.py`
for one or more tables.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    formatter
    plot_clsim_table_summary
    parse_args
    main
'''.split()

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
from pisa.utils.format import format_num

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand, mkdir
from retro.utils.plot import COLOR_CYCLE_ORTHOG


def formatter(mapping, key_only=False, fname=False):
    """Formatter for labels to go in plots and and filenames.

    Parameters
    ----------
    mapping : Mapping
    key_only : bool
    fname : bool

    """
    order = [
        'hash_val',
        'string',
        'depth_idx',
        'seed',
        'table_shape',
        'n_events',
        'ice_model',
        'tilt',
        'n_photons',
        'norm',
        'underflow',
        'overflow'
    ] # yapf: disable

    line_sep = '\n'

    if fname:
        for key in ('n_photons', 'norm', 'underflow', 'overflow'):
            order.remove(key)

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
            label_strs.append(
                '{}{}{}'.format(
                    key, sep, format_num(value, sigfigs=3, sci_thresh=(4, -3))
                )
            )

        elif key in ('depth_idx', 'seed', 'string', 'n_events', 'ice_model',
                     'tilt'):
            label_strs.append('{}{}{}'.format(key, sep, value))

        elif key == 'group_refractive_index':
            label_strs.append('n_grp{}{:.3f}'.format(sep, value))

        elif key == 'phase_refractive_index':
            label_strs.append('n_phs{}{:.3f}'.format(sep, value))

        elif key in ('table_shape', 'underflow', 'overflow'):
            if key == 'table_shape':
                name = 'shape'
            elif key == 'underflow':
                name = 'uflow'
            elif key == 'overflow':
                name = 'oflow'

            str_values = []
            for v in value:
                if float(v) == int(v):
                    str_values.append(format(int(np.round(v)), 'd'))
                else:
                    str_values.append(format_num(v, sigfigs=2, sci_thresh=(4, -3)))

            if fname:
                val_str = '_'.join(str_values)
                fmt = '{}'
            else:
                val_str = ', '.join(str_values)
                fmt = '({})'

            label_strs.append(('{}{}%s' % fmt).format(name, sep, val_str))

        elif key == 'hash_val':
            label_strs.append('hash{}{}'.format(sep, value))

        elif key == 'norm':
            label_strs.append(
                '{}{}{}'.format(
                    key, sep, format_num(value, sigfigs=3, sci_thresh=(4, -3))
                )
            )

    if not label_strs:
        return ''

    if fname:
        return '__'.join(label_strs)

    label_lines = [label_strs[0]]
    for label_str in label_strs[1:]:
        if len(label_lines[-1]) + len(label_str) > 120:
            label_lines.append(label_str)
        else:
            label_lines[-1] += ', ' + label_str

    return line_sep.join(label_lines)


def plot_clsim_table_summary(
        summaries, formats=None, outdir=None, no_legend=False
    ):
    """Plot the table summary produced by `summarize_clsim_table`.

    Plots are made of marginalized 1D distributions, where mean, median, and/or
    max are used to marginalize out the remaining dimensions (where those are
    present in the summaries)..

    Parameters
    ----------
    summaries : string, summary, or iterable thereof
        If string(s) are provided, each is glob-expanded. See
        :method:`glob.glob` for valid syntax.

    formats : None, string, or iterable of strings in {'pdf', 'png'}
        If no formats are provided, the plot will not be saved.

    outdir : None or string
        If `formats` is specified and `outdir` is None, the plots are
        saved to the present working directory.

    no_legend : bool, optional
        Do not display legend on plots (default is to display a legend)

    Returns
    -------
    all_figs : list of three :class:`matplotlib.figure.Figure`

    all_axes : list of three lists of :class:`matplotlib.axes.Axes`

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

    if formats is None:
        formats = []
    elif isinstance(formats, basestring):
        formats = [formats]

    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)

    n_summaries = len(summaries)

    if n_summaries == 0:
        raise ValueError(
            'No summaries found based on argument `summaries`={}'
            .format(orig_summaries)
        )

    for n, fmt in enumerate(formats):
        fmt = fmt.strip().lower()
        assert fmt in ('pdf', 'png'), fmt
        formats[n] = fmt

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

    same_label = formatter(same_items)

    summary_has_detail = False
    if set(['string', 'depth_idx', 'seed']).issubset(all_items.keys()):
        summary_has_detail = True
        strings = sorted(set(all_items['string']))
        depths = sorted(set(all_items['depth_idx']))
        seeds = sorted(set(all_items['seed']))

    plot_kinds = ('mean', 'median', 'max')
    plot_kinds_with_data = set()
    dim_names = summaries[0]['dimensions'].keys()
    n_dims = len(dim_names)

    fig_x = 10 # inches
    fig_header_y = 0.35 # inches
    fig_one_axis_y = 5 # inches
    fig_all_axes_y = n_dims * fig_one_axis_y
    fig_y = fig_header_y + fig_all_axes_y # inches

    all_figs = []
    all_axes = []

    for plot_kind in plot_kinds:
        fig, f_axes = plt.subplots(
            nrows=n_dims, ncols=1, squeeze=False, figsize=(fig_x, fig_y)
        )
        all_figs.append(fig)
        f_axes = list(f_axes.flat)
        for ax in f_axes:
            ax.set_prop_cycle('color', COLOR_CYCLE_ORTHOG)
        all_axes.append(f_axes)

    n_lines = 0
    xlims = [[np.inf, -np.inf]] * n_dims

    summaries_order = []
    if summary_has_detail:
        for string, depth_idx, seed in product(strings, depths, seeds):
            for summary_n, summary in enumerate(summaries):
                if (summary['string'] != string
                        or summary['depth_idx'] != depth_idx
                        or summary['seed'] != seed):
                    continue
                summaries_order.append((summary_n, summary))
    else:
        for summary_n, summary in enumerate(summaries):
            summaries_order.append((summary_n, summary))

    labels_assigned = set()
    for summary_n, summary in summaries_order:
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
            dim_axes = [f_axes[dim_num] for f_axes in all_axes]
            bin_edges = summary[dim_name + '_bin_edges']
            if dim_name == 'deltaphidir':
                bin_edges /= np.pi
            xlims[dim_num] = [
                min(xlims[dim_num][0], np.min(bin_edges)),
                max(xlims[dim_num][1], np.max(bin_edges))
            ]
            for ax, plot_kind in zip(dim_axes, plot_kinds):
                if plot_kind not in dim_info:
                    continue
                plot_kinds_with_data.add(plot_kind)
                vals = dim_info[plot_kind]
                ax.step(bin_edges, [vals[0]] + list(vals),
                        linewidth=1, clip_on=True,
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
    logy_dims = ['r', 'time', 'deltaphidir']

    flabel = ''
    same_flabel = formatter(same_items, fname=True)
    different_flabel = formatter(different_items, key_only=True, fname=True)
    if same_flabel:
        flabel += '__same__' + same_flabel
    if different_flabel:
        flabel += '__differ__' + different_flabel

    for kind_idx, (plot_kind, fig) in enumerate(zip(plot_kinds, all_figs)):
        if plot_kind not in plot_kinds_with_data:
            continue
        for dim_num, (dim_name, ax) in enumerate(zip(dim_names, all_axes[kind_idx])):
            #if dim_num == 0 and different_items:
            if different_items and not no_legend:
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

        for fmt in formats:
            outfpath = ('clsim_table_summaries{}__{}.{}'
                        .format(flabel, plot_kind, fmt))
            if outdir:
                outfpath = join(outdir, outfpath)
            fig.savefig(outfpath, dpi=300)
            print('Saved image to "{}"'.format(outfpath))

    return all_figs, all_axes, summaries


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--formats', choices=('pdf', 'png'), nargs='+', default='pdf',
        help='''Save plots to chosen format(s). Choices are "pdf" and "png".'''
    )
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory to which to save the plot(s). Defaults to same
        directory as the present working directory.'''
    )
    parser.add_argument(
        '--no-legend', action='store_true',
        help='''Do not display a legend on the individual plots'''
    )
    parser.add_argument(
        'summaries', nargs='+',
        help='''Path(s) to summary JSON files to plot. Note that literal
        strings are glob-expanded.'''
    )
    return parser.parse_args()


def main():
    """Main function for calling plot_clsim_table_summary as a script"""
    args = parse_args()
    kwargs = vars(args)
    plot_clsim_table_summary(**kwargs)


if __name__ == '__main__':
    main()
