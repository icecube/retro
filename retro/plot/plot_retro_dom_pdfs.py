#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Plot one or more results from running retro_dom_pdfs.py script
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'ks_test',
    'plot_run_info',
    'plot_run_info2',
    'parse_args',
    'main'
]

from argparse import ArgumentParser
from os.path import abspath, dirname, isdir, join
from os import makedirs
import sys

import numpy as np
#import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.const import get_string_dom_pair, get_sd_idx
from retro.utils.misc import expand


def ks_test(a, b):
    """https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test"""
    acs = np.cumsum(a)
    macs = np.max(acs)
    if macs == 0:
        return np.nan
    else:
        acs /= np.max(acs)
    bcs = np.cumsum(b)
    mbcs = np.max(bcs)
    if mbcs == 0:
        return np.nan
    else:
        bcs /= np.max(bcs)
    return np.max(np.abs(bcs - acs))


def num_fmt(n):
    """Simple number formatting"""
    return format(n, '.2e').replace('-0', '-').replace('+0', '')


def plot_run_info(
        files, labels, outdir, fwd_hists=None, data_or_sim_label=None,
        paired=False, gradient=False, plot=True
    ):
    """Plot `files` using `labels` (one for each file).

    Parameters
    ----------
    files : string or iterable thereof
    labels : string or iterable thereof
    outdir : string
    fwd_hists : string, optional
    data_or_sim_label : string, optional

    """
    if isinstance(files, basestring):
        files = [files]
    if isinstance(labels, basestring):
        labels = [labels]

    outdir = expand(outdir)

    if fwd_hists is not None:
        fwd_hists = load_pickle(fwd_hists)
        if 'binning' in fwd_hists:
            t_min = fwd_hists['binning']['t_min']
            t_max = fwd_hists['binning']['t_max']
            t_window = t_max - t_min
            num_bins = fwd_hists['binning']['num_bins']
            spacing = fwd_hists['binning']['spacing']
            assert spacing == 'linear', spacing
            fwd_hists_binning = np.linspace(t_min, t_max, num_bins + 1)
        elif 'bin_edges' in fwd_hists:
            fwd_hists_binning = fwd_hists['bin_edges']
            t_window = np.max(fwd_hists_binning) - np.min(fwd_hists_binning)
        else:
            raise ValueError(
                'Need "binning" or "bin_edges" in fwd_hists; keys are {}'
                .format(fwd_hists.keys())
            )
        hist_bin_widths = np.diff(fwd_hists_binning)
        if 'results' in fwd_hists:
            fwd_hists = fwd_hists['results']
        else:
            raise ValueError('Could not find key "results" in fwd hists!')
    else:
        raise NotImplementedError('Need fwd hists for now.')

    if not isdir(outdir):
        makedirs(outdir)

    run_infos = []
    all_string_dom_pairs = set()
    mc_true_params = None
    for filepath in files:
        filepath = expand(filepath)
        if isdir(filepath):
            filepath = join(filepath, 'run_info.pkl')
        run_info = load_pickle(filepath)
        run_infos.append(run_info)
        pairs = []
        for sd_idx in run_info['sd_indices']:
            pairs.append(get_string_dom_pair(sd_idx))
        all_string_dom_pairs.update(pairs)
        if data_or_sim_label is None:
            data_or_sim_label = (
                'Simulation: '
                + run_info['sim_to_test'].replace('_', ' ').capitalize()
            )

        if mc_true_params is None:
            if 'sim' in run_info:
                mc_true_params = run_info['sim']['mc_true_params']
            else:
                print('mc_true_params not in run_info', filepath)

    params_label = None
    if mc_true_params is not None:
        params_label = []
        for plab, pval in mc_true_params.items():
            units = ''

            if plab == 't':
                pval = format(int(pval), 'd')
                #plab = r'{}'.format(plab)
                units = r'\, \rm{ ns}'

            elif plab in 'x y z'.split():
                pval = format(pval, '0.1f')
                #plab = r'${}$'.format(plab)
                units = r'\, \rm{ m}'

            elif plab in 'track_energy cascade_energy'.split():
                pval = format(int(pval), 'd')
                plab = r'E_{\rm %s}' % plab.split('_')[0]
                units = r'\, \rm{ GeV}'

            elif plab in 'track_azimuth track_zenith cascade_azimuth cascade_zenith'.split():
                pval = format(pval / np.pi, '.2f')
                if 'azimuth' in plab:
                    ltr = r'\phi'
                elif 'zenith' in plab:
                    ltr = r'\theta'
                plab = ltr + r'_{\rm %s}' % plab.split('_')[0]
                units = r'\, \pi'

            params_label.append('{}={}{}'.format(plab, pval, units))
        params_label = '$' + r',\;'.join(params_label) + '$'

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=72)

    t_indep_tots = []
    tots_incl_noise = []
    tots_excl_noise = []
    kss = []
    ref_tots_incl_noise = []
    ref_tots_excl_noise = []
    ref_areas_incl_noise = []
    for string, dom in reversed(sorted(all_string_dom_pairs)):
        if plot:
            ax.clear()
        all_zeros = True
        xmin = np.inf
        xmax = -np.inf
        ref_y = None
        if fwd_hists:
            if (string, dom) in fwd_hists:
                # Hit rate per nanosecond in each bin (includes noise hit rate)
                ref_y = fwd_hists[(string, dom)] / hist_bin_widths

                # Duplicate first element for plotting via `plt.step`
                ref_y = np.array([ref_y[0]] + ref_y.tolist())

                # Figure out "meaningful" range
                nonzero_mask = ref_y != 0 #~np.isclose(ref_y, 0)
                if np.any(nonzero_mask):
                    all_zeros = False
                    #ref_y_all_zeros = False
                    min_mask = (ref_y - ref_y.min()) >= 0.01 * (ref_y.max() - ref_y.min())
                    xmin = min(xmin, fwd_hists_binning[min_mask].min())
                    xmax = max(xmax, fwd_hists_binning[min_mask].max())
            else:
                ref_y = np.zeros_like(fwd_hists_binning)

            ref_y_areas = ref_y[1:] * hist_bin_widths
            ref_y_area = np.sum(ref_y_areas)

            ref_tots_incl_noise.append(ref_y_area)

            # Following only works if our time window is large enough s.t. exp
            # hits from event is zero somewhere, and then it'll only be noise
            # contributing at that time...
            ref_tots_excl_noise.append(np.sum(ref_y_areas - ref_y_areas.min()))
            ref_areas_incl_noise.append(ref_y_area)

            if plot:
                ax.step(
                    fwd_hists_binning, ref_y,
                    lw=1,
                    label=(
                        r'Fwd: $\Sigma \lambda_q \Delta t$={}'
                        .format(num_fmt(ref_y_area))
                    ),
                    clip_on=True,
                    #color='C0'
                )

        colors = ['C%d' % i for i in range(1, 10)]
        linestyles = ['-', '--']
        linewidths = [5, 3, 2, 2, 2, 2, 2]

        for plt_i, (label, run_info) in enumerate(zip(labels, run_infos)):
            sample_hit_times = run_info['hit_times']
            if len(tots_incl_noise) <= plt_i:
                tots_incl_noise.append([])
                tots_excl_noise.append([])
                t_indep_tots.append([])
                kss.append([])

            results = run_info['results']
            if (string, dom) in pairs:
                rslt = results[get_sd_idx(string, dom)]
                if 'exp_p_at_hit_times' in rslt:
                    y = rslt['exp_p_at_hit_times']
                    y_ti = rslt['exp_p_at_all_times']
                    t_indep_tots[plt_i].append(y_ti)
                else:
                    y = rslt['pexp_at_hit_times']

                nonzero_mask = y != y[0] #~np.isclose(y, 0)
                if np.any(nonzero_mask):
                    all_zeros = False
                    min_mask = y >= 0.01 * max(y)
                    xmin = min(xmin, sample_hit_times[min_mask].min())
                    xmax = max(xmax, sample_hit_times[min_mask].max())
            else:
                y = np.zeros_like(sample_hit_times)

            #y_area = np.sum(

            masked_y = np.ma.masked_invalid(y * hist_bin_widths)
            tot_excl_noise = np.sum(masked_y - masked_y.min())
            tot_incl_noise = masked_y.sum()
            if tot_excl_noise != 0:
                tots_excl_noise[plt_i].append(tot_excl_noise)
                tots_incl_noise[plt_i].append(tot_incl_noise)
            else:
                tots_excl_noise[plt_i].append(0)
                tots_incl_noise[plt_i].append(0)
            kss[plt_i].append(ks_test(y, ref_y[1:]))

            #kl_div = None
            custom_label = r'{:3s}: $\Sigma \lambda_q \Delta t$={}, ti={}'.format(
                label, num_fmt(tots_incl_noise[plt_i][-1]), num_fmt(y_ti)
            )
            #if ref_y is not None: # and not ref_y_all_zeros:
            #    abs_mean_diff = np.abs(np.mean(y - ref_y[1:]))
            #    #rel_abs_mean_diff = abs_mean_diff / np.sum(ref_y[1:])

            #    mask = ref_y[1:] > 0
            #    kl_ref_vals = ref_y[1:][mask]
            #    kl_ref_vals /= np.sum(kl_ref_vals)

            #    y_prob_vals = y[mask]
            #    y_prob_vals /= np.sum(y_prob_vals)

            #    with np.errstate(divide='ignore'):
            #        kl_div = -np.sum(kl_ref_vals * np.log(y_prob_vals / kl_ref_vals))
            #    custom_label = format(rel_abs_mean_diff, '9.6f') + '  ' + label

            if paired:
                c_idx, ls_idx = divmod(plt_i, 2)
                color = colors[c_idx]
                linestyle = linestyles[ls_idx]
            else:
                color = None
                linestyle = None

            if plot:
                ax.plot(
                    sample_hit_times, y,
                    label=custom_label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidths[plt_i],
                    clip_on=True
                )

        if all_zeros:
            continue

        if xmin == xmax:
            xmin = np.min(fwd_hists_binning)
            xmax = np.max(fwd_hists_binning)

        if plot:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, ax.get_ylim()[1])

            for pos in 'bottom left top right'.split():
                ax.spines[pos].set_visible(False)

            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()

            #if kl_div is not None:
            #title = ' '*6 + 'Abs diff'.ljust(8) + '  ' + 'Simulation'
            #else:
            title = 'Code'

            leg = ax.legend(
                #title=title,
                #loc='best',
                loc='upper right',
                #frameon=False,
                framealpha=0.7,
                prop=dict(family='monospace', size=12)
            )
            plt.setp(leg.get_title(), family='monospace', fontsize=12)
            #if kl_div is not None:
            #leg._legend_box.align = "left"
            leg.get_frame().set_linewidth(0)
            ax.set_xlabel('Time from event vertex (ns)', fontsize=14)

            if data_or_sim_label is not None:
                plt.text(
                    0.5, 1.1,
                    data_or_sim_label,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=16
                )
            if params_label is not None:
                plt.text(
                    0.5, 1.05,
                    params_label,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=12
                )

            ax.text(
                0.5, 1.0,
                'String {}, DOM {}'.format(string, dom),
                ha='center', va='bottom',
                transform=ax.transAxes,
                fontsize=14
            )

            fbasename = 'string_{}_dom_{}'.format(string, dom)
            fig.savefig(join(outdir, fbasename + '.png'))
            sys.stdout.write('({}, {}) '.format(string, dom))
            sys.stdout.flush()
    sys.stdout.write('\n\n')
    sys.stdout.flush()

    ref_tots_incl_noise = np.array(ref_tots_incl_noise)
    ref_tots_excl_noise = np.array(ref_tots_excl_noise)
    ref_areas_incl_noise = np.array(ref_areas_incl_noise)

    ref_tot_incl_noise = np.sum(ref_tots_incl_noise)
    ref_tot_excl_noise = np.sum(ref_tots_excl_noise)
    ref_area_incl_noise = np.sum(ref_areas_incl_noise)

    print(
        '{:9s}  {:9s}  {:16s}  {:16s}  {:16s}  {}'
        .format(
            'wtd KS'.rjust(9),
            'avg KS'.rjust(9),
            'Ratio incl noise'.rjust(16),
            'Ratio excl noise'.rjust(16),
            't-indep ratio'.rjust(16),
            'Label'
        )
    )
    for label, ks, tot_incl_noise, tot_excl_noise, ti_tot in zip(labels,
                                                                 kss,
                                                                 tots_incl_noise,
                                                                 tots_excl_noise,
                                                                 t_indep_tots):
        ks = np.array(ks)
        mask = ~np.isnan(ks)
        ks_avg = np.mean(ks[mask])
        ks_wtd_avg = (
            np.sum(ks[mask] * ref_tots_excl_noise[mask])
            / np.sum(ref_tots_excl_noise[mask])
        )
        print(
            '{:9s}  {:9s}  {:16s}  {:16s}  {:16s}  {}'
            .format(
                format(ks_wtd_avg, '.7f').rjust(9),
                format(ks_avg, '.7f').rjust(9),
                format(np.sum(tot_excl_noise) / ref_tot_excl_noise, '.12f').rjust(16),
                format(np.sum(tot_incl_noise) / ref_tot_incl_noise, '.12f').rjust(16),
                format(np.sum(ti_tot) / ref_area_incl_noise, '.12f').rjust(16),
                label
            )
        )


def plot_run_info2(
    fpath,
    only_string,
    subtract_noisefloor=True,
    plot_ref=True,
    scalefact=None,
    axes=None,
):
    """Plot information from `run_info.pkl` file as produced by
    `retro_dom_pdfs.py` script.

    Parameters
    ----------
    fpath : str
        Full path to `run_info.pkl` file
    only_string : int in [1, 86]
        String to plot
    subtract_noisefloor : bool, optional
        Whether to subtract the miniminum value from each distribution, which
        (usually but not always) is the noise floor
    plot_ref : bool, optional
        Plot the forward-simulation distribution
    scalefact : float, optional
        If not specified, a scale factor will be derived from the ratio between
        the forward-simulation and Retro distributions
    axes : length-3 sequence of matplotlib.axis, optional
        Provide the axes on which to plot the distributions; otherwise, a new
        figure with 3 axes will be created

    Returns
    -------
    fig : matplotlib.figure
    axes : length-3 list of matplotlib.axis

    """
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(16, 24), dpi=120)
    else:
        assert len(axes) == 3
        fig = axes[0].get_figure()

    subtract_noisefloor = 1 if subtract_noisefloor else 0

    # -- Extract info from files -- #

    fpath = expand(fpath)
    if isdir(fpath):
        fpath = join(fpath, 'run_info.pkl')
    info = load_pickle(fpath)

    sd_indices = info['sd_indices']
    hit_times = info['hit_times']
    dom_exp = info['dom_exp']
    hit_exp = info['hit_exp']
    dt = np.diff(hit_times)

    fwd = load_pickle(info['sim']['fwd_sim_histo_file'])
    bin_edges = fwd['bin_edges']
    fwd_results = fwd['results']
    dt = np.diff(bin_edges)

    # -- Figure out how many lines are to be plotted -- #

    total_num_lines = 0
    for idx, sd_idx in enumerate(sd_indices):
        he = hit_exp[idx, :]
        string, dom = get_string_dom_pair(sd_idx)
        if string != only_string or np.sum(he) == 0:
            continue
        total_num_lines += 1

    # -- Get info from all distributions -- #

    weights = []
    rats = []
    xmin = np.inf
    ymax = -np.inf
    ymin_at_3k = np.inf
    absdiff3k = np.abs(hit_times - 3000)
    idx_at_3k = np.where(absdiff3k == np.min(absdiff3k))[0][0]
    for idx, sd_idx in enumerate(sd_indices):
        he = hit_exp[idx, :] * dt
        he -= np.min(he)
        string, dom = get_string_dom_pair(sd_idx)
        if np.sum(he) == 0 or (string, dom) not in fwd_results:
            continue
        ref = fwd_results[(string, dom)]
        ref -= np.min(ref)
        mask = (he > 1e-12) & (ref >= 1e-12)
        rats.append(np.sum((ref[mask] / he[mask])*ref[mask]))
        weights.append(np.sum(ref[mask]))
        if string != only_string:
            continue
        xmin_idx = np.where(ref > 0)[0][0]
        xmin = min(xmin, hit_times[xmin_idx])
        ymax = max(ymax, np.max(ref))
        ymin_at_3k = min(ymin_at_3k, ref[idx_at_3k])
    wtdavg_rat = np.sum(rats) / np.sum(weights)
    xmin -= 50
    if ymin_at_3k == 0:
        ymin_at_3k = ymax / 1e6

    if scalefact is None:
        print('wtdavg_rat:', wtdavg_rat, '(using as scalefact)')
        scalefact = wtdavg_rat
    else:
        print('wtdavg_rat:', wtdavg_rat, '(but using {} as scalefact)'.format(scalefact))

    def innerplot(ax): # pylint: disable=missing-docstring
        for idx, sd_idx in enumerate(sd_indices):
            he = hit_exp[idx, :]
            string, dom = get_string_dom_pair(sd_idx)
            if string != only_string or np.sum(he) == 0:
                continue
            line, = ax.plot(
                hit_times, scalefact*(he*dt - subtract_noisefloor*np.min(he*dt)),
                '-', lw=1, label='({}, {})'.format(string, dom)
            )
            if not plot_ref or (string, dom) not in fwd_results:
                continue
            ref = fwd_results[(string, dom)]
            ax.plot(
                hit_times, ref - subtract_noisefloor*np.min(ref),
                linestyle='--', lw=0.5, color=line.get_color()
            )

    # -- Plot overview of distributions -- #

    ax = axes[0]
    num_lines = total_num_lines
    cm = plt.cm.gist_rainbow
    ax.set_prop_cycle('color', [cm(1.*i/num_lines) for i in range(num_lines)])
    innerplot(ax)
    ax.set_ylim(ymin_at_3k, ymax*2)
    ax.set_xlim(xmin, min(xmin+2000, 3000))
    ax.legend(loc='best', fontsize=8, ncol=4, frameon=False)

    # -- Zoom on peaks -- #

    ax = axes[1]
    num_lines = 20
    cm = plt.cm.tab20
    ax.set_prop_cycle('color', [cm(1.*i/num_lines) for i in range(num_lines)])
    innerplot(ax)
    ax.set_ylim(ymax/5e3, ymax*3)
    ax.set_xlim(xmin+25, xmin+750)
    ax.legend(loc='best', fontsize=7, ncol=14, frameon=False)

    # -- Zoom on tails -- #

    ax = axes[2]
    num_lines = 20
    cm = plt.cm.tab20
    ax.set_prop_cycle('color', [cm(1.*i/num_lines) for i in range(num_lines)])
    innerplot(ax)
    ax.set_xlim(xmin+750, 3000)
    ax.set_ylim(ymin_at_3k/2, ymin_at_3k*1e3)
    ax.legend(loc='best', fontsize=7, ncol=6, frameon=False)

    # -- Set common plot things -- #

    axes[0].set_title(info['sim_to_test'])
    axes[-1].set_xlabel('Time (ns)')
    for ax in axes:
        ax.set_ylabel('Charge (PE)')
        ax.set_yscale('log')
    fig.tight_layout()

    return fig, axes


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--files', nargs='+', required=True,
        help='''One or more *run_info.pkl files to plot'''
    )
    parser.add_argument(
        '--labels', nargs='+', required=True,
        help='''One (legend) label per file'''
    )
    parser.add_argument(
        '--fwd-hists', required=False,
        help='''Path to the forward-simulation pickle file'''
    )
    parser.add_argument(
        '--outdir', required=True,
        help='''Directory into which to place the plots.'''
    )
    parser.add_argument(
        '--paired', action='store_true',
        help='''Display style for visually grouping pairs of lines with
        same-colors but different line styles.'''
    )
    parser.add_argument(
        '--gradient', action='store_true',
        help='''Display style for visually showing a progression from line to
        line via a color gradient.'''
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='''Do _not_ make plots, just print summary statistics.'''
    )
    return parser.parse_args()


def main():
    """Get command line args, translate, and run plot function"""
    kwargs = vars(parse_args())
    kwargs['plot'] = not kwargs.pop('no_plot')
    plot_run_info(**kwargs)


if __name__ == '__main__':
    main()
