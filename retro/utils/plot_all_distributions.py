# pylint: disable=bad-indentation

"""
Plot distributions and error distributions comparing Retro to Pegleg (and truth).
"""

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
from os.path import expanduser, expandvars, isdir, join
from os import makedirs

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def interquartile_range(x, lower_percentile=25, upper_percentile=75):
    """Compute the inter-quartile range for values.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    iq_range : float
        Interquartile range

    """
    return np.diff(np.percentile(x, [lower_percentile, upper_percentile]))[0]


def plot_res(xedges, bin_indices, errcol, ax, med_color, quant_color, fill_color,
             med_lw=1, quant_lw=1, number=False):
    quantiles = [0.25, 0.5, 0.75]

    quants = []
    counts = []
    for idx, grp in errcol.groupby(bin_indices):
        quants.append(grp.quantile(quantiles))
        counts.append((idx, grp.count()))
    quants = np.array(quants)
    #print('')
    #counts = np.array(counts)

    x = xedges
    y0 = quants[:, 0]
    y2 = quants[:, 2]

    ax.plot(ebin_edges, np.zeros_like(ebin_edges), 'w:', lw=0.5, alpha=1) #, zorder=-1)

    if fill_color not in [None, 'none']:
        for blk in range(len(ebin_edges) - 1):
            ee = ebin_edges[blk:blk+2]
            y00 = y0[blk], y0[blk]
            y22 = y2[blk], y2[blk]
            ax.fill_between(ee, y00, y22, color=fill_color)

    if med_color not in [None, 'none']:
        for col in [1]:
            y = quants[:, col]
            if col in [0, 2]:
                ls = '--'
            else:
                ls = '-'
            ax.step(x, [y[0]] + y.tolist(), lw=1, ls=ls, c=med_color)

    if quant_color not in [None, 'none']:
        for col in [0, 2]:
            y = quants[:, col]
            if col in [0, 2]:
                ls = '--'
            else:
                ls = '-'
            ax.step(x, [y[0]] + y.tolist(), lw=1, ls=ls, c=quant_color)

    return counts, quants

def stuff():
    ebin_edges = np.logspace(np.log10(1), np.log10(100), 6)

    fill = (0.3,)*3
    figsize = (6, 3)
    dpi = 200
    mc = (.5, .5,0)
    qc = (1, 1, 0)

    bin_indices = pd.cut(events['energy'], bins=ebin_edges, right=False)

    # -- total energy -- #

    ymin, ymax = -0.3, 0.8

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_lograt_en_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    #cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_lograt_en_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_fract_en_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_fract_en_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro track+cascade energy resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.arange(ymin, ymax+0.1, 0.1), minor=True)
    #ax.set_yticks(np.arange(-5.5, 5.51, 0.5), minor=False)
    #ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r'true energy (GeV)')
    #ax.set_ylabel(r'$\log_{10}(E_{\rm reco} / E_{\rm true}), \; {\rm track + cascade}$')
    ax.set_ylabel(r'$E_{\rm reco} / E_{\rm true} - 1, \; {\rm track + cascade}$')
    remove_border(ax)

    fig.tight_layout()

    #outfbase = join(outdir, 'log10_reco_energy_by_true_energy')
    outfbase = join(outdir, 'fract_energy_error')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- track energy -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_lograt_track_en_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_lograt_track_en_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro track energy resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.arange(-.5, .51, 0.1), minor=True)
    #ax.set_yticks(np.arange(-.5, .51, 0.5), minor=False)
    #ax.set_ylim(-.5, .5)

    ax.set_xlabel(r'true energy (GeV)')
    #ax.set_ylabel(r'$\log_{10}(E_{\rm reco} / E_{\rm true}), \; {\rm track}$')
    ax.set_ylabel(r'$E_{\rm reco} / E_{\rm true} - 1, \; {\rm track}$')
    remove_border(ax)

    fig.tight_layout()

    #outfbase = join(outdir, 'log10_reco_track_energy_by_true_track_energy')
    outfbase = join(outdir, 'fract_track_energy_error')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- cascade energy -- #

    ymin, ymax = -1.5, 2

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_lograt_cascade_en_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_lograt_cascade_en_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro cscd energy resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    # ax.set_yticks(np.arange(ymin, ymax+0.1, 0.1), minor=True)
    # ax.set_yticks(np.arange(-5.5, 5.51, 0.5), minor=False)
    #ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r'true energy (GeV)')
    #ax.set_ylabel(r'$\log_{10}(E_{\rm reco} / E_{\rm true}), \; {\rm cascade}$')
    ax.set_ylabel(r'$E_{\rm reco} / E_{\rm true} - 1, \; {\rm cascade}$')
    remove_border(ax)

    fig.tight_layout()

    #outfbase = join(outdir, 'log10_reco_cascade_energy_by_true_track_energy')
    outfbase = join(outdir, 'fract_cascade_energy_error')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- zenith -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_zen_err*180/np.pi, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_zen_err*180/np.pi, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro zenith angle resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.linspace(-25, 25, 11), minor=True)
    #ax.set_ylim(-25, 25)
    ax.set_xlabel(r'true energy (GeV)')
    ax.set_ylabel(r'reco zenith $-$ true zenith (deg)')
    remove_border(ax)

    fig.tight_layout()

    outfbase = join(outdir, 'reco_zenith-true_zenith')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- coszen -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_coszen_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_coszen_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro coszen resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.linspace(-25, 25, 11), minor=True)
    #ax.set_ylim(-25, 25)
    ax.set_xlabel(r'true energy (GeV)')
    ax.set_ylabel(r'reco cos(zen) $-$ true cos(zen)')
    remove_border(ax)

    fig.tight_layout()

    outfbase = join(outdir, 'reco_coszen-true_coszen')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- track coszen -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_track_coszen_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_track_coszen_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro track coszen resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.linspace(-25, 25, 11), minor=True)
    #ax.set_ylim(-25, 25)
    ax.set_xlabel(r'true energy (GeV)')
    ax.set_ylabel(r'reco - true, $cos(\theta_{\rm zen})$ track')
    remove_border(ax)

    fig.tight_layout()

    outfbase = join(outdir, 'reco_track_coszen-true_track_coszen')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- track azimuth -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_track_az_err, ax=ax, med_color='k', quant_color=None, fill_color=fill)
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=mn_track_az_err, ax=ax, med_color=mc, quant_color=qc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro track azimuth resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    #ax.set_yticks(np.linspace(-25, 25, 11), minor=True)
    #ax.set_ylim(-25, 25)
    ax.set_xlabel(r'true energy (GeV)')
    ax.set_ylabel(r'reco - true, $\phi_{\rm az}$ track')
    remove_border(ax)

    fig.tight_layout()

    outfbase = join(outdir, 'reco_track_az-true_track_az')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')

    # -- angle -- #

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cnt0, quants0 = plot_res(ebin_edges, bin_indices, errcol=plmn_alpha*180/np.pi, ax=ax, med_color='w', quant_color='gray', fill_color='none')
    cnt1, quants1 = plot_res(ebin_edges, bin_indices, errcol=retro_alpha*180/np.pi, ax=ax, med_color=qc, quant_color=mc, fill_color='none')

    res0 = np.diff(quants0[:, [0, -1]], axis=1)
    res1 = np.diff(quants1[:, [0, -1]], axis=1)
    print('Retro angle resolutions are:')
    for ivl, r in zip([c[0] for c in cnt0], 100 * (res1 / res0)):
        print('    In [{:5.1f}, {:5.1f}] GeV : {:5.1f}% of PegLeg/MN'.format(ivl.left, ivl.right, r[0]))

    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
    ax.set_xscale('log')

    # ax.set_yticks(np.arange(0, 105+1, 15), minor=True)
    # ax.set_yticks(np.arange(0, 91, 30), minor=False)
    #ax.set_ylim(0, 105)
    ax.set_xlabel(r'true energy (GeV)')
    ax.set_ylabel(r'track angle error (deg)')
    remove_border(ax)

    fig.tight_layout()

    outfbase = join(outdir, 'reco_opening_angle')
    fig.savefig(outfbase + '.png', dpi=300, transparent=True)
    fig.savefig(outfbase + '.pdf')


def plot_comparison(
    pegleg,
    retro,
    xlim,
    truth=None,
    nbins=102,
    logx=False,
    logy=False,
    xlab=None,
    pl_label='Pegleg',
    retro_label='Retro',
    truth_label='Truth',
    stats=('iq50', 'median'),
    n_decimals=3,
    leg_loc='best',
    ax=None,
):
    """Plot comparison between pegleg, retro, and possibly truth distributions."""
    if pegleg is not None:
        pegleg = pegleg[np.isfinite(pegleg)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
        newfig = True
    else:
        fig = ax.get_figure()
        newfig = False

    if logx:
        b = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), nbins)
    else:
        b = np.linspace(xlim[0], xlim[1], nbins)

    if stats:
        stats = tuple(s.lower() for s in stats)
        stats_labels = {
            k: v for k, v in dict(iq50='IQ 50%', mean='mean', median='median').items()
            if k in stats
        }
        label_chars = 1 + int(np.max([len(l) for l in stats_labels.values()]))
        number_chars = 2 + n_decimals + int(n_decimals > 0)
        fmt_str = '{:%d.%df}' % (number_chars, n_decimals)
        total_chars = label_chars + number_chars

        stats = tuple(s.lower() for s in stats)
        pl_label_ = pl_label
        retro_label_ = retro_label
        truth_label_ = truth_label
        pl_sublabs = []
        retro_sublabs = []
        truth_sublabs = []
        for stat in stats:
            label = stats_labels[stat].ljust(label_chars) + fmt_str
            if stat == 'iq50':
                retro_sublabs.append(label.format(interquartile_range(retro)))
                if pegleg is not None:
                    pl_sublabs.append(label.format(interquartile_range(pegleg)))
                if truth is not None:
                    truth_sublabs.append(label.format(interquartile_range(truth)))

            elif stat == 'median':
                retro_sublabs.append(label.format(np.median(retro)))
                if pegleg is not None:
                    pl_sublabs.append(label.format(np.median(pegleg)))
                if truth is not None:
                    truth_sublabs.append(label.format(np.median(truth)))

            elif stat == 'mean':
                retro_sublabs.append(label.format(np.mean(retro)))
                if pegleg is not None:
                    pl_sublabs.append(label.format(np.mean(pegleg)))
                if truth is not None:
                    truth_sublabs.append(label.format(np.mean(truth)))

        pl_label = '    {}\n{}'.format(pl_label_, '\n'.join(pl_sublabs))
        retro_label = '    {}\n{}'.format(retro_label_, '\n'.join(retro_sublabs))
        if truth is not None:
            truth_label = '    {}\n{}'.format(
                truth_label_, '\n'.join(retro_sublabs)
            )

    if truth is not None:
        _, b, _ = ax.hist(
            truth,
            bins=b,
            histtype='stepfilled',
            lw=2,
            color=(0.7,)*3,
            label=truth_label,
        )
    _, b, _ = ax.hist(
        retro,
        bins=b,
        histtype='step',
        lw=2,
        color='C0',
        label=retro_label,
    )
    if pegleg is not None:
        _, b, _ = ax.hist(
            pegleg,
            bins=b,
            histtype='step',
            lw=2,
            color='C1',
            label=pl_label,
            zorder=-1,
        )

    ylim = ax.get_ylim()
    if truth is None:
        ax.plot([0, 0], ylim, 'k-', lw=0.5, zorder=-10)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    leg = ax.legend(loc=leg_loc, prop={'family': 'monospace'}, frameon=False)

    def get_children(obj):
        ch = [obj]
        try:
            children = obj.get_children()
        except:
            return ch
        for child in children:
            ch.extend(get_children(child))
        return ch

    children = get_children(leg)
    das = [c for c in children if isinstance(c, mpl.offsetbox.DrawingArea)]
    rs = [c for c in children if isinstance(c, mpl.patches.Rectangle)]
    for da in das:
        da.set_height(0)
        da.set_width(0)
        #da.xdescent = 200
        #da.ydescent = 200
        offset = da.get_offset()
        offset = (offset[0] + 200, offset[1] + 200)
        da.set_offset(offset)
    for r in rs:
        r.set_width(15)
        r.set_height(6.5)
        r.set_xy((10, 10))

    if xlab is not None:
        ax.set_xlabel(xlab)
    if newfig:
        fig.tight_layout()
    return fig, ax, leg


def plot_all_distributions(
    evts,
    p_reco="Pegleg_Fit_MN",
    r_reco="retro",
    outdir=None,
    n_xe=50,
    n_ye=50,
    n_ze=50,
    n_te=50,
    n_zenerr=50,
    n_azerr=50,
    n_cscdenerr=50,
    n_trckenerr=50,
    n_enerr=50,
    n_x=50,
    n_y=50,
    n_z=50,
    n_t=50,
    n_zen=50,
    n_az=50,
    n_cscden=50,
    n_trcken=50,
    n_en=50,
    n_ang=50,
):
    """Plot all distributions comparing Pegleg, Retro, and, where appropriate, truth.

    Returns
    -------
    axes : OrderedDict

    """
    if outdir is not None:
        outdir = expanduser(expandvars(outdir))
        if not isdir(outdir):
            makedirs(outdir, mode=0o750)

    axes = OrderedDict()

    truth = evts.x
    pegleg = evts['{}_x'.format(p_reco)] if p_reco else None
    retro = evts['{}_x'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_xe,
        xlab='reco x - true x (m)',
    )
    axes['x_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_x_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-150, 250),
        nbins=n_x,
        xlab='x (m)',
    )
    axes['x'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_x')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.y
    pegleg = evts['{}_y'.format(p_reco)] if p_reco else None
    retro = evts['{}_y'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_ye,
        xlab='reco y - true y (m)',
    )
    axes['y_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_y_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-250, 150),
        nbins=n_y,
        xlab='y (m)',
    )
    axes['y'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_y')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.z
    pegleg = evts['{}_z'.format(p_reco)] if p_reco else None
    retro = evts['{}_z'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_ze,
        xlab='reco z - true z (m)',
    )
    axes['z_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_z_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-650, -150),
        nbins=n_z,
        xlab='z (m)',
    )
    axes['z'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_z')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.time
    pegleg = evts['{}_time'.format(p_reco)] if p_reco else None
    retro = evts['{}_time'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-100, 200),
        nbins=n_te,
        xlab='reco time - true time (ns)',
    )
    axes['time_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_time_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(9.4e3, 9.925e3),
        nbins=n_t,
        xlab='time (ns)',
    )
    axes['time'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_time')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = np.arccos(evts.coszen)
    if p_reco:
        key0 = '{}_track_zenith'.format(p_reco)
        key1 = '{}_zenith'.format(p_reco)
        key2 = '{}_coszen'.format(p_reco)
        if key0 in evts:
            pegleg = evts[key0]
        elif key1 in evts:
            pegleg = evts[key1]
        elif key2 in evts:
            pegleg = np.arccos(evts[key2])
    else:
        pegleg = None

    retro = evts['{}_track_zenith'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-.75, .75),
        nbins=n_zenerr,
        xlab=r'reco $\nu$ zenith - true $\nu$ zenith (rad)',
    )
    axes['zenith_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_zenith_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(0, np.pi),
        nbins=n_zen,
        xlab=r'$\nu$ zenith (rad)',
    )
    axes['zenith'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_zenith')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.azimuth
    pegleg = evts['{}_azimuth'.format(p_reco)] if p_reco else None
    retro = evts['{}_azimuth'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=(pegleg - truth + np.pi) % (2*np.pi) - np.pi if p_reco else None,
        retro=(retro - truth + np.pi) % (2*np.pi) - np.pi,
        xlim=(-np.pi/8, np.pi/8),
        nbins=n_azerr,
        xlab=r'reco $\nu$ azimuth - true $\nu$ azimuth (rad)',
    )
    axes['azimuth_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_azimuth_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(0, 2*np.pi),
        nbins=n_az,
        xlab=r'$\nu$ azimuth (rad)',
    )
    axes['azimuth'] = ax
    ax.legend(loc='lower right')
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_azimuth')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    if p_reco:
        p_err = evts['{}_angle_error'.format(p_reco)]
        p_err = p_err[np.isfinite(p_err)]
    else:
        p_err = None

    r_err = evts['{}_angle_error'.format(r_reco)]
    r_err = r_err[np.isfinite(r_err)]

    fig, ax, leg = plot_comparison(
        pegleg=p_err,
        retro=r_err,
        xlim=(0, np.pi),
        nbins=n_ang,
        xlab=r'$\nu$ angle error (rad)',
    )
    axes['angle_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_angle_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.cascade0_em_equiv_energy + evts.cascade1_em_equiv_energy
    pegleg = evts['{}_cascade_energy'.format(p_reco)] if p_reco else None
    retro = evts['{}_cascade_energy'.format(r_reco)]
    mask = np.isfinite(retro) & np.isfinite(truth)
    r_err = retro[mask] - truth[mask]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=r_err,
        xlim=(-20, 20),
        nbins=n_cscdenerr,
        xlab='reco cascade energy - true cascade energy (GeV)',
    )
    axes['cascade_energy_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_cascade_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro[mask],
        truth=truth[mask],
        xlim=(0.0, 20),
        nbins=n_cscden,
        logx=False,
        xlab='cascade energy (GeV)',
    )
    axes['cascade_energy'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_cascade_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.track_energy
    pegleg = evts['{}_track_energy'.format(p_reco)] if p_reco else None
    retro = evts['{}_track_energy'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-30, 40),
        nbins=n_trckenerr,
        xlab='reco track energy - true track energy (GeV)',
    )
    axes['track_energy_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_track_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(.5, 500),
        nbins=n_trcken,
        logx=True,
        xlab='track energy (GeV)',
    )
    axes['track_energy'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_track_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.energy
    pegleg = evts['{}_cascade_energy'.format(p_reco)] + evts['{}_track_energy'.format(p_reco)] if p_reco else None
    retro = evts['{}_cascade_energy'.format(r_reco)]*2 + evts['{}_track_energy'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth if p_reco else None,
        retro=retro - truth,
        xlim=(-40, 40),
        nbins=n_enerr,
        xlab=r'reco $\nu$ energy - true $\nu$ energy (GeV)',
    )
    axes['neutrino_energy_err'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax, leg = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(1, 500),
        nbins=n_en,
        logx=True,
        xlab=r'$\nu$ energy (GeV)',
    )
    axes['neutrino_energy'] = ax
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    return axes
