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
                pl_sublabs.append(label.format(interquartile_range(pegleg)))
                retro_sublabs.append(label.format(interquartile_range(retro)))
                if truth is not None:
                    truth_sublabs.append(label.format(interquartile_range(truth)))

            elif stat == 'median':
                pl_sublabs.append(label.format(np.median(pegleg)))
                retro_sublabs.append(label.format(np.median(retro)))
                if truth is not None:
                    truth_sublabs.append(label.format(np.median(truth)))

            elif stat == 'mean':
                pl_sublabs.append(label.format(np.mean(pegleg)))
                retro_sublabs.append(label.format(np.mean(retro)))
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
    _, b, _ = ax.hist(retro, bins=b, histtype='step', lw=2, color='C0', label=retro_label)
    _, b, _ = ax.hist(pegleg, bins=b, histtype='step', lw=2, color='C1', label=pl_label,
                     zorder=-1)

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
    pegleg = evts['{}_x'.format(p_reco)]
    retro = evts['{}_x'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_y'.format(p_reco)]
    retro = evts['{}_y'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_z'.format(p_reco)]
    retro = evts['{}_z'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_time'.format(p_reco)]
    retro = evts['{}_time'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_track_zenith'.format(p_reco)]
    retro = evts['{}_track_zenith'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_track_azimuth'.format(p_reco)]
    retro = evts['{}_track_azimuth'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=(pegleg - truth + np.pi) % (2*np.pi) - np.pi,
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

    p_err = evts['{}_track_angle_error'.format(p_reco)]
    r_err = evts['{}_track_angle_error'.format(r_reco)]

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

    truth = evts.cascade_energy
    pegleg = evts['{}_cascade_energy'.format(p_reco)]
    retro = evts['{}_cascade_energy'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
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
        retro=retro,
        truth=truth,
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
    pegleg = evts['{}_track_energy'.format(p_reco)]
    retro = evts['{}_track_energy'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
    pegleg = evts['{}_cascade_energy'.format(p_reco)] + evts['{}_track_energy'.format(p_reco)]
    retro = evts['{}_cascade_energy'.format(r_reco)]*2 + evts['{}_track_energy'.format(r_reco)]

    fig, ax, leg = plot_comparison(
        pegleg=pegleg - truth,
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
