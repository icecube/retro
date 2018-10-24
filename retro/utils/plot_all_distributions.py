# pylint: disable=bad-indentation

"""
Plot distributions and error distributions comparing Retro to Pegleg (and truth).
"""

from __future__ import absolute_import, division, print_function

from os.path import expanduser, expandvars, isdir, join
from os import makedirs

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
    include_stats=True,
):
    """Plot comparison between pegleg, retro, and possibly truth distributions."""
    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
    if logx:
        b = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), nbins)
    else:
        b = np.linspace(xlim[0], xlim[1], nbins)

    if truth is None and include_stats:
        pl_label = (
            '    {}\nmean  {:7.3f}\nmed   {:7.3f}\nIQ50% {:7.3f}'
            .format(pl_label, np.mean(pegleg), np.median(pegleg), interquartile_range(pegleg))
        )
        retro_label = (
            '    {}\nmean  {:7.3f}\nmed   {:7.3f}\nIQ50% {:7.3f}'
            .format(retro_label, np.mean(retro), np.median(retro), interquartile_range(retro))
        )

    if truth is not None:
        _, b, _ = ax.hist(
            truth,
            bins=b,
            histtype='stepfilled',
            lw=2,
            color=(0.7,)*3,
            label='Truth',
        )
    _, b, _ = ax.hist(pegleg, bins=b, histtype='step', lw=2, color='C1', label=pl_label)
    _, b, _ = ax.hist(retro, bins=b, histtype='step', lw=2, color='C0', label=retro_label)

    ylim = ax.get_ylim()
    if truth is None:
        ax.plot([0, 0], ylim, 'k-', lw=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.legend(loc='best', prop={'family': 'monospace'})
    if xlab is not None:
        ax.set_xlabel(xlab)
    fig.tight_layout()
    return fig, ax


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
    """Plot all distributions comparing Pegleg, Retro, and, where appropriate, truth."""
    if outdir is not None:
        outdir = expanduser(expandvars(outdir))
        if not isdir(outdir):
            makedirs(outdir, mode=0o750)

    truth = evts.x
    pegleg = evts['{}_x'.format(p_reco)]
    retro = evts['{}_x'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_xe,
        xlab='reco x - true x (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_x_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-150, 250),
        nbins=n_x,
        xlab='x (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_x')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.y
    pegleg = evts['{}_y'.format(p_reco)]
    retro = evts['{}_y'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_ye,
        xlab='reco y - true y (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_y_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-250, 150),
        nbins=n_y,
        xlab='y (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_y')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.z
    pegleg = evts['{}_z'.format(p_reco)]
    retro = evts['{}_z'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_ze,
        xlab='reco z - true z (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_z_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(-650, -150),
        nbins=n_z,
        xlab='z (m)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_z')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.time
    pegleg = evts['{}_time'.format(p_reco)]
    retro = evts['{}_time'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-100, 200),
        nbins=n_te,
        xlab='reco time - true time (ns)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_time_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(9.4e3, 9.925e3),
        nbins=n_t,
        xlab='time (ns)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_time')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = np.arccos(evts.coszen)
    pegleg = evts['{}_track_zenith'.format(p_reco)]
    retro = evts['{}_track_zenith'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-.75, .75),
        nbins=n_zenerr,
        xlab='reco zenith - true zenith (rad)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_track_zenith_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(0, np.pi),
        nbins=n_zen,
        xlab='track zenith (rad)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_zenith')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.track_azimuth
    pegleg = evts['{}_track_azimuth'.format(p_reco)]
    retro = evts['{}_track_azimuth'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=(pegleg - truth + np.pi) % (2*np.pi) - np.pi,
        retro=(retro - truth + np.pi) % (2*np.pi) - np.pi,
        xlim=(-np.pi/8, np.pi/8),
        nbins=n_azerr,
        xlab='reco track azimuth - true track azimuth (rad)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_track_azimuth_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(0, 2*np.pi),
        nbins=n_az,
        xlab='track azimuth (rad)',
    )
    #y0, y1 = ax.get_ylim()
    #ax.set_ylim(y1*0.7, y1*.98)
    ax.legend(loc='lower right')
    if outdir is not None:
        fbp = join(outdir, 'dist_track_azimuth')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    p_err = evts['{}_track_angle_error'.format(p_reco)]
    r_err = evts['{}_track_angle_error'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=p_err,
        retro=r_err,
        xlim=(0, np.pi),
        nbins=n_ang,
        xlab=r'track angle error (rad)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_track_angle_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.cascade_energy
    pegleg = evts['{}_cascade_energy'.format(p_reco)]
    retro = evts['{}_cascade_energy'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-20, 20),
        nbins=n_cscdenerr,
        xlab='reco cascade energy - true cascade energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_cascade_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(0.0, 20),
        nbins=n_cscden,
        logx=False,
        xlab='cascade energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_cascade_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.track_energy
    pegleg = evts['{}_track_energy'.format(p_reco)]
    retro = evts['{}_track_energy'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-30, 40),
        nbins=n_trckenerr,
        xlab='reco track energy - true track energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_track_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(.5, 500),
        nbins=n_trcken,
        logx=True,
        xlab='track energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_track_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')

    truth = evts.energy
    pegleg = evts['{}_cascade_energy'.format(p_reco)] + evts['{}_track_energy'.format(p_reco)]
    retro = evts['{}_cascade_energy'.format(r_reco)]*2 + evts['{}_track_energy'.format(r_reco)]

    fig, ax = plot_comparison(
        pegleg=pegleg - truth,
        retro=retro - truth,
        xlim=(-40, 40),
        nbins=n_enerr,
        xlab=r'reco $\nu$ energy - true $\nu$ energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_energy_error')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
    fig, ax = plot_comparison(
        pegleg=pegleg,
        retro=retro,
        truth=truth,
        xlim=(1, 500),
        nbins=n_en,
        logx=True,
        xlab=r'$\nu$ energy (GeV)',
    )
    if outdir is not None:
        fbp = join(outdir, 'dist_neutrino_energy')
        fig.savefig(fbp + '.png', dpi=120)
        fig.savefig(fbp + '.pdf')
