#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Plot resolutions and distributions for reconstructions.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["xlate_zen", "xlate_cascade", "xlate", "get_pval", "get_nu_flavints_mask", "get_nu_flavints_mask", "get_nu_flavints", "plotit"]

from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, dirname, join
import sys

import numpy as np
from six import string_types

import matplotlib as mpl
#mpl.use("agg")
import matplotlib.pyplot as plt

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.priors import get_point_estimate
from retro.utils.stats import weighted_percentile


def xlate_zen(pname, other_pnames):
    if "coszen" in pname:
        newp = pname.replace("coszen", "zenith")
        xform = np.cos
    elif "zenith" in pname:
        newp = pname.replace("zenith", "coszen")
        xform = np.arccos
    else:
        return None
    if newp in other_pnames:
        return None
    return newp, xform


def xlate_cascade(pname, other_pnames):
    orig_pname = pname
    if "cascade" in pname and not "total_cascade" in pname:
        pname = pname.replace("cascade", "total_cascade")
        xform = None
    if "energy" in pname and not "em_equiv_energy" in pname:
        pname = pname.replace("energy", "em_equiv_energy")
        xform = None
    if pname == orig_pname or pname in other_pnames:
        return None
    return pname, xform


def xlate(pname_xforms, index=None):
    if isinstance(pname_xforms, string_types):
        pname_xforms = [(pname_xforms, None)]
    elif isinstance(pname_xforms, tuple):
        pname_xforms = [pname_xforms]
    if index is None:
        index = len(pname_xforms) - 1

    for pname, xforms in deepcopy(pname_xforms[index:]):
        for xlator in [xlate_zen, xlate_cascade]:
            xlation = xlator(pname, other_pnames=[px[0] for px in pname_xforms])
            if xlation is not None:
                pname_xforms.append(xlation)
        index += 1

    if index < len(pname_xforms) - 1:
        pname_xforms += xlate(pname_xforms, index=index)

    return pname_xforms


def get_pval(array, param):
    array = np.array(array)
    if param in array.dtype.names:
        return get_point_estimate(array[param], estimator="median", expect_scalar=False)

    pnames_xforms = xlate(param)
    for pname, xform in pnames_xforms:
        if pname in array.dtype.names:
            val = get_point_estimate(array[pname], estimator="median", expect_scalar=False)
            if xform is not None:
                print("requested '{}' but retrieving {}({}) instead".format(param, xform, pname))
                val = xform(val)
            return val

    raise ValueError(
        "Couldn't find '{}' or a variant {} in `array` with dtype.names = {}"
        .format(param, [px[0] for px in pnames_xforms], array.dtype.names)
    )


def get_nu_flavints_mask(array, flavints):
    if isinstance(flavints, string_types):
        flavints = [flavints]
    flavints = list(deepcopy(flavints))
    if "nuall_nc" in flavints:
        flavints.remove("nuall_nc")
        flavints.extend(["nu{}_nc".format(n) for n in ["e", "mu", "tau"]])
    if "nuallbar_nc" in flavints:
        flavints.remove("nuallbar_nc")
        flavints.extend(["nu{}bar_nc".format(n) for n in ["e", "mu", "tau"]])
    if "nuall_cc" in flavints:
        flavints.remove("nuall_cc")
        flavints.extend(["nu{}_cc".format(n) for n in ["e", "mu", "tau"]])
    if "nuallbar_cc" in flavints:
        flavints.remove("nuallbar_cc")
        flavints.extend(["nu{}bar_cc".format(n) for n in ["e", "mu", "tau"]])

    pdgs = array["truth"]["pdg_encoding"]
    int_types = array["truth"]["InteractionType"]

    mask = np.zeros(array.shape, dtype=np.bool)
    if "nue_cc" in flavints:
        mask |= (pdgs == 12) & (int_types == 1)
        flavints.remove("nue_cc")
    if "nuebar_cc" in flavints:
        mask |= (pdgs == -12) & (int_types == 1)
        flavints.remove("nuebar_cc")
    if "numu_cc" in flavints:
        mask |= (pdgs == 14) & (int_types == 1)
        flavints.remove("numu_cc")
    if "numubar_cc" in flavints:
        mask |= (pdgs == -14) & (int_types == 1)
        flavints.remove("numubar_cc")
    if "nutau_cc" in flavints:
        mask |= (pdgs == 16) & (int_types == 1)
        flavints.remove("nutau_cc")
    if "nutaubar_cc" in flavints:
        mask |= (pdgs == -16) & (int_types == 1)
        flavints.remove("nutaubar_cc")

    if "nue_nc" in flavints:
        mask |= (pdgs == 12) & (int_types == 2)
        flavints.remove("nue_nc")
    if "nuebar_nc" in flavints:
        mask |= (pdgs == -12) & (int_types == 2)
        flavints.remove("nuebar_nc")
    if "numu_nc" in flavints:
        mask |= (pdgs == 14) & (int_types == 2)
        flavints.remove("numu_nc")
    if "numubar_nc" in flavints:
        mask |= (pdgs == -14) & (int_types == 2)
        flavints.remove("numubar_nc")
    if "nutau_nc" in flavints:
        mask |= (pdgs == 16) & (int_types == 2)
        flavints.remove("nutau_nc")
    if "nutaubar_nc" in flavints:
        mask |= (pdgs == -16) & (int_types == 2)
        flavints.remove("nutaubar_nc")

    assert len(flavints) == 0, str(flavints)

    return mask


def get_nu_flavints(array, flavints):
    mask = get_nu_flavints_mask(array=array, flavints=flavints)
    return array[mask]


#reco_names = ['retro_crs_prefit'] #, 'retro_mn8d']#[2:]
#info = OrderedDict()
#
#for reco_name in reco_names:
#    events, truth, recos = extract_reco(
#        reco=reco_name,
#        #recodir='~/retro_results/recos/2019-04-01.oscnext_l5_v1.code_updates_crsprifix',
#        recodir=destdir,
#    )
#    assert len(recos) == len(truth)
#    mask = recos["fit_status"] == FitStatus.OK
#    if "run_time" in recos.dtype.names:
#        rt_finite = np.isfinite(recos["run_time"])
#        x_finite = np.isfinite(recos["x"]["median"])
#        if not np.all(rt_finite):
#            assert np.all(rt_finite == x_finite)
#            print("fit_status is OK but x not finite for {} of {} events".format(np.count_nonzero(np.logical_not(x_finite) & mask), mask.size))
#            mask &= rt_finite
#    num_ok = np.count_nonzero(mask)
#    print(reco_name, "--", "num_ok:", num_ok, "percent ok:", 100*num_ok/len(recos))
#    info[reco_name] = dict(
#        events=events[mask],
#        truth=truth[mask],
#        recos=recos[mask],
#    )
#
#
#ebin_edges = np.logspace(0, 3, 11)
#all_to_plot = ['retro_crs_prefit', 'retro_mn8d', 'total']
#
#recodict = info
#
#
#for to_plot in all_to_plot:
#
#    run_times = []
#    num_events = []
#    num_llh = []
#    for e0, e1 in zip(ebin_edges[:-1], ebin_edges[1:]):
#        run_time = 0
#        num = 0
#        nllh = 0
#        for reco in recos_to_plot:
#            if reco not in recodict:
#                break
#            rinfo = recodict[reco]
#            true_energy = rinfo["truth"]["energy"]
#            mask = (true_energy >= e0) & (true_energy <= e1)
#            this_recos = rinfo["recos"][mask]
#            run_time += np.nanmean(this_recos["run_time"])
#            nllh += np.nanmean(this_recos["num_llh"])
#        run_times.append(run_time)
#        num_llh.append(nllh)
#        num_events.append(np.count_nonzero(mask))
#    run_times = np.array(run_times)
#    num_llh = np.array(num_llh)
#    num_events = np.array(num_events)
#
#    meanval = np.sum(run_times * num_events) / np.sum(num_events)
#
#    fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
#    axit = iter(axes.flat)
#
#    ax = next(axit)
#    ax.step(ebin_edges, [run_times[0]] + run_times.tolist())
#    if meanval < 60:
#        disp_mean = "{:.1f} sec".format(meanval)
#    else:
#        disp_mean = "{:.2f} min".format(meanval / 60)
#    ax.plot(ebin_edges, [meanval]*len(ebin_edges), '--', label="mean = {}".format(disp_mean))
#    ax.set_ylim(0, ax.get_ylim()[1])
#    ax.set_xscale('log')
#    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
#    ax.set_xlabel('Energy (GeV)')
#    ax.set_ylabel('Mean reco time (s)')
#    ax.legend(loc='best')
#    #ax.set_title('Reconstruction time')
#    # fbase = join(outdir, 'reco_time')
#    # fig.savefig(fbase + '.png', dpi=120)
#    # fig.savefig(fbase + '.pdf')
#
#    ax = next(axit)
#    ax.step(ebin_edges, [num_llh[0]] + num_llh.tolist())
#    #ax.plot(ebin_edges, [meanval]*len(ebin_edges), '--', label="mean = {:.1f} min".format(meanval))
#    #ax.set_ylim(0, ax.get_ylim()[1])
#    ax.set_xscale('log')
#    ax.set_xlim(ebin_edges[0], ebin_edges[-1])
#    ax.set_xlabel('Energy (GeV)')
#    ax.set_ylabel('Num LLH')
#    ax.legend(loc='best')
#    #ax.set_title('Reconstruction time')
#
#    fig.suptitle(to_plot, fontsize=16)
#    fig.tight_layout(rect=(0, 0, 1, 0.95))
#
#    outfbase = join(outdir, "{}_timings".format(to_plot))
#    fig.savefig(outfbase + ".pdf")
#    fig.savefig(outfbase + ".png", dpi=120)


# ================================================================= #


def plotit():
    plot_recos = [
        #"LineFit_DC",
        #"L5_SPEFit11",
        "retro_crs_prefit",
        "retro_mn8d",
        "Pegleg_Fit_MN",
    ][::-1]
    plot_params = [
        "x",
        "y",
        "z",
        "time",
        "azimuth",
        "zenith",
        "coszen",
        "energy",
        "track_energy",
        "track_zenith",
        "track_coszen",
        "track_azimuth",
    ]

    toplot = common
    use_weights = False
    n_bins = 71
    n_ebins = 10
    ebin_edges = np.logspace(0, 3, n_ebins + 1)

    lower_disp_pctile, upper_disp_pctile = 2.5, 97.5
    lower_pct, upper_pct = 25, 75
    iq_pct = upper_pct - lower_pct

    longest_recolen = 0
    for reco in plot_recos:
        longest_recolen = max(longest_recolen, len(reco))
        recos = toplot[reco]["recos"]
        truth = toplot[reco]["truth"]
        if recos.dtype.names and "run_time" in recos.dtype.names:
            print(
                "Mean run time, {}: {:.1f} s; mean energy = {:.1f}"
                .format(reco, recos["run_time"].mean(), truth["energy"].mean())
            )


    #nx = int(np.ceil(np.round(np.sqrt(len(plot_params)*2)/2)*2))
    nx = 4
    ny = int(np.ceil(len(plot_params)*4 / nx))
    f = 1.5
    fig, axes = plt.subplots(ny, nx, figsize=(4*f*nx, 3*f*ny), dpi=120, squeeze=False)
    axit = iter(axes.flat)

    for param in plot_params:
        err_lower, err_upper = np.inf, -np.inf
        lower, upper = np.inf, -np.inf
        stuff = OrderedDict()

        for reco in plot_recos:
            rinfo = toplot[reco]
            try:
                if param == "energy":
                    track_energy = get_pval(rinfo["recos"], "track_energy")
                    cascade_energy = get_pval(rinfo["recos"], "cascade_energy")
                    recos = track_energy + 2.*cascade_energy
                else:
                    recos = get_pval(rinfo["recos"], param)
                truth = get_pval(rinfo["truth"], param)
            except:
                print('exception', param, reco)
                continue
            recos = get_point_estimate(recos, estimator="median", expect_scalar=False)
            if not np.all(np.isfinite(recos)):
                n_nonfinite = np.count_nonzero(np.logical_not(np.isfinite(recos)))
                print('not all finite: {}, {}: {} / {} not finite'.format(param, reco, n_nonfinite, recos.size))
                continue
            weight = rinfo['truth']['weight']
            if "azimuth" in param:
                error = (recos - truth + np.pi) % (2*np.pi) - np.pi
            elif "energy" in param:
                error = recos / truth - 1
            else:
                error = recos - truth

            stuff[reco] = (recos, truth, error, weight)
            if use_weights:
                lower_, upper_ = weighted_percentile(error[np.isfinite(error)], [lower_disp_pctile, upper_disp_pctile], weight)
            else:
                lower_, upper_ = np.percentile(error[np.isfinite(error)], [lower_disp_pctile, upper_disp_pctile])
            err_lower = np.nanmin([lower_, err_lower])
            err_upper = np.nanmax([upper_, err_upper])

            for array in (truth, recos):
                mask = np.isfinite(array)
                if "energy" in param:
                    mask &= array > 0
                if use_weights:
                    lower_, upper_ = weighted_percentile(array[mask], [lower_disp_pctile, upper_disp_pctile], weight)
                else:
                    lower_, upper_ = np.percentile(array[mask], [lower_disp_pctile, upper_disp_pctile])
                lower = np.nanmin([lower_, lower])
                upper = np.nanmax([upper_, upper])

        # -- Plot raw distributions -- #

        ax = next(axit)
        if "energy" in param:
            bins = np.logspace(np.log10(lower), np.log10(upper), n_bins)
            xscale = "log"
        else:
            bins = np.linspace(lower, upper, n_bins)
            xscale = "linear"
        mask = np.isfinite(truth)
        nf = np.count_nonzero(mask)
        if nf != mask.size:
            print(param, "truth", nf, mask.size)

        for reco, (recos, truth, error, weight) in stuff.items():
            mask = np.isfinite(recos)
            nf = np.count_nonzero(mask)
            if nf != mask.size:
                print(reco, recos, nf, mask.size)
            if use_weights:
                pc_lower, median, pc_upper = weighted_percentile(recos[mask], [lower_pct, 50, upper_pct], weight)
            else:
                pc_lower, median, pc_upper = np.percentile(recos[mask], [lower_pct, 50, upper_pct])
            iq = pc_upper - pc_lower
            try:
                _, _, out = ax.hist(
                    recos[mask],
                    weights=weight[mask] if use_weights else None,
                    bins=bins,
                    #label="{}".format(reco.rjust(longest_recolen)),
                    histtype='step',
                    lw=1,
                )
            except:
                print(reco, param, np.all(np.isfinite(recos)), np.nanmin(recos), np.nanmax(recos), lower, upper)
                raise

        recos, truth, error, weight = stuff.values()[0]
        pc_lower, median, pc_upper = np.percentile(truth[mask], [lower_pct, 50, upper_pct])
        iq = pc_upper - pc_lower
        _, _, out = ax.hist(
            truth[mask],
            weights=weight[mask] if use_weights else None,
            bins=bins,
            label="MC truth", #"{}".format("truth".rjust(longest_recolen)),
            histtype='step',
            lw=1.5,
            edgecolor='k',
            zorder=-10,
        )

        ax.set_xlim(lower, upper)
        ax.set_xscale(xscale)
        leg = ax.legend(loc="lower center", frameon=False)
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), family='monospace')
        ax.set_yticks([])
        ax.set_title("{} distribution, all E".format(param))

        # -- Plot errors across all events -- #

        ax = next(axit)
        bins = np.linspace(err_lower, err_upper, n_bins)
        for reco, (recos, truth, error, weight) in stuff.items():
            mask = np.isfinite(error)
            nf = np.count_nonzero(mask)
            if nf != mask.size:
                print(param, reco, "error", nf, mask.size)

            if use_weights:
                pc_lower, median, pc_upper = weighted_percentile(error[mask], [lower_pct, 50, upper_pct], weight)
            else:
                pc_lower, median, pc_upper = np.percentile(error[mask], [lower_pct, 50, upper_pct])

            iq = pc_upper - pc_lower
            try:
                _, _, out = ax.hist(
                    error[mask],
                    weights=weight[mask] if use_weights else None,
                    bins=bins,
                    label="{} {:6.2f} {:6.2f}".format(reco.rjust(longest_recolen), median, iq),
                    histtype='step',
                    lw=1,
                )
            except:
                print(reco, param, np.all(np.isfinite(error)), np.nanmin(error), np.nanmax(error), lower, upper)
                raise
        ax.set_xlim(err_lower, err_upper)
        ax.set_ylim(0, ax.get_ylim()[1]*1.3)
        leg = ax.legend(
            loc="upper right",
            title=(
                "{} {} {}"
                .format(" "*longest_recolen, "median", "IQ {:2d}%".format(iq_pct))
            ),
            markerfirst=False,
            frameon=False,
            framealpha=0.2,
            prop=dict(family='monospace'),
        )
        leg._legend_box.align = "left"
        plt.setp(leg.get_title(), family='monospace')
        ax.set_yticks([])
        if "energy" in param:
            title = "fract {} error, all E".format(param)
        else:
            title = "{} error, all E".format(param)
        ax.set_title(title)


        # -- Plot errors vs. true energy -- #

        ax = next(axit)
        colors = ['C{}'.format(i) for i in range(8)]
        colors_iter = iter(colors)
        for reco, (recos, truth, error, weight) in stuff.items():
            mask = np.isfinite(error)
            true_en = get_pval(toplot[reco]['truth'], "energy")
            idxs = np.digitize(true_en, ebin_edges) - 1
            pc_l = []
            pc_u = []
            for idx in range(n_ebins):
                bin_error = error[(idxs == idx) & mask]
                if use_weights:
                    pc_l_, med_, pc_u_ = weighted_percentile(bin_error, [lower_pct, 50, upper_pct], weight)
                else:
                    pc_l_, med_, pc_u_ = np.percentile(bin_error, [lower_pct, 50, upper_pct])
                pc_l.append(pc_l_)
                pc_u.append(pc_u_)
            color = next(colors_iter)
            ax.fill_between(
                x=ebin_edges,
                y1=[pc_l[0]]+pc_l,
                y2=[pc_u[0]]+pc_u,
                interpolate=False,
                step="post",
                facecolor='none',
                edgecolor=color,
            )
            ax.set_xscale('log')
            ax.set_xlim(ebin_edges[0], ebin_edges[-1])
            if "energy" in param:
                title = "fractional {} error [{}, {}]% vs. true E".format(param, lower_pct, upper_pct)
            else:
                title = "{} error [{}, {}]% vs true E".format(param, lower_pct, upper_pct)
            ax.set_title(title)

        # -- Plot error WIDTHS vs. true energy -- #

        ax = next(axit)
        colors_iter = iter(colors)
        for reco, (recos, truth, error, weight) in stuff.items():
            mask = np.isfinite(error)
            true_en = get_pval(toplot[reco]['truth'], "energy")
            idxs = np.digitize(true_en, ebin_edges) - 1
            widths = []
            for idx in range(n_ebins):
                bin_error = error[(idxs == idx) & mask]
                if use_weights:
                    pc_l_, med_, pc_u_ = weighted_percentile(bin_error, [lower_pct, 50, upper_pct], weight)
                else:
                    pc_l_, med_, pc_u_ = np.percentile(bin_error, [lower_pct, 50, upper_pct])
                widths.append(pc_u_ - pc_l_)
            color = next(colors_iter)
            ax.step(
                x=ebin_edges,
                y=[widths[0]]+widths,
                #facecolor='none',
                color=color,
            )
            ax.set_xscale('log')
            ax.set_xlim(ebin_edges[0], ebin_edges[-1])
            if "energy" in param:
                title = "fractional {} error IQ{}% width vs true E".format(param, iq_pct)
            else:
                title = "{} error IQ{}% width vs true E".format(param, iq_pct)
            ax.set_title(title)


    for ax in axit:
        ax.remove()

    #fig.tight_layout(h_pad=1, w_pad=0.01)
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.draw()
    plt.show()

    fbasename = join(outdir, "distributions")
    fig.savefig(fbasename + ".pdf")
    fig.savefig(fbasename + ".png", dpi=120)
