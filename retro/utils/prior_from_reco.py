#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Use VBWKDE to characterize (with optional "extra" smoothing) the negative-error
distribution for a reconstructed parameter, then sample the resulting pdf for
use as a prior by Retro (see `retro.priors`).
"""

from __future__ import absolute_import, division, print_function

__all__ = ["orient_az_diff", "prior_from_reco"]

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2019 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from collections import Iterable, Mapping, OrderedDict
from copy import deepcopy
from os.path import abspath, dirname, expanduser, expandvars, isdir, join
from os import makedirs
import pickle
import sys

import matplotlib as mpl

mpl.use("agg", warn=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from six import string_types
from scipy import interpolate

from pisa.utils.vbwkde import vbwkde

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.weight_diff_tails import weight_diff_tails


def orient_az_diff(err):
    """Differences between two azimuthal angles wraps around the circle and
    should be centered about the subtractend (reference direction).

    Parameters
    ----------
    diff

    Returns
    -------
    reoriented_diff

    """
    return ((err + np.pi) % (2 * np.pi)) - np.pi


def prior_from_reco(
    reco_array,
    true_array,
    mc_version,
    reco,
    param,
    split_by_reco_param,
    num_split_bins,
    use_flux_weights,
    deweight_by_true_params,
    weight_tails,
    weight_tails_max_weight,
    mirror_about_finite_edges,
    num_density_samples,
    n_dct,
    n_addl_iter,
    n_plot_bins,
    outdir=None,
):
    """Use VBWKDE to characterize (with tunability to allow for extra
    smoothing) the negative-error distribution for a reconstructed parameter,
    then sample the resulting pdf for use as a prior by Retro (see
    `retro.priors`).

    If `outdir` is provided, resulting distributions and plots are saved to
    disk.

    Parameters
    ----------
    reco_array : pandas.DataFrame, struct array, or dict
        * DataFrame must have columns named as {reco}_{param}
        * dict must have reco names as keys & struct arrays as vals
        * struct arrays must contain a field matching `param` (or a field from
          which this can be derived)

    true_array : pandas.DataFrame or struct array
        Monte Carlo truth. Columns (for DataFrame) or fields (for struct array)
        must include the `param` being characterized, the `split_by_reco_param`
        (if one is specified), any `deweight_by_true_params`, and optionally
        "weight" if `use_flux_weights` is True.

    mc_version : str
        String describing the Monte Carlo used for the characterization

    reco : str
        Name of reconstruction being characterized

    param : str
        Name of parameter being characterized (e.g. "coszen", "x", "time", etc.)

    split_by_reco_param : str or None
        Name of param used to split the reconstructed data into parts, e.g. if
        behavior of the reco param is fundamentally different based upon the
        `split_by_reco_param`. Specify `None` to not split the data.

    num_split_bins : int or None
        If a `split_by_reco_param` param name is specified, the splits are this
        many bins (equally-sized in that param's domain)

    use_flux_weights : bool
        Whether to use `true_array["weight"]` for weighting the data

    deweight_by_true_params : str or iterable thereof, empty iterable, or None
        Specify one param (only implemented one for now) for deweighting
        (un-biasing) the data, or `None` to use the distribution of data as
        simulated.

    weight_tails : bool
        For coszen and zenith, which are finite-range parameters, tails are
        inherent when subtracting two such values from one another. Setting
        this to `True` attempts to correct for these tails.

    weight_tails_max_weight : float in [0, np.inf] or None
        Weighting the tails can produce very large weights where there is
        little MC; this parameter clips the weights from above by this value.
        Specify np.inf or None to effectively allow any weights. This param is
        ignored if `weight_tails` is `False`.

    mirror_about_finite_edges : bool
        VBWKDE can force edges to 0 for parameters that do not go to 0 at
        finite boundaries; mirroring the data about these can lessen edge
        effects. This parameter only affects finite boundaries (so e.g. "x" is
        not affected); "azimuth" and "coszen" are good candidates to try this
        out with.

    num_density_samples : int > 0
        Passed to `pisa.utils.vbwkde.vbwkde`

    n_dct : int > 0
        Passed to `pisa.utils.vbwkde.vbwkde`

    n_addl_iter : int >= 0
        Passed to `pisa.utils.vbwkde.vbwkde`

    n_plot_bins : int > 0
        Number of histogram bins to use (only for plots, not for deriving the
        prior)

    outdir : str, optional
        If provided, results and plots are stored in this directory

    Returns
    -------
    neg_err_dists : OrderedDict
    axes : 2D array of matplotlib.Axis

    """
    # `reco_array`

    if isinstance(reco_array, string_types):
        reco_array = pickle.load(open(reco_array, "rb"))

    if isinstance(reco_array, pd.DataFrame):
        pnames = set([param])
        if split_by_reco_param is not None:
            pnames.add(split_by_reco_param)
        pnames = sorted(pnames)

        new_df = pd.DataFrame()
        for pname in pnames:
            field = "{}_{}".format(reco, pname)
            if field in reco_array:
                vals = reco_array[field]
            elif pname == "coszen":
                vals = np.cos(reco_array["{}_zenith".format(reco)].values)
            elif pname == "zenith":
                vals = np.arccos(reco_array["{}_coszen".format(reco)].values)
            else:
                raise ValueError(
                    'retrieving pname "{}" via field "{}" failed'.format(pname, field)
                )
            new_df[pname] = vals
        reco_array = np.array(new_df[pnames].to_records())[pnames]

    if isinstance(reco_array, Mapping):
        reco_array = reco_array[reco]

    # `true_array`

    if isinstance(true_array, string_types):
        true_array = pickle.load(open(true_array, "rb"))

    # `param`

    try:
        true_vals = true_array[param]
    except (KeyError, ValueError):
        if "zenith" in param:
            newp = param.replace("zenith", "coszen")
            true_vals = np.arccos(true_array[newp])
        elif "coszen" in param:
            newp = param.replace("coszen", "zenith")
            true_vals = np.cos(true_array[newp])
        else:
            raise
    if isinstance(true_vals, pd.Series):
        true_vals = true_vals.values

    if param in reco_array.dtype.names:
        reco_vals = reco_array[param]
    elif param == "coszen":
        reco_vals = np.cos(reco_array["zenith"])
    elif param == "zenith":
        reco_vals = np.arccos(reco_array["coszen"])
    else:
        raise ValueError(
            'param "{}" not accessible in reco "{}" with dtype.names {}'.format(
                param, reco, reco_vals.dtype.names
            )
        )

    # `split_by_reco_param`

    if split_by_reco_param is None:
        assert num_split_bins in (0, 1, None, np.nan)
        num_split_bins = 1
        split_bin_edges = None
        split_by_vals = None
    else:
        if split_by_reco_param in reco_array.dtype.names:
            split_by_vals = reco_array[split_by_reco_param]
        elif split_by_reco_param == "coszen":
            split_by_vals = np.cos(reco_array["zenith"])
        elif split_by_reco_param == "zenith":
            split_by_vals = np.arccos(reco_array["coszen"])
        else:
            raise ValueError(
                'split_by_reco_param "{}" not in reco "{}" with dtype.names {}'.format(
                    split_by_reco_param, reco, reco_vals.dtype.names
                )
            )
        if split_by_reco_param == "coszen":
            split_bin_edges = np.linspace(-1, 1, num_split_bins + 1)
        elif split_by_reco_param == "zenith":
            split_bin_edges = np.linspace(0, np.pi, num_split_bins + 1)
        else:
            raise NotImplementedError(str(split_by_reco_param))

    # `use_flux_weights`

    assert isinstance(use_flux_weights, bool)
    weights = true_array["weight"]
    if isinstance(weights, pd.Series):
        weights = weights.values
    if not use_flux_weights:
        weights = np.ones_like(weights)

    # `deweight_by_true_params`

    if isinstance(deweight_by_true_params, string_types):
        deweight_by_true_params = [deweight_by_true_params]

    if isinstance(deweight_by_true_params, Iterable):
        deweight_by_true_params = list(deweight_by_true_params)

    if not deweight_by_true_params:
        deweight_by_true_params = None

    if deweight_by_true_params is not None:
        if len(deweight_by_true_params) > 1:
            raise NotImplementedError(
                "`deweight_by_true_params` = {}".format(deweight_by_true_params)
            )

        dwtp = true_array[deweight_by_true_params[0]]
        if isinstance(dwtp, pd.Series):
            dwtp = dwtp.values

        hist_vals, hist_edges = np.histogram(dwtp, bins=1000, weights=weights)
        bin_labels = np.digitize(dwtp, bins=hist_edges)
        new_weights = deepcopy(weights)
        for bnum, hist_val in enumerate(hist_vals):
            new_weights[bin_labels == (bnum + 1)] /= hist_val
        weights = new_weights

    # `weight_tails`, `weight_tails_max_weight`

    assert isinstance(weight_tails, bool)
    if weight_tails:
        if param not in ("coszen", "zenith"):
            raise ValueError("Weighting tails only makes sense for zenith or coszen")
        assert weight_tails_max_weight >= 0
    else:
        weight_tails_max_weight = np.nan

    # `mirror_about_finite_edges`

    assert isinstance(mirror_about_finite_edges, bool)

    # VBWKDE params

    assert num_density_samples > 2
    assert n_dct > 2
    assert n_addl_iter >= 0

    # Plotting params

    assert n_plot_bins > 0

    # Saving results params

    outdir = expanduser(expandvars(outdir))
    if not isdir(outdir):
        makedirs(outdir, mode=0o750)

    # Create dict for storing results

    neg_err_dists = OrderedDict(
        [
            (
                "metadata",
                OrderedDict(
                    [
                        ("mc_version", mc_version),
                        ("reco", reco),
                        ("param", param),
                        ("split_by_reco_param", split_by_reco_param),
                        ("num_split_bins", num_split_bins),
                        ("use_flux_weights", use_flux_weights),
                        ("deweight_by_true_params", deweight_by_true_params),
                        ("weight_tails", weight_tails),
                        ("weight_tails_max_weight", weight_tails_max_weight),
                        ("mirror_about_finite_edges", mirror_about_finite_edges),
                        ("num_density_samples", num_density_samples),
                        ("n_dct", n_dct),
                        ("n_addl_iter", n_addl_iter),
                    ]
                ),
            ),
            ("dists", OrderedDict()),
        ]
    )

    # Define max possible ranges for the param (assumed to apply equally to
    # reco and true in below logic; would need to revisit for e.g. energy)

    if param == "coszen":
        range_lower, range_upper = -1, 1
    elif param == "zenith":
        range_lower, range_upper = 0, np.pi
    elif param == "azimuth":
        range_lower, range_upper = 0, 2 * np.pi
    elif param in ("time", "x", "y", "z"):
        range_lower, range_upper = -np.inf, np.inf
    else:
        raise NotImplementedError(param)

    # Setup plots, calculate number of subplots in x and y

    nx = int(np.ceil(np.sqrt(num_split_bins)))
    ny = int(np.ceil(num_split_bins / nx))
    width = 5 * nx  # inches
    height = 4 * ny  # inches
    fig, axes = plt.subplots(ny, nx, figsize=(width, height), dpi=120, squeeze=False)
    axiter = iter(axes.flat)

    # Do the work

    for i in range(num_split_bins):
        ax = next(axiter)

        # Select only data inside the split bin (all data if no splits)

        if split_by_reco_param is None:
            split_bin_lower = range_lower
            split_bin_upper = range_upper
            mask = np.isfinite(reco_vals)
        else:
            split_bin_lower = split_bin_edges[i]
            split_bin_upper = split_bin_edges[i + 1]
            mask = (
                (split_by_vals >= split_bin_lower)
                & (split_by_vals <= split_bin_upper)
                & np.isfinite(split_by_vals)
                & np.isfinite(reco_vals)
            )
        this_reco_vals = reco_vals[mask]
        this_true_vals = true_vals[mask]
        this_weights = weights[mask]

        # Compute negative of error, accounting for azimuth wraparound

        neg_err = this_true_vals - this_reco_vals
        if param == "azimuth":
            # az negative-error is 0 at reco-azimuth and ranges from -pi to pi
            neg_err = orient_az_diff(neg_err)

        # Compute limits of reco param values given the split bin edges

        if param == "zenith" and split_by_reco_param == "coszen":
            inbin_lower = np.arccos(split_bin_upper)
            inbin_upper = np.arccos(split_bin_lower)
        elif param == "coszen" and split_by_reco_param == "zenith":
            inbin_lower = np.cos(split_bin_upper)
            inbin_upper = np.cos(split_bin_lower)
        elif param == split_by_reco_param:
            inbin_lower = split_bin_lower
            inbin_upper = split_bin_upper
        else:  # split_by_reco_param is orthogonal to param
            inbin_lower = range_lower
            inbin_upper = range_upper

        # Maximum ranges possible for true and reco:
        #   true ∈ [range_lower, range_upper]
        #   reco ∈ [inbin_lower, inbin_upper]
        # therefore the difference:
        #   (true - reco) ∈ [range_lower - inbin_upper, range_upper - inbin_lower]

        if param == "azimuth":
            theor_neg_err_lower = -np.pi
            theor_neg_err_upper = +np.pi
        else:
            theor_neg_err_lower = range_lower - inbin_upper
            theor_neg_err_upper = range_upper - inbin_lower

        # Optionally remove bias in distributions that results from subtracting
        # two finite-range numbers from one another

        if weight_tails:
            this_weights, dlims = weight_diff_tails(
                diff=neg_err,
                inbin_lower=inbin_lower,
                inbin_upper=inbin_upper,
                range_lower=range_lower,
                range_upper=range_upper,
                weights=this_weights,
                max_weight=weight_tails_max_weight,
            )
            # double checks
            assert theor_neg_err_lower == dlims[0]
            assert theor_neg_err_upper == dlims[1]

        # Mirror error about left and/or right edges (for finite-range params);
        # this reduces edge effects of KDE

        neg_err_to_cat = [neg_err]
        if np.isfinite(theor_neg_err_lower) and mirror_about_finite_edges:
            neg_err_to_cat.append(2 * theor_neg_err_lower - neg_err)
        if np.isfinite(theor_neg_err_upper) and mirror_about_finite_edges:
            neg_err_to_cat.append(2 * theor_neg_err_upper - neg_err)
        num_copies = len(neg_err_to_cat)
        cat_neg_err = np.concatenate(neg_err_to_cat)
        cat_weights = np.concatenate([this_weights] * num_copies)

        # For finite-range limits, evaluate kernels to full extent; for
        # infinite-range params, 25% below and above error range actually seen
        # in data

        actual_neg_err_lower = np.nanmin(neg_err)
        actual_neg_err_upper = np.nanmax(neg_err)
        actual_neg_err_range = np.abs(actual_neg_err_upper - actual_neg_err_lower)

        if np.isfinite(theor_neg_err_lower):
            eval_lower = theor_neg_err_lower
        else:
            eval_lower = actual_neg_err_lower - actual_neg_err_range / 10

        if np.isfinite(theor_neg_err_upper):
            eval_upper = theor_neg_err_upper
        else:
            eval_upper = actual_neg_err_upper + actual_neg_err_range / 10

        x = np.linspace(eval_lower, eval_upper, num_density_samples)

        _, _, dens = vbwkde(
            data=cat_neg_err,
            weights=cat_weights,
            n_dct=n_dct,
            n_addl_iter=n_addl_iter,
            evaluate_at=x,
            evaluate_dens=True,
        )

        # We made range N-times as wide as originally; take middle third => norm = N
        dens *= num_copies

        # Ensure pdf (trapezoidal-rule) integral is 1
        pdf = dens / np.trapz(y=dens, x=x)

        # cdf is integral of pdf; note use of `trapz` instead of `cumsum`
        cdf = np.array([np.trapz(x=x[:ii], y=pdf[:ii]) for ii in range(1, len(x) + 1)])

        # cdf must start at 0
        cdf -= cdf[0]

        # cdf must end at 1
        cdf /= cdf[-1]

        # Record result to dict
        key = (split_bin_lower, split_bin_upper) if split_by_reco_param else None
        neg_err_dists["dists"][key] = OrderedDict(
            [("x", x), ("pdf", pdf), ("cdf", cdf)]
        )

        # _, interp = generate_lerp(
        #    x=cdf,
        #    y=x,
        #    low_behavior='error',
        #    high_behavior='error',
        # )
        interp = interpolate.UnivariateSpline(x=cdf, y=x, ext="raise", s=0)
        samps = interp(np.linspace(0, 1, int(1e6)))

        plot_bins = np.linspace(eval_lower, eval_upper, n_plot_bins + 1)
        samp_bins = np.linspace(eval_lower, eval_upper, n_plot_bins * 1 + 1)
        if use_flux_weights or deweight_by_true_params or weight_tails:
            ax.hist(
                neg_err,
                weights=None,
                bins=plot_bins,
                density=True,
                histtype="step",
                color="C0",
                linewidth=1,
                label="unweighted",
            )

        label_parts = []
        if use_flux_weights:
            label_parts.append("flux-weighted")
        if deweight_by_true_params:
            label_parts.append(
                "deweighted by true {}".format(", ".join(deweight_by_true_params))
            )
        if weight_tails:
            label_parts.append("tail-comp")

        if label_parts:
            label = ", ".join(label_parts)
        else:
            label = "unweighted"

        ax.hist(
            neg_err,
            weights=this_weights,
            bins=plot_bins,
            density=True,
            histtype="step",
            color="C2",
            linewidth=1,
            label=label,
        )
        ax.hist(
            samps,
            weights=None,
            bins=samp_bins,
            density=True,
            histtype="step",
            color="C3",
            linewidth=1,
            label="histo of generated samples",
        )
        ax.plot(x, dens, lw=2, ls="--", zorder=+10, color="C1", label="vbwkde")

        ax.set_xlim(eval_lower, eval_upper)
        ax.set_yticklabels([0])

        if split_by_reco_param:
            ax.set_title(
                r"Reco {} $\in$ [{:.3f}, {:.3f}]".format(
                    split_by_reco_param, split_bin_lower, split_bin_upper
                )
            )

        if i == 0:
            ax.legend(loc="best", frameon=False, fontsize=6)

    # Turn off any excess axes in the grid
    for ax in axiter:
        ax.axis("off")

    suptitle = "{} {}: true $-$ reco".format(reco, param)
    fname = "{}_{}_neg_error".format(reco, param)
    fontsize = 8
    if split_by_reco_param:
        suptitle += " split by reco {}".format(split_by_reco_param)
        fname += "_splitby_reco_{}_{:d}".format(split_by_reco_param, num_split_bins)
        fontsize = 16

    fig.suptitle(suptitle, fontsize=fontsize)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if outdir is not None:
        basefpath = join(outdir, fname)
        fig.savefig(basefpath + ".png", dpi=120)
        fig.savefig(basefpath + ".pdf")
        print('saved plots to "{}.{{png, pdf}}"'.format(basefpath))

        outfpath = basefpath + ".pkl"
        pickle.dump(
            neg_err_dists, open(outfpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )
        print('saved neg err dist(s) to "{}"'.format(outfpath))

    return neg_err_dists, axes
