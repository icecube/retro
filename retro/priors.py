#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Prior definition generator and prior funcion generator to use for multinest
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "PRI_UNIFORM",
    "PRI_LOG_UNIFORM",
    "PRI_LOG_NORMAL",
    "PRI_COSINE",
    "PRI_GAUSSIAN",
    "PRI_INTERP",
    "PRI_AZ_INTERP",
    "PRI_TIME_RANGE",
    "PRI_SPEFIT2",
    "PRI_SPEFIT2TIGHT",
    "PRI_OSCNEXT_L5_V1_PREFIT",
    "PRI_OSCNEXT_L5_V1_CRS",
    "OSCNEXT_L5_V1_PRIORS",
    "get_point_estimate",
    "define_prior_from_prefit",
    "define_generic_prior",
    "get_prior_func",
]

__author__ = "J.L. Lanfranchi, P. Eller"
__license__ = """Copyright 2017 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, basename, dirname, join
import sys

import enum
import numpy as np
from scipy import interpolate, stats
from six import string_types

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == "__main__" and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import MissingOrInvalidPrefitError
from retro.const import TWO_PI
from retro.retro_types import FitStatus
from retro.utils.misc import LazyLoader


PRI_UNIFORM = "uniform"
PRI_LOG_UNIFORM = "log_uniform"
PRI_LOG_NORMAL = "log_normal"
PRI_COSINE = "cosine"
PRI_GAUSSIAN = "gaussian"
PRI_INTERP = "interp"
PRI_AZ_INTERP = "az_interp"
PRI_TIME_RANGE = "time_range"
PRI_SPEFIT2 = "spefit2"
"""From fits to DRAGON (GRECO?) i.e. pre-oscNext MC"""

PRI_SPEFIT2TIGHT = "spefit2tight"
"""From fits to DRAGON (GRECO?) i.e. pre-oscNext MC"""

PRI_OSCNEXT_L5_V1_PREFIT = "oscnext_l5_v1_prefit"
"""Priors from L5_SPEFit11 (and fallback to LineFit_DC) fits to oscNext level 5
(first version of processing, or v1) events. See
  retro/notebooks/plot_prior_reco_candidates.ipynb for the fitting process.
"""

PRI_OSCNEXT_L5_V1_CRS = "oscnext_l5_v1_crs"
"""Priors from CRS fits to oscNext level 5 (first version of processing, or v1)
events. See
  retro/notebooks/plot_prior_reco_candidates.ipynb for the fitting process.
"""


class Bound(enum.IntEnum):
    """Specify a boundary is absolute or relative"""

    ABS = 0
    REL = 1


EXT_TIGHT = dict(
    x=((-200, Bound.REL), (200, Bound.REL)),
    y=((-200, Bound.REL), (200, Bound.REL)),
    z=((-100, Bound.REL), (100, Bound.REL)),
    time=((-1000, Bound.REL), (1000, Bound.REL)),
)

EXT_MN = dict(
    x=((-300, Bound.REL), (300, Bound.REL)),
    y=((-300, Bound.REL), (300, Bound.REL)),
    z=((-400, Bound.REL), (400, Bound.REL)),
    time=((-1500, Bound.REL), (1500, Bound.REL)),
)

EXT_IC = dict(
    x=((-860, Bound.ABS), (870, Bound.ABS)),
    y=((-780, Bound.ABS), (770, Bound.ABS)),
    z=((-780, Bound.ABS), (790, Bound.ABS)),
)

EXT_DC = dict(
    x=((-150, Bound.ABS), (270, Bound.ABS)),
    y=((-210, Bound.ABS), (150, Bound.ABS)),
    z=((-770, Bound.ABS), (760, Bound.ABS)),
)

EXT_DC_SUBDUST = deepcopy(EXT_DC)
EXT_DC_SUBDUST["z"] = ((-610, Bound.ABS), (60, Bound.ABS))


PRISPEC_OSCNEXT_PREFIT_TIGHT = dict(
    x=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["x"]),
    y=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["y"]),
    z=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["z"]),
    time=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_TIGHT["time"]),
    azimuth=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
    zenith=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
)

PRISPEC_OSCNEXT_CRS = dict(
    x=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_MN["x"]),
    y=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_MN["y"]),
    z=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_MN["z"]),
    time=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT, extents=EXT_MN["time"]),
    azimuth=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
    zenith=dict(kind=PRI_OSCNEXT_L5_V1_PREFIT),
)

PRISPEC_OSCNEXT_CRS_MN = dict(
    x=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["x"]),
    y=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["y"]),
    z=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["z"]),
    time=dict(kind=PRI_OSCNEXT_L5_V1_CRS, extents=EXT_MN["time"]),
    #azimuth=dict(kind=PRI_OSCNEXT_L5_V1_CRS),
    #zenith=dict(kind=PRI_OSCNEXT_L5_V1_CRS),
)


OSCNEXT_L5_V1_PRIORS = OrderedDict()
for _dim in ("time", "x", "y", "z", "azimuth", "zenith", "coszen"):
    OSCNEXT_L5_V1_PRIORS[_dim] = OrderedDict()
    for _reco in ("L5_SPEFit11", "LineFit_DC", "retro_crs_prefit"):
        OSCNEXT_L5_V1_PRIORS[_dim][_reco] = LazyLoader(
            datasource=join(
                RETRO_DIR,
                "retro_data",
                "priors",
                "{reco}_{dim}_neg_error.pkl".format(reco=_reco, dim=_dim),
            )
        )


def get_point_estimate(val, estimator, expect_scalar=True):
    """Retrieve a scalar for a reconstructed value.

    Allows for simple scalar recos, or for "estimates from LLH/Params" where a
    number of values are returned using different estimation techniques.

    Parameters
    ----------
    val : scalar, numpy array of struct dtype, or Mapping
    estimator : str
        Not used if `val` is a scalar, otherwise used to get field from numpy
        struct array or item from a Mapping
    expect_scalar : bool

    Returns
    -------
    scalar_val

    """
    # Convert a recarray to a simple array with struct dtype; convert scalar to 0-d array
    valarray = np.array(val)

    # If struct dtype, extract the point estimator
    if valarray.dtype.names:
        valarray = np.array(valarray[estimator])

    if expect_scalar:
        assert valarray.size == 1

        # Since we've forced it to be an array and we don't know the exact
        # dimensionality, use `flat` iterator and extract the first element of the
        # array
        return next(valarray.flat)

    return valarray


def define_prior_from_prefit(
    dim_name, event, priors, candidate_recos, point_estimator, extents=None
):
    """Define a prior from pre-fit(s). Priors are defined by the interpolation
    of KDE'd negative-error distribution for the pre-fits, and "fallback" fits
    can be defined in case one or more fits failed."""
    if isinstance(candidate_recos, string_types):
        candidate_recos = [candidate_recos]

    reco = None
    for candidate_reco in candidate_recos:
        try:
            fit_status = event["recos"][candidate_reco]["fit_status"]
        except (KeyError, ValueError):
            fit_status = FitStatus.OK
        if fit_status == FitStatus.OK:
            reco = candidate_reco
            break

    if reco is None:
        raise MissingOrInvalidPrefitError(
            "Couldn't find a valid prefit reco from among {}".format(candidate_recos)
        )

    # Remove "track_*", etc prefixes
    for prefix in ("track", "cascade"):
        if dim_name.startswith(prefix):
            dim_name = dim_name[len(prefix) :].lstrip("_")
            break

    try:
        reco_val = get_point_estimate(
            event["recos"][reco][dim_name], estimator=point_estimator
        )
    except (KeyError, ValueError):
        if dim_name == "coszen":
            reco_val = np.cos(
                get_point_estimate(
                    event["recos"][reco]["zenith"], estimator=point_estimator
                )
            )
        elif dim_name == "zenith":
            reco_val = np.arccos(
                get_point_estimate(
                    event["recos"][reco]["coszen"], estimator=point_estimator
                )
            )
        else:
            print('No dim "{}" in reco "{}"'.format(dim_name, reco))
            raise

    if not np.isfinite(reco_val):
        raise MissingOrInvalidPrefitError(
            'dim_name "{}", reco "{}": reco val = {}'.format(dim_name, reco, reco_val)
        )

    prior_info = priors[dim_name][reco].data
    prior_sha256 = priors[dim_name][reco].sha256
    prior_fname = basename(priors[dim_name][reco].datasource)

    split_by_reco_param = prior_info["metadata"]["split_by_reco_param"]
    if split_by_reco_param is None:
        split_val = None
    else:
        if split_by_reco_param == "coszen":
            split_val = np.cos(event["recos"][reco]["zenith"])
        else:
            split_val = event["recos"][reco][split_by_reco_param]

        if not np.isfinite(split_val):
            raise MissingOrInvalidPrefitError(
                'Reco "{}", split val "{}" = {}'.format(
                    reco, split_by_reco_param, split_val
                )
            )

    pri = None
    for edges, pri_ in prior_info["dists"].items():
        if split_by_reco_param is None:
            pri = pri_
            break

        if edges[0] <= split_val <= edges[1]:
            pri = pri_
            break

    if pri is None:
        raise ValueError(
            '`split_by_reco_param` "{}" value={} outside binned ranges?: {}'.format(
                split_by_reco_param, split_val, prior_info["dists"].keys()
            )
        )

    xvals = pri["x"] + reco_val

    if extents is None:
        low = np.min(xvals)
        high = np.max(xvals)
    else:
        (low, low_absrel), (high, high_absrel) = extents
        low = low if low_absrel == Bound.ABS else reco_val + low
        high = high if high_absrel == Bound.ABS else reco_val + high
        # extra correction for bias in LineFit_DC's z reco
        if (reco, dim_name) == ("LineFit_DC", "z"):
            if low_absrel == Bound.REL:
                low -= 15
            if high_absrel == Bound.REL:
                high -= 15

    basic_pri_kind = PRI_AZ_INTERP if "azimuth" in dim_name else PRI_INTERP

    prior_def = (
        basic_pri_kind,
        (reco, reco_val, prior_sha256, xvals, pri["pdf"], low, high),
    )

    misc = deepcopy(prior_info["metadata"])
    misc["prior_file_name"] = prior_fname
    misc["prior_file_sha256"] = prior_sha256[:10]
    misc["reco_val"] = reco_val
    misc["split_val"] = split_val

    return prior_def, misc


def define_generic_prior(kind, extents, kwargs):
    """Create prior definition for a `kind` that exists in `scipy.stats.distributions`.

    Parameters
    ----------
    kind : str
        Must be a continuous distribution in `scipy.stats.distributions`
    extents
    kwargs : Mapping
        Must contain keys for any `shapes` (shape parameters) taken by the
        distribution as well as "loc" and "scale" (which are required for all
        distributions).

    Returns
    -------
    prior_def : tuple
        As defined/used in `retro.priors.get_prior_func`; i.e., formatted as ::

            (kind, (arg0, arg1, ..., argN, low, high)

    """
    loc = kwargs["loc"]
    scale = kwargs["scale"]
    dist = getattr(stats.distributions, kind)
    (low, low_absrel), (high, high_absrel) = extents
    if not low_absrel == high_absrel == Bound.ABS:
        raise ValueError('Only absolute bound allowed for `kind` "{}"'.format(kind))
    if dist.shapes:
        args = []
        for shape_param in dist.shapes:
            args.append(kwargs[shape_param])
        args = tuple(args)
    else:
        args = tuple()
    prior_def = (kind, args + (loc, scale, low, high))
    return prior_def


def get_prior_func(dim_num, dim_name, event, kind=None, extents=None, **kwargs):
    """Generate prior function given a prior definition and the actual event

    Parameters
    ----------
    dim_num : int
        the cube dimension number from multinest
    dim_name : str
        parameter name
    event : event
    extents : str or sequence of two floats, optional
    kwargs : any additional arguments

    Returns
    -------
    prior_func : callable
    prior_def : tuple
    misc : OrderedDict

    """
    # -- Set default prior kind & extents depending on dimension name -- #

    if "zenith" in dim_name:
        if kind is None:
            kind = PRI_COSINE
        if extents is None:
            extents = ((0, Bound.ABS), (np.pi, Bound.ABS))
    elif "coszen" in dim_name:
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            extents = ((-1, Bound.ABS), (1, Bound.ABS))
    elif "azimuth" in dim_name:
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            extents = ((0, Bound.ABS), (2 * np.pi, Bound.ABS))
    elif dim_name == "x":
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            extents = EXT_IC[dim_name]
    elif dim_name == "y":
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            extents = EXT_IC[dim_name]
    elif dim_name == "z":
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            extents = EXT_IC[dim_name]

    elif dim_name == "time":
        if kind is None:
            kind = PRI_TIME_RANGE

        if extents is None or kind == PRI_TIME_RANGE:
            tr_keys = []
            for key in event["pulses"].keys():
                if key.endswith("TimeRange"):
                    tr_keys.append(key)
            if not tr_keys:
                raise KeyError("No <pulse series>TimeRange key in pulses")
            if len(tr_keys) > 1:
                sys.stderr.write(
                    "WARNING: found multiple <pulse series>TimeRange keys, using the"
                    " first of {}\n".format(tr_keys)
                )
            time_range = event["pulses"][tr_keys[0]]

            if extents is None:
                low = 0
                high = 0
            else:
                (low, low_absrel), (high, high_absrel) = extents
                assert low_absrel == Bound.REL
                assert high_absrel == Bound.REL

            extents = ((time_range[0] + low, Bound.ABS), (time_range[1] + high, Bound.ABS))

        if kind == PRI_TIME_RANGE:
            kind = PRI_UNIFORM

    elif "energy" in dim_name:
        if kind is None:
            kind = PRI_UNIFORM
        if extents is None:
            if kind in (PRI_LOG_NORMAL, PRI_LOG_UNIFORM):
                extents = ((0.1, Bound.ABS), (1e3, Bound.ABS))
            else:
                extents = ((0.0, Bound.ABS), (1e3, Bound.ABS))

    else:
        raise ValueError('Unrecognized dimension "{}"'.format(dim_name))

    # -- Re-cast priors as something simple to make a func out of -- #

    misc = OrderedDict()

    if kind in (PRI_UNIFORM, PRI_COSINE):
        (low, low_absrel), (high, high_absrel) = extents
        if not low_absrel == high_absrel == Bound.ABS:
            raise ValueError(
                "Dim #{} ({}): Don't know what to do with relative bounds for prior {}".format(
                    dim_num, dim_name, kind
                )
            )
        prior_def = (kind, (low, high))
    elif kind == PRI_OSCNEXT_L5_V1_PREFIT:
        prior_def, misc = define_prior_from_prefit(
            dim_name=dim_name,
            event=event,
            priors=OSCNEXT_L5_V1_PRIORS,
            candidate_recos=["L5_SPEFit11", "LineFit_DC"],
            extents=extents,
            point_estimator="median",
        )
    elif kind == PRI_OSCNEXT_L5_V1_CRS:
        prior_def, misc = define_prior_from_prefit(
            dim_name=dim_name,
            event=event,
            priors=OSCNEXT_L5_V1_PRIORS,
            candidate_recos=["retro_crs_prefit"],
            extents=extents,
            point_estimator="median",
        )
    elif hasattr(stats.distributions, kind):
        prior_def = define_generic_prior(kind, extents, kwargs)
    else:
        raise ValueError(
            'Unhandled or invalid prior "{}" for dim_name "{}"'.format(kind, dim_name)
        )

    # -- Create prior function -- #

    # pylint: disable=unused-argument, missing-docstring

    kind, prior_args = prior_def

    if kind == PRI_UNIFORM:
        low, high = prior_args
        width = high - low

        def prior_func(cube, n=dim_num, width=width, low=low):
            cube[n] = cube[n] * width + low

    elif kind == PRI_LOG_UNIFORM:
        low, high = prior_args
        log_low = np.log(low)
        log_width = np.log(high) - log_low

        def prior_func(cube, n=dim_num, log_width=log_width, log_low=log_low):
            cube[n] = np.exp(cube[n] * log_width + log_low)

    elif kind == PRI_COSINE:
        zen_low, zen_high = prior_args
        cz_low = np.cos(zen_high)
        cz_high = np.cos(zen_low)
        cz_diff = cz_high - cz_low

        def prior_func(cube, n=dim_num, cz_low=cz_low, cz_diff=cz_diff):
            x = (cz_diff * cube[n]) + cz_low
            cube[n] = np.arccos(x)

    elif kind == PRI_GAUSSIAN:
        raise NotImplementedError("limits not correctly working")  # TODO
        mean, stddev, low, high = prior_args
        norm = 1 / (stddev * np.sqrt(TWO_PI))

        def prior_func(cube, n=dim_num, norm=norm, mean=mean, stddev=stddev):
            cube[n] = norm * np.exp(-((cube[n] - mean) / stddev) ** 2)

    elif kind in (PRI_INTERP, PRI_AZ_INTERP):
        x, pdf, low, high = prior_args[-4:]

        if (
            kind == PRI_AZ_INTERP
            and not np.isclose(x.max() - x.min(), high - low)
            or kind == PRI_INTERP
            and (x.min() > low or x.max() < high)
        ):
            print(
                'Dim "{}", prior kind "{}" `x` range = [{}, {}] does not cover'
                " [low, high] range = [{}, {}]".format(
                    dim_name, kind, x.min(), x.max(), low, high
                )
            )

        if kind == PRI_AZ_INTERP:
            if not (np.isclose(low, 0) and np.isclose(high, 2 * np.pi)):
                raise ValueError("az range [low, high) must be [0, 2pi)")

            # Ensure x covers exactly the same distance as (low, high) defines
            highlow_range = high - low
            x = x.min() + (x - x.min()) * highlow_range / (x.max() - x.min())

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            # Create smooth spline interpolator for isf (inverse of cdf)
            isf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext="raise", s=0)

            def prior_func(cube, n=dim_num, isf_interp=isf_interp):
                cube[n] = isf_interp(cube[n]) % (2 * np.pi)

        else:
            # If x covers _more_ than the allowed [low, high] range, resample the
            # pdf in the allowed range (expected to occur for binned zenith and
            # coszen error distributions)
            if x.min() < low or x.max() > high:
                x_orig = x
                pdf_orig = pdf
                x = np.linspace(
                    start=max(low, x_orig.min()),
                    stop=min(high, x_orig.max()),
                    num=len(x_orig),
                ).squeeze()
                pdf = np.interp(x=x, xp=x_orig, fp=pdf_orig)
                integral = np.trapz(x=x, y=pdf)
                try:
                    pdf /= integral
                except:
                    print("low, high:", low, high)
                    print("x_orig.min, x_orig.max:", x_orig.min(), x_orig.max())
                    print("len(x):", len(x))
                    print("x.shape:", x.shape)
                    print("x:", x)
                    print("x_orig.shape:", x_orig.shape)
                    print("x_orig:", x_orig)
                    print("pdf_orig.shape:", pdf_orig.shape)
                    print("pdf.shape:", pdf.shape)
                    print("integral.shape:", integral.shape)
                    raise

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            # Create smooth spline interpolator for isf (inverse of cdf)
            isf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext="raise", s=0)

            def prior_func(cube, n=dim_num, isf_interp=isf_interp):
                cube[n] = isf_interp(cube[n])

    elif hasattr(stats.distributions, kind):
        dist_args = prior_args[:-2]
        low, high = prior_args[-2:]
        frozen_dist_isf = getattr(stats.distributions, kind)(*dist_args).isf

        def prior_func(
            cube, frozen_dist_isf=frozen_dist_isf, dim_num=dim_num, low=low, high=high
        ):
            cube[dim_num] = np.clip(
                frozen_dist_isf(cube[dim_num]), a_min=low, a_max=high
            )

    else:
        raise NotImplementedError('Prior "{}" not implemented.'.format(kind))

    return prior_func, prior_def, misc
