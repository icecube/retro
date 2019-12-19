# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Linear interpolation, numba-compiled
"""

from __future__ import absolute_import, division, print_function

__all__ = ["OutOfBoundsBehavior", "generate_lerp"]

import enum
from os.path import abspath, dirname
import sys

import numba
import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.geom import generate_digitizer
from retro.utils.misc import validate_and_convert_enum

__author__ = 'J.L. Lanfranchi'

__license__ = '''Copyright 2018 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


class OutOfBoundsBehavior(enum.IntEnum):
    """Methods for handling x-values outside the range of the original x array"""
    error = 0
    constant = 1
    extrapolate = 2


def generate_lerp(
    x, y, low_behavior, high_behavior, low_val=None, high_val=None,
):
    """Generate a numba-compiled linear interpolation function.

    Parameters
    ----------
    x : array

    y : array

    low_behavior : OutOfBoundsBehavior or str in {"constant", "extrapolate", or "error"}

    high_behavior : OutOfBoundsBehavior or str in {"constant", "extrapolate", or "error"}

    low_val : float, optional
        If `low_behavior` is "constant", use this value; if `low_val` is not
        specified, the y-value corresponding to the lowest `x` is used.

    high_val : float, optional
        If `high_behavior` is "constant", use this value; if `high_val` is not
        specified, the y-value corresponding to the highest `x` is used.

    Returns
    -------
    scalar_lerp : callable
        Takes a scalar x-value and returns corresponding y value if x is in the range of
        the above `x` array; if not, `low_behavior` and `high_behavior` are followed.

    vectorized_lerp : callable
        Identical `scalar_lerp` but callable with numpy arrays (via `numba.vectorize`)

    """
    sort_ind = np.argsort(x)
    x_ = x[sort_ind]
    y_ = y[sort_ind]

    x_min, x_max = x_[0], x_[-1]
    y_min, y_max = y_[0], y_[-1]

    # set `clip=True` so extrapolation works
    digitize = generate_digitizer(bin_edges=x, clip=True)

    low_behavior = validate_and_convert_enum(
        val=low_behavior,
        enum_type=OutOfBoundsBehavior,
    )
    high_behavior = validate_and_convert_enum(
        val=high_behavior,
        enum_type=OutOfBoundsBehavior,
    )

    # Note that Numba requires all values to be same type on compile time, so if
    # `{low,high}_val` is not used, convert to a float (use np.nan if not being used)

    if low_behavior in (OutOfBoundsBehavior.error, OutOfBoundsBehavior.extrapolate):
        if low_val is not None:
            raise ValueError(
                "`low_val` is unused for {} `low_behavior`"
                .format(low_behavior.name)
            )
        low_val = np.nan
    elif low_behavior is OutOfBoundsBehavior.constant:
        if low_val is not None:
            low_val = np.float(low_val)
        else:
            low_val = np.float(y_min)

    if high_behavior in (OutOfBoundsBehavior.error, OutOfBoundsBehavior.extrapolate):
        if high_val is not None:
            raise ValueError(
                "`high_val` is unused for {} `high_behavior`"
                .format(high_behavior.name)
            )
        high_val = np.nan
    elif high_behavior is OutOfBoundsBehavior.constant:
        if high_val is not None:
            high_val = np.float(high_val)
        else:
            high_val = np.float(y_max)

    @numba.jit(fastmath=False, cache=True, nogil=True)
    def scalar_lerp(x):
        """Linearly interpolate to find `y` from `x`.

        Parameters
        ----------
        x

        Returns
        -------
        y

        """
        if x < x_min:
            if low_behavior is OutOfBoundsBehavior.error:
                raise ValueError("`x` is below valid range")
            elif low_behavior is OutOfBoundsBehavior.constant:
                return low_val

        if x > x_max:
            if high_behavior is OutOfBoundsBehavior.error:
                raise ValueError("`x` is above valid range")
            elif high_behavior is OutOfBoundsBehavior.constant:
                return high_val

        bin_num = digitize(x)
        x0 = x_[bin_num]
        x1 = x_[bin_num + 1]
        y0 = y_[bin_num]
        y1 = y_[bin_num + 1]
        f = (x - x0) / (x1 - x0)

        return y0*(1 - f) + y1*f

    vectorized_lerp = numba.vectorize()(scalar_lerp)

    return scalar_lerp, vectorized_lerp


def test_generate_lerp():
    """Unit tests for `generate_lerp` and the functions it generates."""
    from scipy.interpolate import UnivariateSpline
    func = lambda x: 100 + x**2
    x = np.linspace(1, 10, 5)
    y = func(x)
    spl = UnivariateSpline(x, y, k=1, s=0, ext=0)
    scalar_lerp, vectorized_lerp = generate_lerp(
        x=x,
        y=y,
        low_behavior="extrapolate",
        high_behavior="extrapolate",
    )
    xx = np.linspace(-1, 12, 1000)
    spl_yy = spl(xx)
    scalar_lerp_yy = np.array([scalar_lerp(xval) for xval in xx])
    vectorized_lerp_yy = vectorized_lerp(xx)
    assert np.allclose(scalar_lerp_yy, spl_yy)
    assert np.allclose(vectorized_lerp_yy, spl_yy)
    print("<< PASS : test_generate_lerp >>")


if __name__ == "__main__":
    test_generate_lerp()
