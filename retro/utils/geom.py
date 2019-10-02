# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Utils for binning and geometry
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'GEOM_FILE_PROTO',
    'GEOM_META_PROTO',
    'generate_geom_meta',
    'linbin',
    'test_linbin',
    'powerbin',
    'test_powerbin',
    'powerspace',
    'inv_power_2nd_diff',
    'infer_power',
    'test_infer_power',
    'sample_powerlaw_binning',
    'generate_digitizer',
    'test_generate_digitizer',
    'bin_edges_to_binspec',
    'linear_bin_centers',
    'spacetime_separation',
    'spherical_volume',
    'sph2cart',
    'pol2cart',
    'cart2pol',
    'cart2sph',
    'cart2sph_np',
    'sph2cart_np',
    'rotsph2cart',
    'rotate_point',
    'rotate_points',
    'add_vectors',
    'fill_from_spher',
    'fill_from_cart',
    'reflect',
]

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from collections import OrderedDict
import math
from os.path import abspath, dirname
import sys
from time import time

import numpy as np
from scipy.optimize import brentq

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DEBUG, DFLT_NUMBA_JIT_KWARGS, load_pickle, numba_jit
from retro.const import SPEED_OF_LIGHT_M_PER_NS
from retro.retro_types import TimeSphCoord
from retro.utils.misc import hash_obj


NUMBA_JIT_KWARGS = dict(nopython=True, nogil=True, fastmath=True, error_model="numpy")

GEOM_FILE_PROTO = 'geom_{hash:s}.npy'
"""File containing detector geometry as a Numpy 5D array with coordinates
(string, om, x, y, z)"""

GEOM_META_PROTO = 'geom_{hash:s}_meta.json'
"""File containing metadata about source of detector geometry"""


def generate_geom_meta(geom):
    """Generate geometry metadata dict. Currently, this sinmply hashes on the
    geometry coordinates. Note that the values are rounded to the nearest
    centimeter for hashing purposes. (Also, the values are converted to
    integers at this precision to eliminate any possible float32 / float64
    issues that could cause discrepancies in hash values for what we consider
    to be equal geometries.)

    Parameters
    ----------
    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float{32,64}

    Returns
    -------
    metadata : OrderedDict
        Contains the item:
            'hash' : length-8 str
                Hex characters convert to a string of length 8

    """
    assert len(geom.shape) == 3
    assert geom.shape[2] == 3
    rounded_ints = np.round(geom * 100).astype(np.int)
    geom_hash = hash_obj(rounded_ints, fmt='hex')[:8]
    return OrderedDict([('hash', geom_hash)])


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def _linbin_numba(val, start, stop, num):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int <= 2**31
        Number of bin _edges_ (number of bins is one less than `num`)

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    assert num_bins < 2**31
    width = (stop - start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.int32)
    for i in range(val.size):
        bin_num.flat[i] = (val.flat[i] - start) // width
    return bin_num


def _linbin_numpy(val, start, stop, num):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : scalar or array
    start : float
    stop : float
    num : int <= 2**31
        Number of bin _edges_ (number of bins is one less than `num`)

    Returns
    -------
    bin_num : scalar or ndarray of dtype np.int32

    """
    num_bins = num - 1
    assert num_bins < 2**31
    width = (stop - start) / num_bins
    bin_num = (val - start) // width
    if np.isscalar(bin_num):
        bin_num = np.int32(bin_num)
    else:
        bin_num = bin_num.astype(np.int32)
    return bin_num


# Pick the fastest implementation
linbin = _linbin_numba # pylint: disable=invalid-name


def test_linbin():
    """Unit tests for function `linbin`."""
    kw = dict(start=0, stop=100, num=200)

    for ftype in [np.float64]:
        print("ftype:", ftype)
        bin_edges = np.linspace(**kw).astype(ftype)
        rand = np.random.RandomState(seed=0)
        x = rand.uniform(0, 100, int(1e6)).astype(ftype)

        test_args = (
            np.array([0.0], dtype=ftype),
            ftype(kw['start']),
            ftype(kw['stop']),
            np.int32(kw['num']),
        )
        _linbin_numba(*test_args)

        test_args = (
            x,
            ftype(kw['start']),
            ftype(kw['stop']),
            np.int32(kw['num']),
        )
        t0 = time()
        bins_ref = np.digitize(x, bin_edges) - 1
        t1 = time()
        bins_test_numba = _linbin_numba(*test_args)
        t2 = time()
        bins_test_numpy = _linbin_numpy(*test_args)
        t3 = time()

        print('np.digitize:   {} s'.format(t1 - t0))
        print('_linbin_numba: {} s'.format(t2 - t1))
        print('_linbin_numpy: {} s'.format(t3 - t2))

        assert np.all(bins_test_numba == bins_ref), (
            "\n{}\n{}\n{}".format(
                x[bins_test_numba != bins_ref],
                bins_test_numba[bins_test_numba != bins_ref],
                bins_ref[bins_test_numba != bins_ref]
            )
        )
        assert np.all(bins_test_numpy == bins_ref), (
            "\n{}\n{}\n{}".format(
                x[bins_test_numpy != bins_ref],
                bins_test_numpy[bins_test_numpy != bins_ref],
                bins_ref[bins_test_numpy != bins_ref]
            )
        )
        scalar_result = _linbin_numpy(1, **kw)
        assert isinstance(scalar_result, np.int32), type(scalar_result)
        scalar_result = _linbin_numpy(np.array([1]), **kw)[0]
        assert isinstance(scalar_result, np.int32), type(scalar_result)
        print("")

    print('<< PASS : test_linbin >>')


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def _powerbin_numba(val, start, stop, num, power):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : array
    start : float
    stop : float
    num : int <= 2**31
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float

    Returns
    -------
    bin_num : ndarray of dtype np.int32

    """
    num_bins = num - 1
    assert num_bins < 2**31
    inv_power = 1.0 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.int32)
    for i in range(val.size):
        bin_num.flat[i] = (val.flat[i]**inv_power - inv_power_start) // inv_power_width
    return bin_num


def _powerbin_numpy(val, start, stop, num, power):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : scalar or array
    start : float
    stop : float
    num : int <= 2**31
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float

    Returns
    -------
    bin_num : scalar or ndarray of dtype np.int32

    """
    num_bins = num - 1
    assert num_bins < 2**31
    inv_power = 1 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = (val**inv_power - inv_power_start) // inv_power_width
    if np.isscalar(bin_num):
        bin_num = np.int32(bin_num)
    else:
        bin_num = bin_num.astype(np.int32)
    return bin_num


# Pick the fastest implementation
powerbin = _powerbin_numba # pylint: disable=invalid-name


def test_powerbin():
    """Unit tests for function `powerbin`."""
    kw = dict(start=0, stop=100, num=100, power=2)

    rand = np.random.RandomState(seed=0)

    for ftype in [np.float64]:
        bin_edges = powerspace(**kw).astype(ftype)
        print(ftype)
        utype = np.int32
        x = rand.uniform(0, 100, int(1e6)).astype(ftype)
        test_args = (
            ftype(kw['start']),
            ftype(kw['stop']),
            utype(kw['num']),
            utype(kw['power']),
        )

        # Run once to force compilation
        _powerbin_numba(np.array([0.0], dtype=ftype), *test_args)

        # Run actual tests / take timings
        t0 = time()
        bins_ref = np.digitize(x, bin_edges) - 1
        t1 = time()
        bins_test_numba = _powerbin_numba(x, *test_args)
        t2 = time()
        bins_test_numpy = _powerbin_numpy(x, *test_args)
        t3 = time()

        print('np.digitize:     {:e} s'.format(t1 - t0))
        print('_powerbin_numba: {:e} s'.format(t2 - t1))
        print('_powerbin_numpy: {:e} s'.format(t3 - t2))

        assert np.all(bins_test_numba == bins_ref), (
            "{}\n{}".format(
                bins_test_numba[bins_test_numba != bins_ref],
                bins_ref[bins_test_numba != bins_ref]
            )
        )
        assert np.all(bins_test_numpy == bins_ref), (
            "{}\n{}".format(
                bins_test_numpy[bins_test_numpy != bins_ref],
                bins_ref[bins_test_numpy != bins_ref]
            )
        )
        scalar_result = _powerbin_numpy(1, **kw)
        assert isinstance(scalar_result, np.int32), type(scalar_result)
        scalar_result = _powerbin_numba(np.array([1]), **kw)[0]
        assert isinstance(scalar_result, np.int32), type(scalar_result)
        print('')

    print('<< PASS : test_powerbin >>')


# TODO: add `endpoint`, `retstep`, and `dtype` kwargs
def powerspace(start, stop, num, power):
    """Create bin edges evenly spaced w.r.t. ``x**power``.

    Reverse engineered from Jakob van Santen's power axis, with arguments
    defined with analogy to :function:`numpy.linspace` (adding `power` but
    removing the `endpoint`, `retstep`, and `dtype` arguments).

    Parameters
    ----------
    start : float
        Lower-most bin edge

    stop : float
        Upper-most bin edge

    num : int
        Number of edges (this defines ``num - 1`` bins)

    power : float
        Power-law to use for even spacing

    Returns
    -------
    edges : numpy.ndarray of shape (1, num)
        Edges

    """
    inv_power = 1 / power
    liner_edges = np.linspace(start=np.power(start, inv_power),
                              stop=np.power(stop, inv_power),
                              num=num)
    bin_edges = np.power(liner_edges, power)
    return bin_edges


def inv_power_2nd_diff(power, edges):
    """Second finite difference of edges**(1/power)"""
    return np.diff(edges**(1/power), n=2)


def infer_power(edges, dtype=None):
    """Infer the power used for bin edges evenly spaced w.r.t. ``x**power``.

    Parameters
    ----------
    edges : array
    dtype : numpy floating dtype, optional
        Sets the precision for the solver (to 4*dtype.eps). Defaults to
        `edges.dtype`.

    Returns
    -------
    power : float

    """
    atol = 1e-15
    if dtype is None:
        dtype = edges.dtype
    first_three_edges = edges[:3]
    rtol = 4*np.finfo(dtype).eps
    power = None
    try:
        power = brentq(
            f=inv_power_2nd_diff,
            a=1, b=100,
            maxiter=1000, xtol=atol, rtol=rtol,
            args=(first_three_edges,)
        )
    except RuntimeError:
        raise ValueError('Edges do not appear to be power-spaced'
                         ' (optimizer did not converge)')
    diff = inv_power_2nd_diff(power, edges)
    if not np.allclose(diff, diff[0], atol=1000*atol, rtol=10*rtol):
        raise ValueError('Edges do not appear to be power-spaced'
                         ' (power found does not hold for all edges)\n%s'
                         % str(diff))
    return power


def test_infer_power():
    """Unit test for function `infer_power`"""
    ref_powers = np.arange(1, 10, 0.001)
    total_time = 0.0
    for ref_power in ref_powers:
        edges = powerspace(start=0, stop=400, num=201, power=ref_power)
        try:
            t0 = time()
            inferred_power = infer_power(edges)
            t1 = time()
        except ValueError:
            print(ref_power, edges)
            raise
        assert np.isclose(inferred_power, ref_power,
                          atol=1e-14, rtol=4*np.finfo(np.float).eps), ref_power
        total_time += t1 - t0
    print('Average time to infer power: {} s'
          .format(total_time/len(ref_powers)))
    print('<< PASS : test_infer_power >>')


def sample_powerlaw_binning(edges, samples_per_bin):
    """Draw samples evenly distributed in bins which are spaced evenly with
    respect to the (inverse) power of the dimension.

    Parameters
    ----------
    edges : array
        Edges of bins to sample within.

    samples_per_bin : int > 0
        Number of samples per bin.

    Returns
    -------
    samples : array

    """
    num_bins = len(edges) - 1
    pwr = infer_power(edges)
    edges_to_inv_pwr = edges**(1/pwr)
    delta_eip = (edges_to_inv_pwr[-1] - edges_to_inv_pwr[0]) / num_bins
    half_delta_eip_s = delta_eip / samples_per_bin / 2

    samples = np.linspace(
        start=edges_to_inv_pwr[0] + half_delta_eip_s,
        stop=edges_to_inv_pwr[-1] - half_delta_eip_s,
        num=samples_per_bin * num_bins
    ) ** pwr

    return samples


def generate_digitizer(bin_edges, clip=True, handle_under_overflow=True):
    """Factory to generate a specialized Numba function for "digitizing" data
    (i.e., returning which bin a value falls within).

    Parameters
    ----------
    bin_edges : array-like

    clip : bool, optional
        If `True`, clip values to valid range: return 0 for underflow or `num_bins - 1`
        for overflow; if `False`, return -1 and `num_bins` for underflow and overflow,
        respectively. `handle_under_overflow` = False means that `clip` is
        effectively ignored.

    handle_under_overflow : bool, optional
        Whether or not to ensure values below smallest / above largest bin
        return a valid value. If False, `clip` is ignored.

    Returns
    -------
    digitize : callable

    Notes
    -----
    All bins except the last are half open (i.e., include their lower edges but exclude
    their upper edges), except for the last bin, which is closed (i.e., include both
    lower and upper edges).

    The digitizer returned does NOT fail if a value lies outside the
    binning boundaries.

    """
    # pylint: disable=missing-docstring, function-redefined
    bin_edges = np.asarray(bin_edges)
    assert np.all(np.diff(bin_edges) > 0)
    start = bin_edges[0]
    stop = bin_edges[-1]
    num_bin_edges = len(bin_edges)
    num_bins = num_bin_edges - 1

    if not clip:
        underflow_idx = -1
        overflow_idx = num_bins
    else:
        underflow_idx = 0
        overflow_idx = num_bins - 1

    power = None
    bin_widths = np.diff(bin_edges)
    if np.allclose(bin_widths, bin_widths[0]):
        power = 1
    else:
        try:
            power = infer_power(bin_edges)
        except ValueError:
            pass
        else:
            recip_power = 1 / power
            start_recip_power = start**recip_power
            stop_recip_power = stop**recip_power
            power_width = (stop_recip_power - start_recip_power) / num_bins
            recip_power_width = 1 / power_width

    is_log = False
    if power is None and start > 0:
        log_bin_edges = np.log(bin_edges)
        logwidth = np.diff(log_bin_edges)
        if np.allclose(logwidth, logwidth[0]):
            if DEBUG:
                print('log')
            is_log = True
            logwidth = (log_bin_edges[-1] - log_bin_edges[0]) / num_bins
            recip_logwidth = 1 / logwidth
            log_start = log_bin_edges[0]

    digitize = None
    bindescr = None

    if power == 1:
        dx = (stop - start) / num_bins
        recip_dx = 1 / dx
        bindescr = (
            '{} bins linearly spaced from {} to {}'.format(num_bins, start, stop)
        )
        if handle_under_overflow:
            def digitize(val):
                if val < start:
                    return underflow_idx
                if val > stop:
                    return overflow_idx
                idx = int((val - start) * recip_dx)
                return min(max(0, idx), num_bins - 1)
        else:
            def digitize(val):
                return int((val - start) * recip_dx)

    elif power:
        bindescr = (
            '{} bins spaced from {} to {} spaced with power of {}'
            .format(num_bins, start, stop, power)
        )
        if np.isclose(power, 2):
            if handle_under_overflow:
                def digitize(val):
                    if val < start:
                        return underflow_idx
                    if val > stop:
                        return overflow_idx
                    idx = int((math.sqrt(val) - start_recip_power) * recip_power_width)
                    return min(max(0, idx), num_bins - 1)
            else:
                def digitize(val):
                    return int((math.sqrt(val) - start_recip_power) * recip_power_width)

        elif num_bins > 1e3: # faster to do binary search if fewer bins
            if handle_under_overflow:
                def digitize(val):
                    if val < start:
                        return underflow_idx
                    if val > stop:
                        return overflow_idx
                    idx = int((val**recip_power - start_recip_power) * recip_power_width)
                    return min(max(0, idx), num_bins - 1)
            else:
                def digitize(val):
                    return int((val**recip_power - start_recip_power) * recip_power_width)

    elif is_log:
        bindescr = (
            '{} bins logarithmically spaced from {} to {}'.format(num_bins, start, stop)
        )
        if num_bins > 20: # faster to do binary search if fewer bins
            if handle_under_overflow:
                def digitize(val):
                    if val < start:
                        return underflow_idx
                    if val > stop:
                        return overflow_idx
                    idx = int((math.log(val) - log_start) * recip_logwidth)
                    return min(max(0, idx), num_bins - 1)
            else:
                def digitize(val):
                    return int((math.log(val) - log_start) * recip_logwidth)

    if bindescr is None:
        bindescr = (
            '{} bins unevenly spaced from {} to {}'.format(num_bins, start, stop)
        )

    if digitize is None:
        if handle_under_overflow:
            def digitize(val):
                if val < start:
                    return underflow_idx
                if val > stop:
                    return overflow_idx
                # -- Binary search -- #
                left_idx = 0
                right_idx = num_bins
                while left_idx < right_idx:
                    idx = left_idx + ((right_idx - left_idx) >> 1)
                    if val >= bin_edges[idx]:
                        left_idx = idx + 1
                    else:
                        right_idx = idx
                idx = left_idx - 1
                return min(max(0, idx), num_bins - 1)
        else:
            def digitize(val):
                # -- Binary search -- #
                left_idx = 0
                right_idx = num_bins
                while left_idx < right_idx:
                    idx = left_idx + ((right_idx - left_idx) >> 1)
                    if val >= bin_edges[idx]:
                        left_idx = idx + 1
                    else:
                        right_idx = idx
                return left_idx - 1

    digitize.__doc__ = (
        """Find bin index for a value.

        Binning is set to {}.

        Parameters
        ----------
        val : scalar
            Value for which to find bin index.

        Returns
        -------
        idx : int
            Bin index; `idx < 0` or `idx >= num_bins` indicates `val` is
            outside binning.

        """.format(bindescr)
    )
    digitize = numba_jit(**NUMBA_JIT_KWARGS)(digitize)

    if DEBUG:
        print(bindescr)

    return digitize


def test_generate_digitizer():
    """Test the functions that `generate_digitizer` produces."""
    # TODO: use local file for this test
    meta = load_pickle(
        '/home/icecube/retro/tables/'
        'large_5d_notilt_combined/stacked/stacked_ckv_template_map_meta.pkl'
    )
    binning = meta['binning']

    for dim, edges in binning.items():
        assert np.all(np.diff(edges) > 0)
        num_bins = len(edges) - 1
        digitize = generate_digitizer(edges)
        digitize_overflow = generate_digitizer(edges, clip=False)
        rand = np.random.RandomState(0)

        # Check lots of values within the valid range of the binning
        vals = rand.uniform(low=edges[0], high=edges[-1], size=int(1e5))
        test = np.array([digitize(v) for v in vals])
        ref = np.digitize(vals, bins=edges, right=False) - 1
        assert np.all(test == ref), dim

        # Check edge cases
        assert digitize(edges[0]) == 0, dim
        assert digitize(edges[0] - 1e-8) == 0, dim
        assert digitize_overflow(edges[0] - 1e-8) < 0, dim
        assert digitize(edges[-1]) == num_bins - 1, dim
        assert digitize(edges[-1] + 1e-8) == num_bins - 1, dim
        assert digitize_overflow(edges[-1] + 1e-8) == num_bins, dim

    print('<< PASS : test_generate_digitizer >>')


def bin_edges_to_binspec(edges):
    """Convert bin edges to a binning specification (start, stop, and num_bins).

    Note:
    * t-bins are assumed to be linearly spaced in ``t``
    * r-bins are assumed to be evenly spaced w.r.t. ``r**2``
    * theta-bins are assumed to be evenly spaced w.r.t. ``cos(theta)``
    * phi bins are assumed to be linearly spaced in ``phi``

    Parameters
    ----------
    edges

    Returns
    -------
    start : TimeSphCoord containing floats
    stop : TimeSphCoord containing floats
    num_bins : TimeSphCoord containing ints

    """
    dims = TimeSphCoord._fields
    start = TimeSphCoord(*(np.min(getattr(edges, d)) for d in dims))
    stop = TimeSphCoord(*(np.max(getattr(edges, d)) for d in dims))
    num_bins = TimeSphCoord(*(len(getattr(edges, d)) - 1 for d in dims))

    return start, stop, num_bins


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def linear_bin_centers(bin_edges):
    """Return bin centers for bins defined in a linear space.

    Parameters
    ----------
    bin_edges : sequence of numeric
        Note that all numbers contained must be of the same dtype (this is a
        limitation due to numba, at least as of version 0.35).

    Returns
    -------
    bin_centers : numpy.ndarray
        Length is one less than that of `bin_edges`, and datatype is inferred
        from the first element of `bin_edges`.

    """
    num_edges = len(bin_edges)
    bin_centers = np.empty(num_edges - 1, bin_edges.dtype)
    edge0 = bin_edges[0]
    for n in range(num_edges - 1):
        edge1 = bin_edges[n + 1]
        bin_centers[n] = 0.5 * (edge0 + edge1)
        edge0 = edge1
    return bin_centers


def spacetime_separation(dt, dx, dy, dz):
    """Compute the separation between two events in spacetime. Negative values
    are non-causal.

    Parameters
    ----------
    dt, dx, dy, dz : numeric
        Separation between events in nanoseconds (dt) and meters (dx, dy, and
        dz).

    """
    return SPEED_OF_LIGHT_M_PER_NS*dt - np.sqrt(dx**2 + dy**2 + dz**2)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def spherical_volume(rmin, rmax, dcostheta, dphi):
    """Find volume of a finite element defined in spherical coordinates.

    Parameters
    ----------
    rmin, rmax : float (in arbitrary distance units)
        Difference between initial and final radii.

    dcostheta : float
        Difference between initial and final zenith angles' cosines (where
        zenith angle is defined as out & down from +Z axis).

    dphi : float (in units of radians)
        Difference between initial and final azimuth angle (defined as positive
        from +X-axis towards +Y-axis looking "down" on the XY-plane (i.e.,
        looking in -Z direction).

    Returns
    -------
    vol : float
        Volume in units of the cube of the units that ``dr`` is provided in.
        E.g. if those are provided in meters, ``vol`` will be in units of `m^3`.

    """
    return -dcostheta * (rmax**3 - rmin**3) * dphi / 3


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def sph2cart(r, theta, phi, x, y, z):
    """Convert spherical polar (r, theta, phi) to 3D Cartesian (x, y, z)
    Coordinates.

    Parameters
    ----------
    r, theta, phi : numeric of same shape

    x, y, z : numpy.ndarrays of same shape as `r`, `theta`, `phi`
        Result is stored in these arrays.

    """
    shape = r.shape
    num_elements = int(np.prod(np.array(shape)))
    r_flat = r.flat
    theta_flat = theta.flat
    phi_flat = phi.flat
    x_flat = x.flat
    y_flat = y.flat
    z_flat = z.flat
    for idx in range(num_elements):
        rfi = r_flat[idx]
        thetafi = theta_flat[idx]
        phifi = phi_flat[idx]
        rhofi = rfi * math.sin(thetafi)
        x_flat[idx] = rhofi * math.cos(phifi)
        y_flat[idx] = rhofi * math.sin(phifi)
        z_flat[idx] = rfi * math.cos(thetafi)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pol2cart(r, theta, x, y):
    """Convert plane polar (r, theta) to Cartesian (x, y) Coordinates.

    Parameters
    ----------
    r, theta : numeric of same shape

    x, y : numpy.ndarrays of same shape as `r`, `theta`
        Result is stored in these arrays.

    """
    shape = r.shape
    num_elements = int(np.prod(np.array(shape)))
    r_flat = r.flat
    theta_flat = theta.flat
    x_flat = x.flat
    y_flat = y.flat
    for idx in range(num_elements):
        rf = r_flat[idx]
        tf = theta_flat[idx]
        x_flat[idx] = rf * math.cos(tf)
        y_flat[idx] = rf * math.sin(tf)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def cart2pol(x, y, r, theta):
    """Convert plane Cartesian (x, y) to plane polar (r, theta) Coordinates.

    Parameters
    ----------
    x, y : numeric of same shape

    r, theta : numpy.ndarrays of same shape as `x`, `y`
        Result is stored in these arrays.

    """
    shape = x.shape
    num_elements = int(np.prod(np.array(shape)))
    x_flat = x.flat
    y_flat = y.flat
    r_flat = r.flat
    theta_flat = theta.flat
    for idx in range(num_elements):
        xfi = x_flat[idx]
        yfi = y_flat[idx]
        r_flat[idx] = math.sqrt(xfi*xfi + yfi*yfi)
        theta_flat[idx] = math.atan2(yfi, xfi)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def cart2sph(x, y, z, r, theta, phi):
    """Convert 3D Cartesian (x, y, z) to spherical polar (r, theta, phi)
    Coordinates.

    Parameters
    ----------
    x, y, z : numeric of same shape

    r, theta, phi : numpy.ndarrays of same shape as `x`, `y`, `z`
        Result is stored in these arrays.

    """
    shape = x.shape
    num_elements = int(np.prod(np.array(shape)))
    x_flat = x.flat
    y_flat = y.flat
    z_flat = z.flat
    r_flat = r.flat
    theta_flat = theta.flat
    phi_flat = phi.flat
    for idx in range(num_elements):
        xfi = x_flat[idx]
        yfi = y_flat[idx]
        zfi = z_flat[idx]
        rfi = math.sqrt(xfi*xfi + yfi*yfi + zfi*zfi)
        r_flat[idx] = rfi
        phi_flat[idx] = math.atan2(yfi, xfi)
        theta_flat[idx] = math.acos(zfi / rfi)


def cart2sph_np(x, y, z):
    """Convert Cartesian (x, y, z) coordinates into spherical (r, theta, phi)
    coordinates, where the latter follows the convention that theta is the
    zenith angle from the z-axis towards the xy-plane (in [0, pi]) and phi is
    the azimuthal angle from the x-axis towards the y-axis (in [0, 2pi)).

    Parameters
    ----------
    x, y, z : scalars or arrays of same shape

    Returns
    -------
    r, theta, phi : scalars or arrays of same shape as {x, y, z}

    """
    is_scalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    if is_scalar:
        if r == 0:
            theta = 0.
        else:
            theta = np.arccos(z / r)
    else:
        mask = r > 0
        theta = np.zeros_like(r)
        theta[mask] = np.arccos(z[mask] / r[mask])
    phi = np.arctan2(y, x)
    return r, theta, phi


def sph2cart_np(r, theta, phi):
    """Convert spherical (r, theta, phi) coordinates into Cartesian (x, y, z)
    coordinates, where the former follows the convention that theta is the
    zenith angle from the z-axis towards the xy-plane (in [0, pi]) and phi is
    the azimuthal angle from the x-axis towards the y-axis (in [0, 2pi)).

    Parameters
    ----------
    r, theta, phi : scalars or arrays of same shape

    Returns
    -------
    x, y, z : scalars or arrays of same shape as {r, theta, phi}

    """
    z = r * np.cos(theta)
    rho = r * np.sin(theta)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y, z


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def rotsph2cart(p_sintheta, p_costheta, p_phi, rot_sintheta, rot_costheta,
                rot_phi):
    """Rotate a vector `p` (defined by its theta and phi coordinates) by `rot`
    (defined by the theta and phi components) and return the result `q`
    (defined by its x, y, and z coordinates).

    Parameters
    ----------
    p_costheta, p_phi
        Spherical angular coordinates of vector to be rotated
    rot_costheta, rot_phi
        Angles to rotate vector by

    Returns
    -------
    x, y, z
        Cartesian components of the rotated vector (length = 1)

    Notes
    -----
    See `cone_in_spherical_coords.ipynb` for derivation of the equations.

    """
    # TODO: use trig identities to reduce number of trig function calls; should
    # speed code up significantly

    p_sinphi = np.sin(p_phi)
    p_cosphi = np.cos(p_phi)

    rot_sinphi = np.sin(rot_phi)
    rot_cosphi = np.cos(rot_phi)

    # Intermediate computations that can be reused
    psphi_pstheta = p_sinphi * p_sintheta
    pcphi_pstheta = p_cosphi * p_sintheta
    pctheta_rstheta = p_costheta * rot_sintheta

    q_x = (-psphi_pstheta * rot_sinphi
           + pcphi_pstheta * rot_cosphi * rot_costheta
           + pctheta_rstheta * rot_cosphi)
    q_y = (psphi_pstheta * rot_cosphi
           + pcphi_pstheta * rot_sinphi * rot_costheta
           + pctheta_rstheta * rot_sinphi)
    q_z = -pcphi_pstheta * rot_sintheta + p_costheta * rot_costheta

    return q_x, q_y, q_z


@numba_jit
def rotate_point(p_theta, p_phi, rot_theta, rot_phi):
    """Rotate a point `p` by `rot` resulting in a new point `q`.

    Parameters
    ----------
    p_theta :  float
        Zenith

    p_phi : float
        Azimuth

    rot_theta : float
        Rotate the point to have axis of symmetry defined by (rot_theta, rot_phi)

    rot_phi : float
        Rotate the point to have axis of symmetry defined by (rot_theta, rot_phi)

    Returns
    -------
    q_theta : float
        theta coordinate of rotated point

    q_phi : float
        phi coordinate of rotated point

    """
    sin_rot_theta = math.sin(rot_theta)
    cos_rot_theta = math.cos(rot_theta)

    sin_rot_phi = math.sin(rot_phi)
    cos_rot_phi = math.cos(rot_phi)

    sin_p_theta = math.sin(p_theta)
    cos_p_theta = math.cos(p_theta)

    sin_p_phi = math.sin(p_phi)
    cos_p_phi = math.cos(p_phi)

    q_theta = math.acos(-sin_p_theta * sin_rot_theta * cos_p_phi + cos_p_theta * cos_rot_theta)
    q_phi = math.atan2(
        (sin_p_phi * sin_p_theta * cos_rot_phi)
        + (sin_p_theta * sin_rot_phi * cos_p_phi * cos_rot_theta)
        + (sin_rot_phi * sin_rot_theta * cos_p_theta),

        (-sin_p_phi * sin_p_theta * sin_rot_phi)
        + (sin_p_theta * cos_p_phi * cos_rot_phi * cos_rot_theta)
        + (sin_rot_theta * cos_p_theta * cos_rot_phi)
    )

    return q_theta, q_phi


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def rotate_points(p_theta, p_phi, rot_theta, rot_phi, q_theta, q_phi):
    """Rotate points `p` by `rot` resulting in new points `q`.

    Parameters
    ----------
    p_theta :  array of float
        theta coordinate.

    p_phi : array of float
        Azimuth on the circle

    rot_theta :  array of float
        Rotate the points to have axis of symmetry defined by (rot_theta, rot_phi)

    rot_phi :  array of float
        Rotate the points to have axis of symmetry defined by (rot_theta, rot_phi)

    q_theta : array of float
        theta coordinate of rotated points

    q_phi : array of float
        phi coordinate of rotated points

    """
    for i in range(len(p_theta)):  # pylint: disable=consider-using-enumerate
        sin_rot_theta = math.sin(rot_theta[i])
        cos_rot_theta = math.cos(rot_theta[i])

        sin_rot_phi = math.sin(rot_phi[i])
        cos_rot_phi = math.cos(rot_phi[i])

        sin_p_theta = math.sin(p_theta[i])
        cos_p_theta = math.cos(p_theta[i])

        sin_p_phi = math.sin(p_phi[i])
        cos_p_phi = math.cos(p_phi[i])

        q_theta[i] = math.acos(
            -sin_p_theta * sin_rot_theta * cos_p_phi
            + cos_p_theta * cos_rot_theta
        )
        q_phi[i] = math.atan2(
            (sin_p_phi * sin_p_theta * cos_rot_phi)
            + (sin_p_theta * sin_rot_phi * cos_p_phi * cos_rot_theta)
            + (sin_rot_phi * sin_rot_theta * cos_p_theta),

            (-sin_p_phi * sin_p_theta * sin_rot_phi)
            + (sin_p_theta * cos_p_phi * cos_rot_phi * cos_rot_theta)
            + (sin_rot_theta * cos_p_theta * cos_rot_phi)
        )

        q_phi[i] = q_phi[i] % (2 * math.pi)


def fill_from_spher(s):
    """Fill in the remaining values in SPHER_T type giving the two angles `zen` and
    `az`.

    Parameters
    ----------
    s : SPHER_T

    """
    s['sinzen'] = np.sin(s['zen'])
    s['coszen'] = np.cos(s['zen'])
    s['sinaz'] = np.sin(s['az'])
    s['cosaz'] = np.cos(s['az'])
    s['x'] = s['sinzen'] * s['cosaz']
    s['y'] = s['sinzen'] * s['sinaz']
    s['z'] = s['coszen']


def fill_from_cart(s_vector):
    """Fill in the remaining values in SPHER_T type giving the cart, coords. `x`, `y`
    and `z`.

    Parameters
    ----------
    s_vector : SPHER_T

    """
    for s in s_vector:
        radius = np.sqrt(s['x']**2 + s['y']**2 + s['z']**2)
        if radius != 0:
            # make sure they're length 1
            s['x'] /= radius
            s['y'] /= radius
            s['z'] /= radius
            s['az'] = np.arctan2(s['y'], s['x']) % (2 * np.pi)
            s['coszen'] = s['z']
            s['zen'] = np.arccos(s['coszen'])
            s['sinzen'] = np.sin(s['zen'])
            s['sinaz'] = np.sin(s['az'])
            s['cosaz'] = np.cos(s['az'])
        else:
            s['z'] = 1
            s['az'] = 0
            s['zen'] = 0
            s['coszen'] = 1
            s['sinzen'] = 0
            s['cosaz'] = 1
            s['sinaz'] = 0


def reflect(old, centroid, new):
    """Reflect the old point around the centroid into the new point on the sphere.

    Parameters
    ----------
    old : SPHER_T
    centroid : SPHER_T
    new : SPHER_T

    """
    x = old['x']
    y = old['y']
    z = old['z']

    ca = centroid['cosaz']
    sa = centroid['sinaz']
    cz = centroid['coszen']
    sz = centroid['sinzen']

    new['x'] = (
        2*ca*cz*sz*z
        + x*(ca*(-ca*cz**2 + ca*sz**2) - sa**2)
        + y*(ca*sa + sa*(-ca*cz**2 + ca*sz**2))
    )
    new['y'] = (
        2*cz*sa*sz*z
        + x*(ca*sa + ca*(-cz**2*sa + sa*sz**2))
        + y*(-ca**2 + sa*(-cz**2*sa + sa*sz**2))
    )
    new['z'] = 2*ca*cz*sz*x + 2*cz*sa*sz*y + z*(cz**2 - sz**2)

    fill_from_cart(new)


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def add_vectors(r1, theta1, phi1, r2, theta2, phi2, r3, theta3, phi3):
    """Add two vectors v1 + v2 = v3 in spherical coordinates."""
    for i in range(len(r1)):  # pylint: disable=consider-using-enumerate
        x1 = r1[i] * math.sin(theta1[i]) * math.cos(phi1[i])
        y1 = r1[i] * math.sin(theta1[i]) * math.sin(phi1[i])
        z1 = r1[i] * math.cos(theta1[i])
        x2 = r2[i] * math.sin(theta2[i]) * math.cos(phi2[i])
        y2 = r2[i] * math.sin(theta2[i]) * math.sin(phi2[i])
        z2 = r2[i] * math.cos(theta2[i])
        x3 = x1 + x2
        y3 = y1 + y2
        z3 = z1 + z2
        r3[i] = math.sqrt(x3**2 + y3**2 + z3**2)
        theta3[i] = math.acos(z3/r3[i])
        phi3[i] = math.atan2(y3, x3) % (2 * math.pi)


if __name__ == '__main__':
    test_infer_power()
    test_linbin()
    test_powerbin()
    test_generate_digitizer()
