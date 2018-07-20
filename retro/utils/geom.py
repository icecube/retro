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
    'generate_digitizer',
    'inv_power_2nd_diff',
    'infer_power',
    'test_infer_power',
    'sample_powerlaw_binning',
    'bin_edges_to_binspec',
    'linear_bin_centers',
    'spacetime_separation',
    'spherical_volume',
    'sph2cart',
    'pol2cart',
    'cart2pol',
    'cart2sph',
    'rotsph2cart',
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
from retro import DFLT_NUMBA_JIT_KWARGS, load_pickle, numba_jit
from retro.const import SPEED_OF_LIGHT_M_PER_NS
from retro.retro_types import TimeSphCoord
from retro.utils.misc import hash_obj


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
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    out_type

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    width = (stop - start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.uint32)
    bin_num_flat = bin_num.flat
    for i, v in enumerate(val.flat):
        bin_num_flat[i] = (v - start) // width
    return bin_num


def _linbin_numpy(val, start, stop, num):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    width = (stop - start) / num_bins
    bin_num = (val - start) // width
    #if np.isscalar(bin_num):
    #    bin_num = int(bin_num)
    #else:
    #    bin_num =
    return bin_num


linbin = _linbin_numba # pylint: disable=invalid-name


def test_linbin():
    """Unit tests for function `linbin`."""
    kw = dict(start=0, stop=100, num=200)
    bin_edges = np.linspace(**kw)

    rand = np.random.RandomState(seed=0)
    x = rand.uniform(0, 100, int(1e6))

    test_args = (np.array([0.0]), np.float64(kw['start']),
                 np.float64(kw['stop']), np.uint32(kw['num']))
    _linbin_numba(*test_args)

    test_args = (x, np.float64(kw['start']), np.float64(kw['stop']),
                 np.uint32(kw['num']))
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

    assert np.all(bins_test_numba == bins_ref)
    assert np.all(bins_test_numpy == bins_ref)
    #assert isinstance(_linbin_numpy(1, **kw), int)

    print('<< PASS : test_linbin >>')


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def _powerbin_numba(val, start, stop, num, power): #, out_type=np.uint64):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : np.ndarray
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float
    out_type

    Returns
    -------
    bin_num : np.ndarray of dtype `out_type`

    """
    num_bins = num - 1
    inv_power = 1.0 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = np.empty(shape=val.shape, dtype=np.uint32)
    bin_num_flat = bin_num.flat
    for i, v in enumerate(val.flat):
        bin_num_flat[i] = (v**inv_power - inv_power_start) // inv_power_width
    return bin_num


def _powerbin_numpy(val, start, stop, num, power):
    """Determine the bin number(s) in a powerspace binning of value(s).

    Parameters
    ----------
    val : scalar or array
    start : float
    stop : float
    num : int
        Number of bin _edges_ (number of bins is one less than `num`)
    power : float

    Returns
    -------
    bin_num : int or np.ndarray of dtype int
        If `val` is scalar, returns int; if `val` is a sequence or array-like,
        returns `np.darray` of dtype int.

    """
    num_bins = num - 1
    inv_power = 1 / power
    inv_power_start = start**inv_power
    inv_power_stop = stop**inv_power
    inv_power_width = (inv_power_stop - inv_power_start) / num_bins
    bin_num = (val**inv_power - inv_power_start) // inv_power_width
    if np.isscalar(bin_num):
        bin_num = int(bin_num)
    else:
        bin_num = bin_num.astype(int)
    return bin_num


powerbin = _powerbin_numpy # pylint: disable=invalid-name


#def powerbin(val, start, stop, num, power):
#    """Determine the bin number(s) in a powerspace binning of value(s).
#
#    Parameters
#    ----------
#    val : scalar or array
#    start : float
#    stop : float
#    num : int
#        Number of bin _edges_ (number of bins is one less than `num`)
#    power : float
#
#    Returns
#    -------
#    bin_num : int or np.ndarray of dtype int
#        If `val` is scalar, returns int; if `val` is a sequence or array-like,
#        returns `np.darray` of dtype int.
#
#    """
#    if np.isscalar(val):
#        val = np.array(val)
#    else:
#        val = np.asarray(val)
#
#    if num < 1000:
#        pass


def test_powerbin():
    """Unit tests for function `powerbin`."""
    kw = dict(start=0, stop=100, num=100, power=2)
    bin_edges = powerspace(**kw)

    rand = np.random.RandomState(seed=0)
    x = rand.uniform(0, 100, int(1e6))

    ftype = np.float32
    utype = np.uint32
    test_args = (ftype(kw['start']), ftype(kw['stop']),
                 utype(kw['num']), utype(kw['power']))

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

    assert np.all(bins_test_numba == bins_ref), str(bins_test_numba) + '\n' + str(bins_ref)
    assert np.all(bins_test_numpy == bins_ref), str(bins_test_numpy) + '\n' + str(bins_ref)
    #assert isinstance(_powerbin_numpy(1, **kw), int)

    print('<< PASS : test_powerbin >>')


def generate_digitizer(bin_edges):
    """Factory to generate a specialized Numba function for "digitizing" data
    (i.e., returning which bin values fall within).

    Parameters
    ----------
    bin_edges : array-like

    Returns
    -------
    digitize : callable

    Notes
    -----
    The digitizer returned does not fail if a value lies outside the
    binning boundaries. This condition is indicated by returned indices that
    are either negative or > num_bins. Therefore a check can be performed on
    the returned index to check for this, e.g. via .. ::

        idx = digitize(val)
        if idx < 0 or idx >= num_bins:
            raise ValueError('val outside binning')

    """
    # pylint: disable=missing-docstring, function-redefined
    bin_edges = np.asarray(bin_edges)
    assert np.all(np.diff(bin_edges) > 0)
    start = bin_edges[0]
    stop = bin_edges[-1]
    num_bin_edges = len(bin_edges)
    num_bins = num_bin_edges - 1

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
        def digitize(val):
            if val < start:
                return -1
            if val > stop:
                return num_bins
            return int((val - start) * recip_dx)

    elif power:
        bindescr = (
            '{} bins spaced from {} to {} spaced with power of {}'
            .format(num_bins, start, stop, power)
        )
        if np.isclose(power, 2):
            def digitize(val):
                if val < start:
                    return -1
                if val > stop:
                    return num_bins
                return int((math.sqrt(val) - start_recip_power) * recip_power_width)

        elif num_bins > 1e3: # faster to do binary search if fewer bins
            def digitize(val):
                if val < start:
                    return -1
                if val > stop:
                    return num_bins
                return int((val**recip_power - start_recip_power) * recip_power_width)

    elif is_log:
        bindescr = (
            '{} bins logarithmically spaced from {} to {}'.format(num_bins, start, stop)
        )
        if num_bins > 20: # faster to do binary search if fewer bins
            def digitize(val):
                if val < start:
                    return -1
                if val > stop:
                    return num_bins
                return int((math.log(val) - log_start) * recip_logwidth)

    if bindescr is None:
        bindescr = (
            '{} bins unevenly spaced from {} to {}'.format(num_bins, start, stop)
        )

    if digitize is None:
        def digitize(val):
            if val < start:
                return -1
            if val > stop:
                return num_bins
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
    digitize = numba_jit(fastmath=True, nogil=True, cache=True)(digitize)

    print(bindescr)

    return digitize


def test_generate_digitizer():
    """Test the functions that `generate_digitizer` produces."""
    # TODO: use local file for this test
    meta = load_pickle('/home/icecube/retro/tables/large_5d_notilt_combined/stacked/stacked_ckv_template_map_meta.pkl')
    binning = meta['binning']

    for dim, edges in binning.items():
        assert np.all(np.diff(edges) > 0)
        num_bins = len(edges) - 1
        digitize = generate_digitizer(edges)
        rand = np.random.RandomState(0)

        # Check lots of values within the valid range of the binning
        vals = rand.uniform(low=edges[0], high=edges[-1], size=int(1e5))
        test = np.array([digitize(v) for v in vals])
        ref = np.digitize(vals, bins=edges, right=False) - 1
        assert np.all(test == ref), dim

        # Check edge cases
        assert digitize(edges[0]) == 0, dim
        assert digitize(edges[0] - 1e-8) < 0, dim
        assert digitize(edges[-1]) > num_bins, dim
        assert digitize(edges[-1] + 1e-8) > num_bins, dim

    print('<< PASS : test_generate_digitizer >>')


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


def infer_power(edges, dtype=np.float64):
    """Infer the power used for bin edges evenly spaced w.r.t. ``x**power``."""
    first_three_edges = edges[:3]
    atol = 1e-15
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


if __name__ == '__main__':
    test_infer_power()
    test_linbin()
    test_powerbin()
    test_generate_digitizer()
