#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name

"""
Simplistic forward and Retro 2D simulations used to validate Retro
normalization.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    # -- Constants -- #
    'SPEED_OF_LIGHT_IN_VACUUM', 'DOM_RADIUS',

    # -- Helper functions -- #
    'sgnstar',

    # -- Primary functions of the module -- #
    'forward_survival_prob', 'retro_point_dom_survival_prob',
    'retro_finite_dom_survival_prob', 'test_survival_prob'
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
from time import time

import numpy as np


# -- Constants -- #

SPEED_OF_LIGHT_IN_VACUUM = 299792458 # m/s
DOM_RADIUS = 0.165 # m


# -- Helper functions -- #

def sgnstar(x):
    """Like `np.sign` but positive _and_ 0 values map to +1 """
    out = np.empty_like(x)
    mask = x < 0
    out[mask] = -1
    out[~ mask] = 1
    return out


def pick_nonzero_bin(binning, r_idx, phi_idx, dom_radius, speed_of_light):
    """Pick a bin from `binning` that will have non-zero survival probability.

    The algorithm simply looks at the midpoint in the r-bin and finds the
    corresponding t-bin from this; The phidir bin is chosen opposite from the
    phi bin.

    Parameters
    ----------
    binning : pisa.core.binning.MultiDimBinning
        Must contain r, t, phi, and phidir dimensions (in any order).

    r_idx : int
    phi_idx : int
    dom_radius : float, optional
        Units of meters

    speed_of_light : float, optional
        Speed of light in m/s

    Returns
    -------
    bin : pisa.core.binning.MultiDimBinning
    indices : tuple of 5 integers

    """
    r0 = binning.r.midpoints[r_idx].m_as('m')
    dr = r0 - dom_radius # m
    t0 = dr / speed_of_light * 1e9 # ns
    phi0 = binning.phi.midpoints[phi_idx].m_as('rad')
    phidir0 = (phi0 + np.pi) % (2*np.pi)

    point = np.array([[r0, phi0, t0, phidir0]])

    hist, _ = np.histogramdd(
        point,
        bins=binning.bin_edges,
        normed=False
    )
    indices = tuple(c[0] for c in np.where(hist == 1))
    return binning[indices], indices


# -- Primary functions of the module -- #

def forward_survival_prob(absorption_length, binning, n_photons,
                          dom_efficiency, dom_radius, speed_of_light, seed):
    """Simulate random "simple" photons (absorption but no scattering) within a
    bin, find how many hit the DOM, and from this estimate the survival
    probability for the bin.

    This is the "forward" simulation, which contrasts with "retro" simulation
    which starts photons at the DOM and records how many end up in the bin in
    order to estimate the same survival probability we compute here (i.e., the
    probability that a photon from the bin will be detected by the DOM).

    In addition to no scattering, the simulation is performed in the xy-plane
    and neither wavelength nor angular dependencies are taken into account.

    Parameters
    ----------
    absorption_length : float
        Units of m.

    binning : pisa.core.binning.MultiDimBinning
        Must contain r, t, phi, and phidir dimensions.

    n_photons : int
        Number of random photons to generate in the bin.

    speed_of_light : float
        Speed of light in the medium, in units of m/s

    dom_radius : float
        Units of m.

    dom_efficiency : float

    seed : int

    Returns
    -------
    pdet : float in [0, 1]
        Probability of a photon in the bin being detected by the DOM (in our
        simplistic model).

    num_hits : float
        Number of photons that hit the DOM. E.g., the geometric "norm" is
        num_hits / n_photons
        (a norm not weighted by absorption length or DOM efficiency)

    """
    dom_radius_sq = dom_radius**2 # m^2

    r_edges = sorted(binning.r.bin_edges.m_as('m'))
    t_edges = sorted(binning.t.bin_edges.m_as('ns'))
    phi_edges = sorted(binning.phi.bin_edges.m_as('rad'))
    phidir_edges = sorted(binning.phidir.bin_edges.m_as('rad'))

    rand = np.random.RandomState(seed)
    phi_samples = rand.uniform(phi_edges[0], phi_edges[1], n_photons)
    r_samples = np.sqrt(rand.uniform(r_edges[0]**2, r_edges[-1]**2, n_photons))
    x_samples = r_samples * np.cos(phi_samples)
    y_samples = r_samples * np.sin(phi_samples)

    phidir_samples = rand.uniform(phidir_edges[0], phidir_edges[-1], n_photons)

    # Test for intersection with the circular DOM.
    # See http://mathworld.wolfram.com/Circle-LineIntersection.html

    ray_distance = 10 * r_edges[-1] # m

    dx = ray_distance * np.cos(phidir_samples)
    dy = ray_distance * np.sin(phidir_samples)
    endpt_x = x_samples + dx
    endpt_y = y_samples + dy
    determinant = x_samples * endpt_y - endpt_x * y_samples
    dr_sq = dx**2 + dy**2
    discriminant = dom_radius_sq * dr_sq - determinant**2

    # Select values that intersect the DOM

    mask = discriminant >= 0
    num_hits = mask.sum()
    x = np.compress(mask, x_samples)
    y = np.compress(mask, y_samples)
    dx = np.compress(mask, dx)
    dy = np.compress(mask, dy)
    endpt_x = np.compress(mask, endpt_x)
    endpt_y = np.compress(mask, endpt_y)
    determinant = np.compress(mask, determinant)
    dr_sq = np.compress(mask, dr_sq)
    discriminant = np.compress(mask, discriminant)

    sqrt_discr = np.sqrt(discriminant)

    det_dy = determinant * dy
    neg_det_dx = - determinant * dx
    sgnstar_dx_sqrt_discr = sgnstar(dy) * dx * sqrt_discr
    abs_dy_sqrt_discr = np.abs(dy) * sqrt_discr

    x0_int = (det_dy + sgnstar_dx_sqrt_discr) / dr_sq
    x1_int = (det_dy - sgnstar_dx_sqrt_discr) / dr_sq

    y0_int = (neg_det_dx + abs_dy_sqrt_discr) / dr_sq
    y1_int = (neg_det_dx - abs_dy_sqrt_discr) / dr_sq

    dist0sq = (x - x0_int)**2 + (y - y0_int)**2
    dist1sq = (x - x1_int)**2 + (y - y1_int)**2
    mask = dist0sq < dist1sq

    distsq = np.where(mask, dist0sq, dist1sq)

    # Now select only those points that fall within the time bin

    dist = np.sqrt(distsq)
    transit_time = dist / speed_of_light * 1e9 # ns

    mask = (transit_time >= t_edges[0]) & (transit_time < t_edges[1])
    num_hits = np.sum(mask)

    dist = np.compress(mask, dist)
    pdet = (
        np.sum(dom_efficiency * np.exp(-dist / absorption_length)) / n_photons
    )

    return pdet, num_hits


def retro_point_dom_survival_prob(absorption_length, binning, n_photons,
                                  dom_efficiency, dom_radius, speed_of_light,
                                  step_length, seed):
    """Retro simulation to find photon survival probability, starting photons
    at a point located at the center of the DOM.

    Parameters
    ----------
    absorption_length : float in (0, np.inf]
        Units of m.

    binning : MultiDimBinning
        Must contain dimensions "r", "phi", "t", and "phidir"; units are
        converted internally so no need to specify these in particular units..

    n_photons : int

    dom_efficiency : float in [0, 1]

    dom_radius : float
        Units of m

    speed_of_light : float
        Units of m/s

    step_length : float
        Units of m

    seed : int in [0, 2**32)

    Returns
    -------
    unnormed_pdet
    num_hits

    """
    # Convert binning units to those expected internally
    units = dict(r='m', phi='rad', t='ns', phidir='rad')
    binning = binning.to(*[units[name] for name in binning.names])

    # TODO: use np.histogramdd to allow for arbitrary binning!
    assert binning.tot_num_bins == 1

    # Find limits of binning
    r_min, r_max = binning.r.domain.m
    phi_min, phi_max = binning.phi.domain.m
    t_min, t_max = binning.t.domain.m
    phidir_min, phidir_max = binning.phidir.domain.m

    rand = np.random.RandomState(seed)

    retro_phi_samples = rand.uniform(low=0, high=2*np.pi, size=n_photons)
    retro_phidir_samples = (retro_phi_samples + np.pi) % (2*np.pi)
    mask_phi = (retro_phi_samples >= phi_min) & (retro_phi_samples < phi_max)
    mask_phidir = (retro_phidir_samples >= phidir_min) & (retro_phidir_samples < phidir_max)
    mask = mask_phi & mask_phidir

    num_photons_in_bin = np.sum(mask)

    retro_phi_samples = np.compress(mask, retro_phi_samples)
    retro_phidir_samples = np.compress(mask, retro_phidir_samples)

    radial_samples_r0 = rand.uniform(
        low=r_min,
        high=r_min + step_length,
        size=num_photons_in_bin
    )

    radial_samples = []
    r_kept = 0
    t_kept = 0
    for r_step_idx in range(int((r_max - r_min) // step_length)):
        rsamp = radial_samples_r0 + r_step_idx * step_length
        mask = rsamp < r_max
        r_kept += np.sum(mask)
        rsamp = np.compress(mask, rsamp)
        tsamp = rsamp / speed_of_light * 1e9 # ns
        mask = (tsamp >= t_min) & (tsamp < t_max)
        t_kept += np.sum(mask)
        radial_samples.append(np.compress(mask, rsamp))

    radial_samples = np.concatenate(radial_samples)

    num_counts = radial_samples.size

    unnormed_pdet = np.sum(
        dom_efficiency * np.exp(-radial_samples / absorption_length)
    )

    return unnormed_pdet, num_photons_in_bin, num_counts


def retro_finite_dom_survival_prob(absorption_length, binning, n_photons,
                                   dom_efficiency, dom_radius, speed_of_light,
                                   step_length, seed):
    """Retro simulation to find photon survival probability, starting photons
    on the surface of the DOM.

    Starting angles of the photons are randomized to be between +/-pi/2 off of
    the DOM normal at the photon starting point.

    Parameters
    ----------
    absorption_length : float in (0, np.inf]
        Units of m.

    binning : MultiDimBinning
        Must contain dimensions "r", "phi", "t", and "phidir"; units are
        converted internally so no need to specify these in particular units..

    n_photons : int

    dom_efficiency : float in [0, 1]

    dom_radius : float
        Units of m.

    speed_of_light : float
        Units of m/s

    step_length : float
        Units of m

    seed : int in [0, 2**32)

    """
    # Convert binning units to those expected internally
    units = dict(r='m', phi='rad', t='ns', phidir='rad')
    binning = binning.to(*[units[name] for name in binning.names])
    r_max = np.max(binning.r.bin_edges.m)
    phidir_min, phidir_max = binning.phidir.domain.m

    rand = np.random.RandomState(seed)

    # Photon starting point in polar coords (radius is just DOM radius)
    start_phi_samp = rand.uniform(low=0, high=2*np.pi, size=n_photons)

    # Flip direction to be opposite of phi and randomize photon direction by
    # +/-pi/2 from the DOM normal
    phidir_samp = (
        rand.uniform(
            low=np.pi - np.pi/2,
            high=np.pi + np.pi/2,
            size=n_photons
        )
        + start_phi_samp
    ) % (2*np.pi)
    mask = (phidir_samp >= phidir_min) & (phidir_samp < phidir_max)
    n_phidir_keep = np.sum(mask)

    start_phi_samp = np.compress(mask, start_phi_samp)
    phidir_samp = np.compress(mask, phidir_samp)

    # Starting point in Cartesian coordinates
    x0 = dom_radius * np.cos(start_phi_samp)
    y0 = dom_radius * np.sin(start_phi_samp)

    displ_first_samp = rand.uniform(
        low=0,
        high=step_length,
        size=n_phidir_keep
    )

    displ_samp_offsets = np.arange(
        0,
        2*dom_radius + r_max + step_length,
        step_length
    )
    displ_samp = np.add.outer(displ_samp_offsets, displ_first_samp)

    # Note negatives are due to direction being opposite in Retro
    x_samp = x0 - displ_samp * np.cos(phidir_samp)
    y_samp = y0 - displ_samp * np.sin(phidir_samp)

    # Need polar coords relative to origin to find what's inside the binning
    r_samp = np.hypot(x_samp, y_samp)
    t_samp = displ_samp / (speed_of_light / 1e9)
    phi_samp = np.arctan2(y_samp, x_samp)

    r_samp = r_samp.flatten()
    t_samp = t_samp.flatten()
    phi_samp = phi_samp.flatten()

    samp_map = dict(r=r_samp, t=t_samp, phi=phi_samp)
    samp = [samp_map[name] for name in binning.names if name != 'phidir']

    bin_edges = OrderedDict(
        [(name, binning[name].bin_edges)
         for name in binning.names if name != 'phidir']
    )

    unnormed_pdet, _ = np.histogramdd(
        samp,
        bins=bin_edges.values(),
        weights=np.exp(-displ_samp.flatten() / absorption_length),
        normed=False
    )
    unnormed_pdet *= dom_efficiency

    num_counts, _ = np.histogramdd(samp, bins=bin_edges.values(), normed=False)

    return unnormed_pdet, num_counts


def test_survival_prob():
    """Run forward simulation over a few bins"""
    t_start = time()
    from pisa.core.binning import OneDimBinning

    absorption_length = 1 # m
    n_photons = int(1e6)
    speed_of_light = SPEED_OF_LIGHT_IN_VACUUM #/ 1.5 # m/s
    dom_radius = DOM_RADIUS
    dom_efficiency = 1
    seed = 0

    r_binning = OneDimBinning(
        name='r',
        domain=(0, 25),
        is_lin=True,
        num_bins=25,
        units='m'
    )
    phi_binning = OneDimBinning(
        name='phi',
        tex=r'\phi',
        domain=(0, 2*np.pi),
        is_lin=True,
        num_bins=32,
        units='rad'
    )
    t_binning = OneDimBinning(
        name='time',
        domain=(0, 100),
        is_lin=True,
        num_bins=10,
        units='ns'
    )
    phidir_binning = OneDimBinning(
        name='phidir',
        tex=r'\phi_{\rm dir}',
        domain=(0, 2*np.pi),
        is_lin=True,
        num_bins=96,
        units='rad'
    )
    binning = r_binning * phi_binning * t_binning * phidir_binning

    for r_idx in range(binning.r.num_bins):
        # Pick a bin that will have non-zero entries
        out_binning, _ = pick_nonzero_bin(
            binning=binning,
            r_idx=r_idx,
            phi_idx=0,
            dom_radius=dom_radius,
            speed_of_light=speed_of_light
        )
        #print('indices of bin0 in table:', indices0)
        #print('bin0:\n{}'.format(bin0))

        #indices0 = np.array(indices0)

        #r0_idx, phi0_idx, t0_idx, phidir0_idx = indices0

        # Get probs for nominal r and t bins +/- 2 bins away
        #r_slice = slice(r0_idx - 2, r0_idx + 3)
        #t_slice = slice(t0_idx - 2, t0_idx + 3)
        #r_slice = slice(None)
        #t_slice = slice(None)
        #out_binning = binning[r_slice, phi0_idx, t_slice, phidir0_idx + 2]

        survival_probs = out_binning.empty(name='survival_prob')
        geom_norms = out_binning.empty(name='geom_norm')
        for bin_flat_idx, this_bin in enumerate(out_binning.iterbins()):
            prob, num_hits = forward_survival_prob(
                absorption_length=absorption_length,
                binning=this_bin,
                n_photons=n_photons,
                dom_efficiency=dom_efficiency,
                dom_radius=dom_radius,
                speed_of_light=speed_of_light,
                seed=seed
            )
            #print('')
            survival_probs[out_binning.index2coord(bin_flat_idx)] = prob
            geom_norms[out_binning.index2coord(bin_flat_idx)] = num_hits / n_photons

    print('Total time to run:', time() - t_start)

    return survival_probs, geom_norms, binning


if __name__ == '__main__':
    survival_probs, geom_norms, binning = test_survival_prob() # pylint: disable=invalid-name
