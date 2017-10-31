# coding: utf-8
# pylint: disable=wrong-import-position

"""
Generate single-DOM Retro tables binned in (t,r,theta).
"""


from __future__ import absolute_import, division, print_function

import numpy as np


__all__ = ['generate_t_r_theta_table']


#@numba.jit(**DFLT_NUMBA_JIT_KWARGS)
def generate_t_r_theta_table(data, n_photons, p_theta_centers,
                             p_delta_phi_centers, theta_bin_edges):
    """Transform information from a raw single-DOM table (as output from CLSim)
    into a more compact representation, with probability and average direction
    (theta and phi) binned in (t, r, theta).

    Parameters
    ----------
    data
    n_photons
    p_theta_centers
    p_delta_phi_centers
    theta_bin_edges

    Returns
    -------
    n_photons
    average_thetas
    average_phis
    lengths

    """
    average_thetas = np.empty_like(n_photons)
    average_phis = np.empty_like(n_photons)
    lengths = np.empty_like(n_photons)

    for i in xrange(n_photons.shape[0]):
        for j in xrange(n_photons.shape[1]):
            for k in xrange(n_photons.shape[2]):
                # flip coszen?
                weights = data[i, j, k, ::-1, :]
                if weights.sum() == 0:
                    # If no photons, just set the average direction to the
                    # theta of the bin center...
                    average_theta = 0.5 * (theta_bin_edges[j]
                                           + theta_bin_edges[j + 1])
                    # ... and lengths to 0
                    length = 0.
                    average_phi = 0.
                else:
                    # Average theta
                    weights_theta = np.sum(weights, axis=1)
                    average_theta = np.average(p_theta_centers,
                                               weights=weights_theta)

                    # Average delta phi
                    projected_n_photons = (
                        weights * np.sin(p_theta_centers)[:, np.newaxis]
                    )
                    weights_phi = np.sum(projected_n_photons, axis=0)
                    average_phi = np.average(p_delta_phi_centers,
                                             weights=weights_phi)

                    # Length of vector (using projections from all vectors
                    # onto average vector cos(angle) between average vector
                    # and all angles)
                    coscos = np.cos(p_theta_centers)*np.cos(average_theta)
                    sinsin = np.sin(p_theta_centers)*np.sin(average_theta)
                    cosphi = np.cos(p_delta_phi_centers - average_phi)
                    # Other half of sphere
                    cospsi = coscos[:, np.newaxis] + np.outer(sinsin, cosphi)
                    length = max(
                        0,
                        2 * (np.average(cospsi, weights=weights) - 0.5)
                    )

                average_thetas[i, j, k] = average_theta
                average_phis[i, j, k] = average_phi
                lengths[i, j, k] = length

    # Invert tables (r, cz, t) -> (-t, r, cz) and also flip coszen binning
    n_photons = np.flipud(np.rollaxis(np.fliplr(n_photons), 2, 0))
    average_thetas = np.flipud(np.rollaxis(np.fliplr(average_thetas), 2, 0))
    average_phis = np.flipud(np.rollaxis(np.fliplr(average_phis), 2, 0))
    lengths = np.flipud(np.rollaxis(np.fliplr(lengths), 2, 0))

    return n_photons, average_thetas, average_phis, lengths
