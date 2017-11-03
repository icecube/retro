# coding: utf-8
# pylint: disable=wrong-import-position

"""
Generate single-DOM Retro tables binned in (t,r,theta).
"""


from __future__ import absolute_import, division, print_function

import numpy as np

import numba


__all__ = ['generate_t_r_theta_table']


@numba.jit(nopython=True, nogil=True, cache=True)
def weighted_average(x, w):
    """Average of elements in `x` weighted by `w`.

    Parameters
    ----------
    x : numpy.ndarray
        Values to average

    w : numpy.ndarray
        Weights, same shape as `x`

    Returns
    -------
    avg : numpy.ndarray
        Weighted average, same shape as `x`

    """
    sum_xw = 0.0
    sum_w = 0.0
    for x_i, w_i in zip(x, w):
        sum_xw += x_i * w_i
        sum_w += w_i
    return sum_xw / sum_w


#def showarray(x):
#    print(','.join(format(x_, '0.15e') for x_ in x))


@numba.jit(nopython=True, nogil=True, cache=True)
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
    # Destination tables are to be binned in (t, r, costheta) (there are as
    # many costheta bins as theta bins in the original tables)
    ## OLD >>>>
    #average_thetas = np.empty_like(n_photons)
    #average_phis = np.empty_like(n_photons)
    #lengths = np.empty_like(n_photons)
    ## <<<< OLD

    n_r_bins = n_photons.shape[0]
    n_theta_bins = n_photons.shape[1]
    n_t_bins = n_photons.shape[2]

    # Source tables are photon counts binned in
    # (r, theta, t, dir_theta, dir_phi)
    # NEW >>>>
    dest_shape = (n_t_bins, n_r_bins, n_theta_bins)
    n_photons1 = np.empty(dest_shape, dtype=np.float32)
    average_thetas1 = np.empty(dest_shape, dtype=np.float32)
    average_phis1 = np.empty(dest_shape, dtype=np.float32)
    lengths1 = np.empty(dest_shape, dtype=np.float32)
    # <<<< NEW

    for r_i in range(n_r_bins):
        for theta_j in range(n_theta_bins):
            for t_k in range(n_t_bins):
                # flip coszen?
                weights = data[r_i, theta_j, t_k, ::-1, :].astype(np.float64)
                weights_tot = weights.sum()
                if weights_tot == 0:
                    # If no photons, just set the average direction to the
                    # theta of the bin center...
                    average_theta1 = 0.5 * (theta_bin_edges[theta_j]
                                            + theta_bin_edges[theta_j + 1])
                    # ... and lengths to 0
                    length1 = 0.0
                    average_phi1 = 0.0
                else:
                    # Average theta
                    weights_theta = weights.sum(axis=1)
                    ## OLD >>>>
                    #average_theta = np.average(p_theta_centers,
                    #                           weights=weights_theta)
                    ## OLD <<<<
                    # NEW >>>>
                    average_theta1 = weighted_average(p_theta_centers,
                                                      weights_theta)
                    #if not np.allclose(average_theta1, average_theta, equal_nan=True):
                    #    print('p_theta_centers:')
                    #    showarray(p_theta_centers)
                    #    print('weights_theta:')
                    #    showarray(weights_theta)
                    #    print('average_theta:', format(average_theta, '0.15e'))
                    #    print('average_theta1:', format(average_theta1, '0.15e'))
                    #    sys.exit()
                    # NEW <<<<

                    # Average delta phi

                    ## OLD >>>>
                    #projected_n_photons = (
                    #    weights * np.sin(p_theta_centers)[:, np.newaxis]
                    #)
                    #weights_phi = projected_n_photons.sum(axis=0)
                    #average_phi = np.average(p_delta_phi_centers,
                    #                         weights=weights_phi)
                    ## OLD <<<<
                    # NEW >>>>
                    projected_n_photons1 = (
                        (weights.T * np.sin(p_theta_centers)).T
                    )
                    weights_phi1 = projected_n_photons1.sum(axis=0)
                    average_phi1 = weighted_average(p_delta_phi_centers,
                                                    weights_phi1)
                    #if not np.allclose(average_phi1, average_phi, equal_nan=True):
                    #    print('p_delta_phi_centers:')
                    #    showarray(p_delta_phi_centers)
                    #    print('weights_phi:')
                    #    showarray(weights_phi)
                    #    print('weights_phi1:')
                    #    showarray(weights_phi1)
                    #    print('average_phi:', format(average_phi, '0.15e'))
                    #    print('average_phi1:', format(average_phi1, '0.15e'))
                    #    sys.exit()
                    # NEW <<<<

                    # Length of vector (using projections from all vectors
                    # onto average vector cos(angle) between average vector
                    # and all angles)
                    coscos = np.cos(p_theta_centers)*np.cos(average_theta1)
                    sinsin = np.sin(p_theta_centers)*np.sin(average_theta1)
                    cosphi = np.cos(p_delta_phi_centers - average_phi1)
                    # Other half of sphere
                    ## OLD >>>>
                    #cospsi = coscos[:, np.newaxis] + np.outer(sinsin, cosphi)
                    #cospsi_avg = np.average(cospsi, weights=weights)
                    #length = max(0, 2 * (cospsi_avg - 0.5))
                    ## OLD <<<<
                    # NEW >>>>
                    cospsi1 = (coscos + np.outer(sinsin, cosphi).T).T
                    cospsi_avg1 = (cospsi1 * weights).sum() / weights_tot
                    length1 = max(0.0, 2 * (cospsi_avg1 - 0.5))
                    #if not np.allclose(length1, length, equal_nan=True):
                    #    print('weights:')
                    #    print(repr(weights))
                    #    print('weights_tot:', weights_tot)
                    #    print('cospsi:')
                    #    print(repr(cospsi))
                    #    print('cospsi1:')
                    #    print(repr(cospsi1))
                    #    print('cospsi_avg:', cospsi_avg)
                    #    print('cospsi_avg1:', cospsi_avg1)
                    #    print('length:', length)
                    #    print('length1:', length1)
                    #    sys.exit()
                    # NEW <<<<

                ## OLD >>>>
                #average_thetas[r_i, theta_j, t_k] = average_theta1
                #average_phis[r_i, theta_j, t_k] = average_phi1
                #lengths[r_i, theta_j, t_k] = length1
                ## <<<< OLD

                # Output tables are expected to be in (flip(t), r, costheta).
                # In addition to time being flipped, coszen is expected to be
                # ascending, and therefore its binning is also flipped as
                # compared to the theta binning in the original.
                # NEW >>>>
                dest_bin = (
                    n_t_bins - 1 - t_k,
                    r_i,
                    n_theta_bins - 1 - theta_j
                )

                n_photons1[dest_bin] = n_photons[r_i, theta_j, t_k]
                average_thetas1[dest_bin] = average_theta1
                average_phis1[dest_bin] = average_phi1
                lengths1[dest_bin] = length1
                # NEW <<<<

    ## Invert tables (r, cz, t) -> (-t, r, cz) and also flip coszen binning
    ## OLD >>>>
    #n_photons = np.flipud(np.rollaxis(np.fliplr(n_photons), 2, 0))
    #average_thetas = np.flipud(np.rollaxis(np.fliplr(average_thetas), 2, 0))
    #average_phis = np.flipud(np.rollaxis(np.fliplr(average_phis), 2, 0))
    #lengths = np.flipud(np.rollaxis(np.fliplr(lengths), 2, 0))
    ## <<<< OLD

    #if not np.allclose(n_photons1, n_photons):
    #    print('n_photons:')
    #    print(repr(n_photons))
    #    print('n_photons1:')
    #    print(repr(n_photons1))
    #    sys.exit()

    return n_photons1, average_thetas1, average_phis1, lengths1
