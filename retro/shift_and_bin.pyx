cimport cython

cimport openmp
from cython.parallel import parallel, prange

from libc.math cimport ceil, floor, round, sqrt

import numpy as np
cimport numpy as np


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def shift_and_bin(list ind_arrays,
                  list vol_arrays,
                  np.ndarray dom_coords,
                  float[:, :] survival_prob,
                  float[:, :] prho,
                  float[:, :] pz,
                  unsigned int nr,
                  unsigned int ntheta,
                  double[:, :, :] binned_spv,
                  double[:, :, :] binned_px_spv,
                  double[:, :, :] binned_py_spv,
                  double[:, :, :] binned_pz_spv,
                  double[:, :, :] binned_one_minus_sp,
                  unsigned int nx,
                  unsigned int ny,
                  unsigned int nz,
                  float x0,
                  float y0,
                  float z0,
                  float xbw,
                  float ybw,
                  float zbw,
                  unsigned int x_oversample,
                  unsigned int y_oversample,
                  unsigned int z_oversample):
    r"""Shift (r, theta) retro tables (i.e., (t, r, theta) tables with time
    marginalized out) to each DOM location and aggregate its quantities (with
    appropriate weighting) in (x, y, z) retro tables.

    Note that the results are aggregated in the input arrays (which are assumed
    to be appropriately instantiated), so there are no return values.

    The convention used is that DOMs are indexed by `i`, polar bins are indexed
    by `j` (flattened from 2D to 1D), and Cartesian bins are indexed by `k`
    (flattened from 3D to 1D). There are a total of `I` DOMs and each is
    assumed to have the same number of polar bins `J = J_r \times J_\theta`
    (where `J_r` and `J_\theta` are the numbers of bins in the radial and
    azimuthal directions, respectively). Each polar bin `j` maps to `K_j`
    Cartesian bins (n.b. this number can be different for each polar bin).

    There are a total of `N_x`, `N_y`, and `N_z` Cartesian bins in x-, y-, and
    z-directions, respectively.

    See Notes for more detailed description of the process carried out by this
    function.


    Parameters
    ----------
    ind_arrays : length-J list of shape (K_j, 3) numpy.ndarrays, dtype float32
        Each 3-element row contains the x-, y-, and z-coordinates of a
        Cartesian bin onto which the polar bin maps. There is one ind_array per
        polar bin in the first octant.

    vol_arrays : length-J list of shape (K_j,) numpy.ndarrays, dtype float32
        Volume from the polar bin to be applied to each of the N Cartesian
        bins. There is one vol_array per polar bin in the first octant.

    dom_coords : shape (I, 3) numpy.ndarray, dtype float32
        Each 3-element row represents a DOM x-, y, and z-coordinate at which to
        apply the photon info.

    survival_prob : shape (J_r, J_theta) numpy.ndarray, dtype float32
        Survival probability of a photon at this polar coordinate

    px, py, pz : shape (J_r, J_theta) numpy.ndarrays, dtype float32
        Average photon x-, y-, and z-components at this coordinate

    binned_spv : numpy.ndarray, same shape as `x`, `y`, and `z`
        Binned photon survival probabilities * volumes, accumulated for all
        DOMs to normalize the average surviving photon info (`binned_px_spv`,
        etc.) in the end

    binned_one_minus_sp : numpy.ndarray of shape (nx, ny, nz)
        Existing array to which ``1 - normed_survival_probability`` is
        multiplied (where the normalization factor is not infinite)

    binned_px_spv, binned_py_spv, binned_pz_spv : shape (nx, ny, nz) numpy.ndarray, dtype float64
        Existing arrays into which average photon components are accumulated

    nx, ny, nz : unsigned int
    x0, y0, z0 : float
    xbw, ybw, zbw : float
    x_oversample, y_oversample, z_oversample : unsigned int


    Notes
    -----
    For DOMs labeled `i`, each of which is characterized in polar bins `j`, we
    wish to find the "aggregated" survival probability `sp` and photon average
    behavior (`px`, `py`, and `pz`) in the Cartesian bins `k`.

    Weight survival probability by volume of overlap
        spv_{ijk} = sp_{ij} * v_{ijk}

    Weight average photon by volume of overlap times survival probability
        px_spv_{ijk} = px_{ij} * spv_{ijk}
        py_spv_{ijk} = py_{ij} * spv_{ijk}
        pz_spv_{ijk} = pz_{ij} * spv_{ijk}

    Total weighted survival probability for DOM `i`, Cartesian bin `k`
        spv_{ik} = \sum_j spv_{ijk}

    Normalization factor used for weighted survival probability contribution of
    DOM `i` to Cartesian bin `k' is just the sum of the overlap volumes.
        v_{ik} = \sum_j v_{ijk}

    Normalize survival probability weighted average before aggregating with
    other DOMs since, from DOM-to-DOM, this is _not_ a weighted average but
    rather a product (i.e., it's only a weighted average for polar bins that
    contribute to it for one DOM)
        sp_{ik} = spv_{ik} / v_{ik}

    Normalization factor for average photon; application of _this_ norm
    constant can be delayed until all DOM weighted avg photons have been
    aggregated since the final value is a simple weighted average. Note that
    only a single value in the following sum is added in this function, since it
        spv_k = \sum_i spv_{ik}

    In the end, one must compute the final survival probability and average
    photon in bin `k`. Note these operations are _not_ performed in this
    function since this function handles only a single type of DOM (i.e., a
    single source retro table, even if it is applied to multiple DOMs), and it
    is assumed that other types of DOMs must be aggregated before the final
    values are computed.
        sp_k = 1 - \prod_i (1 - sp_{ik})
        px_k = (1/spv_k) \cdot \sum_{ij} px_spv_{ijk}
        py_k = (1/spv_k) \cdot \sum_{ij} py_spv_{ijk}
        pz_k = (1/spv_k) \cdot \sum_{ij} pz_spv_{ijk}

    Identifying each `k` given a DOM `i` and polar bin `j` entails shifting
    (and possibly negating and/or swapping) the pre-calculated first-octant
    polar-to-Cartesian indices mapping (which assume they're centered at the
    origin) by the location of the DOM. The negation and/or swapping is
    performed to map to octants other than the first. The pre-calculated
    indices mapping is found in `sphbin2cartbin.ipynb`.

    See Also
    --------
    sphbin2cartbin.ipynb
        Computes the mapping from each polar bin in the first octant to the
        indices of the Cartesian bins it overlaps and volume of overlap, and
        stores these to disk for use by this function.

    """
    DEF PI = 3.1415926535897932384626433827
    DEF TWO_PI = 6.2831853071795862319959269371
    DEF PI_BY_2 = 1.5707963267948965579989817343

    # Logic below about extrapolating from first octant to other octants fails
    # if bin widths are different sizes in different dimensions
    assert xbw == ybw == zbw
    assert x_oversample == y_oversample == z_oversample

    cdef:
        float[:, :] ind_array
        float[:] vol_array

        Py_ssize_t num_first_octant_pol_bins = len(vol_arrays)
        Py_ssize_t num_cart_bins = nx * ny * nz

        float x_half_os_bw = xbw / (2 * <float>x_oversample)
        float y_half_os_bw = ybw / (2 * <float>y_oversample)
        float z_half_os_bw = zbw / (2 * <float>z_oversample)

        float x_mirror_pt = -1.0 / <float>x_oversample
        float y_mirror_pt = -1.0 / <float>y_oversample
        float z_mirror_pt = -1.0 / <float>z_oversample

        float dom_x, dom_y, dom_z
        float dom_x_float_idx, dom_y_float_idx, dom_z_float_idx
        float x, y, z, x_float_idx, y_float_idx, z_float_idx
        float bin_pos_rho_norm

        double vol, prho_, px_, py_, pz_
        double px_nom, py_nom
        double sp, spv, px_spv, py_spv, pz_spv

        unsigned int octant
        unsigned int ntheta_in_quad = <unsigned int>ceil(<double>ntheta / 2.0)
        Py_ssize_t flat_pol_idx, r_idx, theta_idx, theta_idx_
        int x_idx, y_idx, z_idx, ix
        unsigned int[:] num_cart_bins_in_pol_bin = np.empty(num_first_octant_pol_bins, dtype=np.uint32)
        unsigned int hemisphere, quadrant

        unsigned int[:, :, :] vol_mask = np.zeros((nx, ny, nz), dtype=np.uint32)
        double[:, :, :] binned_vol = np.zeros((nx, ny, nz), dtype=np.float64)

        unsigned int dom_idx
        unsigned int num_doms = <unsigned int>dom_coords.shape[0]

    assert num_first_octant_pol_bins == nr * ntheta / 2

    for ix in range(num_first_octant_pol_bins):
        num_cart_bins_in_pol_bin[ix] = vol_arrays[ix].shape[0]
        #vol_array_pointers[ix] = vol_arrays[ix]
        #ind_array_pointers[ix] = ind_arrays[ix]

    with nogil, parallel():
        for dom_idx in prange(num_doms, schedule='static'):
            with gil:
                (dom_x, dom_y, dom_z) = dom_coords[dom_idx]
            dom_x_float_idx = (dom_x - x0) / xbw
            dom_y_float_idx = (dom_y - y0) / ybw
            dom_z_float_idx = (dom_z - z0) / zbw
            for r_idx in range(nr):
                for theta_idx in range(ntheta / 2):
                    flat_pol_idx = theta_idx + r_idx*ntheta_in_quad

                    #with gil:
                    #    ind_array = ind_arrays[flat_pol_idx]
                    #    vol_array = vol_arrays[flat_pol_idx]

                    for ix in range(num_cart_bins_in_pol_bin[flat_pol_idx]):
                        #vol = <double>vol_array[ix]
                        #x_float_idx = ind_array[ix, 0]
                        #y_float_idx = ind_array[ix, 1]
                        #z_float_idx = ind_array[ix, 2]
                        with gil:
                            vol = <double>vol_arrays[flat_pol_idx][ix]

                            x_float_idx = ind_arrays[flat_pol_idx][ix, 0]
                            y_float_idx = ind_arrays[flat_pol_idx][ix, 1]
                            z_float_idx = ind_arrays[flat_pol_idx][ix, 2]

                        x = x_float_idx * xbw + x_half_os_bw
                        y = y_float_idx * ybw + y_half_os_bw
                        bin_pos_rho_norm = 1 / sqrt(x*x + y*y)
                        x = x * bin_pos_rho_norm
                        y = y * bin_pos_rho_norm

                        for hemisphere in range(2):
                            if hemisphere == 0:
                                z_idx = <int>round(z_float_idx + dom_z_float_idx)
                                theta_idx_ = theta_idx
                            else:
                                z_idx = <int>round(z_mirror_pt - z_float_idx + dom_z_float_idx)
                                theta_idx_ = ntheta - 1 - theta_idx

                            if z_idx < 0 or z_idx >= nz:
                                continue

                            sp = <double>survival_prob[r_idx, theta_idx_]
                            spv = sp * vol
                            prho_ = <double>prho[r_idx, theta_idx_]
                            pz_ = <double>pz[r_idx, theta_idx_]

                            px_nom = x * prho_
                            py_nom = y * prho_

                            for quadrant in range(4):
                                if quadrant == 0:
                                    x_idx = <int>round(x_float_idx + dom_x_float_idx)
                                    y_idx = <int>round(y_float_idx + dom_y_float_idx)
                                    px_ = px_nom
                                    py_ = py_nom

                                # x -> +y, y -> -x
                                elif quadrant == 1:
                                    x_idx = <int>round(y_mirror_pt - y_float_idx + dom_x_float_idx)
                                    y_idx = <int>round(x_float_idx + dom_y_float_idx)
                                    px_ = -py_nom
                                    py_ = px_nom

                                # x -> -x, y -> -y
                                elif quadrant == 2:
                                    x_idx = <int>round(x_mirror_pt - x_float_idx + dom_x_float_idx)
                                    y_idx = <int>round(y_mirror_pt - y_float_idx + dom_y_float_idx)
                                    px_ = -px_nom
                                    py_ = -py_nom

                                # x -> -y, y -> x
                                elif quadrant == 3:
                                    x_idx = <int>round(y_float_idx + dom_x_float_idx)
                                    y_idx = <int>round(x_mirror_pt - x_float_idx + dom_y_float_idx)
                                    px_ = py_nom
                                    py_ = -px_nom

                                if x_idx < 0 or x_idx >= nx or y_idx < 0 or y_idx >= ny:
                                    continue

                                with gil:
                                    vol_mask[x_idx, y_idx, z_idx] = 1
                                    binned_vol[x_idx, y_idx, z_idx] += vol
                                    binned_spv[x_idx, y_idx, z_idx] += spv
                                    binned_px_spv[x_idx, y_idx, z_idx] += px_ * spv
                                    binned_py_spv[x_idx, y_idx, z_idx] += py_ * spv
                                    binned_pz_spv[x_idx, y_idx, z_idx] += pz_ * spv

            # Normalize the weighted sum of survival probabilities for this DOM and
            # then include it in the overall survival probability via probabilistic
            # "or" statement:
            #     P(A) or P(B) = 1 - (1 - P(A)) * (1 - P(B))
            # though we stop at just the two factors on the right and more
            # probabilities can be easily combined before being subtracted form one
            # to yield the overall probability.
            for x_idx in range(nx): #, nogil=True, schedule='guided'):
                for y_idx in range(ny):
                    for z_idx in range(nz):
                        if vol_mask[x_idx, y_idx, z_idx] == 0:
                            continue
                        with gil:
                            binned_one_minus_sp[x_idx, y_idx, z_idx] *= (
                                1 - binned_spv[x_idx, y_idx, z_idx] / binned_vol[x_idx, y_idx, z_idx]
                            )
