cimport cython

from libc.stdlib cimport free, malloc
from libc.math cimport abs, ceil, log, round, sqrt

import numpy as np
cimport numpy as np


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def shift_and_bin(list ind_arrays,
                  list vol_arrays,
                  np.ndarray[double, ndim=2] dom_coords,
                  double[:, :] survival_prob,
                  double[:, :] prho,
                  double[:, :] pz,
                  int nr,
                  int ntheta,
                  double r_max,
                  double[:] binned_spv,
                  double[:] binned_px_spv,
                  double[:] binned_py_spv,
                  double[:] binned_pz_spv,
                  double[:] binned_log_one_minus_sp,
                  double x_min,
                  double x_max,
                  double y_min,
                  double y_max,
                  double z_min,
                  double z_max,
                  double binwidth,
                  int oversample,
                  anisotropy):
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

    There are a total of `nx`, `ny`, and `nz` Cartesian bins in x-, y-, and
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

    dom_coords : shape (I, 3) numpy.ndarray, dtype float64
        Each 3-element row represents a DOM x-, y, and z-coordinate at which to
        apply the photon info.

    survival_prob : shape (J_r, J_theta) numpy.ndarray, dtype float64
        Survival probability of a photon at this polar coordinate

    prho, pz : shape (J_r, J_theta) numpy.ndarrays, dtype float64
        Average photon rho (i.e., sqrt(x^2 + y^2)) and z-components at each
        polar (r, theta) coordinate

    r_max : float64
        Maximum radius in the radial binning

    binned_spv : shape (nx*ny*nz,) numpy.ndarray, dtype float64
        Binned photon survival probabilities * volumes, accumulated for all
        DOMs to normalize the average surviving photon info (`binned_px_spv`,
        etc.) in the end

    binned_px_spv, binned_py_spv, binned_pz_spv : shape (nx*ny*nz,) numpy.ndarray, dtype float64
        Existing arrays into which average photon components are accumulated

    binned_log_one_minus_sp : shape (nx*ny*nz,) numpy.ndarray, dtype float64
        Existing array to which ``1 - normed_survival_probability`` is
        multiplied (where the normalization factor is not infinite)

    x_min, x_max, y_min, y_max, z_min, z_max : double
        Binning limits (lower- and upper-most edges in each dimension)

    binwidth : double
        Bin width (all dimensions share bin width). Note that `binwidth` must
        be such that there are an integral number of bins in each dimension.

    oversample : int
        Oversampling factor, which allows for finer sub-bin shifting before the
        final binning is performed, e.g. to more accurately account for
        off-grid DOM positions while maintaining a reasonable final grid size.

    anisotropy : None or tuple
        Anisotropy parameter(s). Not yet implemented.

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
    retro.sphbin2cartbin.sphbin2cartbin
        Computes the mapping from each polar bin in the first octant to the
        indices of the Cartesian bins it overlaps and volume of overlap, and
        stores these to disk for use by this function.

    """
    assert binwidth > 0
    assert oversample >= 1
    assert anisotropy is None

    cdef:
        unsigned int num_first_octant_pol_bins = len(vol_arrays)

        double os_bw = binwidth / <double>oversample
        double inv_os_bw = 1.0 / os_bw

        double half_os_bw = os_bw / 2.0

        unsigned int nx = <unsigned int>round((x_max - x_min) / binwidth)
        unsigned int ny = <unsigned int>round((y_max - y_min) / binwidth)
        unsigned int nz = <unsigned int>round((z_max - z_min) / binwidth)

        double dom_x, dom_y, dom_z
        int dom_x_os_idx, dom_y_os_idx, dom_z_os_idx
        int x_os_idx, y_os_idx, z_os_idx
        double bin_pos_rho_norm

        double vol, prho_, px_, py_, pz_
        double px_unnormed, py_unnormed, pz_unnormed
        double px_firstquad, py_firstquad
        double sp, spv

        unsigned int ntheta_in_quad = <unsigned int>ceil(<double>ntheta / 2.0)
        unsigned int flat_pol_idx, r_idx, theta_idx, theta_idx_
        int x_idx, y_idx, z_idx
        int ix
        int[:] num_cart_bins_in_pol_bin = np.empty(num_first_octant_pol_bins, dtype=np.int32)
        int hemisphere, quadrant

        unsigned char[:] vol_mask = np.zeros((nx*ny*nz), dtype=np.uint8)
        double[:] binned_vol = np.zeros((nx*ny*nz), dtype=np.float64)
        unsigned int flat_cart_ix

        int dom_idx
        int num_doms = <int>dom_coords.shape[0]
        int nrows, ix0

        np.ndarray[unsigned int, ndim=2] ind_array
        np.ndarray[float, ndim=1] vol_array
        unsigned int **ind_array_ptrs = <unsigned int**>malloc(num_first_octant_pol_bins * sizeof(unsigned int*))
        float **vol_array_ptrs = <float**>malloc(num_first_octant_pol_bins * sizeof(float*))

    try:
        # Enforce < 1 micrometer accumulated error for binning
        assert abs(x_min + nx * binwidth - x_max) < 1e-6
        assert abs(y_min + ny * binwidth - y_max) < 1e-6
        assert abs(z_min + nz * binwidth - z_max) < 1e-6

        assert num_first_octant_pol_bins == nr * ntheta / 2

        for array, name in [(binned_spv, 'binned_spv'),
                            (binned_px_spv, 'binned_px_spv'),
                            (binned_py_spv, 'binned_py_spv'),
                            (binned_pz_spv, 'binned_pz_spv'),
                            (binned_log_one_minus_sp, 'binned_log_one_minus_sp')]:
            assert array.shape[0] == nx*ny*nz

        for ix in range(num_first_octant_pol_bins):
            num_cart_bins_in_pol_bin[ix] = vol_arrays[ix].shape[0]
            ind_array = ind_arrays[ix]
            vol_array = vol_arrays[ix]
            ind_array_ptrs[ix] = <unsigned int*>ind_array.data
            vol_array_ptrs[ix] = <float*>vol_array.data

        for dom_idx in range(num_doms):
            dom_x = dom_coords[dom_idx, 0]
            dom_y = dom_coords[dom_idx, 1]
            dom_z = dom_coords[dom_idx, 2]

            # Quick-and-dirty check to see if we can circumvent this DOM
            # altogether if the polar binning falls outside the binned volume
            # (This is not precise: won't exclude DOMs that do overlap, but
            # might include DOM that have no overlap)
            if dom_x + r_max <= x_min or dom_x - r_max >= x_max:
                continue
            if dom_y + r_max <= y_min or dom_y - r_max >= y_max:
                continue
            if dom_z + r_max <= z_min or dom_z - r_max >= z_max:
                continue

            dom_x_os_idx = <int>round((dom_x - x_min) * inv_os_bw)
            dom_y_os_idx = <int>round((dom_y - y_min) * inv_os_bw)
            dom_z_os_idx = <int>round((dom_z - z_min) * inv_os_bw)

            for r_idx in range(nr):
                for theta_idx in range(ntheta_in_quad):
                    flat_pol_idx = theta_idx + r_idx*ntheta_in_quad

                    nrows = num_cart_bins_in_pol_bin[flat_pol_idx]
                    for ix in range(nrows):
                        vol = <double>vol_array_ptrs[flat_pol_idx][ix]
                        ix0 = ix * 3
                        x_os_idx = <int>ind_array_ptrs[flat_pol_idx][ix0]
                        y_os_idx = <int>ind_array_ptrs[flat_pol_idx][ix0 + 1]
                        z_os_idx = <int>ind_array_ptrs[flat_pol_idx][ix0 + 2]

                        # Azimuth angle is detrmined by (x, y) bin center since
                        # we assume azimuthal symmetry
                        px_unnormed = <double>x_os_idx * os_bw  + half_os_bw
                        py_unnormed = <double>y_os_idx * os_bw  + half_os_bw
                        bin_pos_rho_norm = 1.0 / sqrt(px_unnormed*px_unnormed + py_unnormed*py_unnormed)
                        px_unnormed = px_unnormed * bin_pos_rho_norm
                        py_unnormed = py_unnormed * bin_pos_rho_norm

                        for hemisphere in range(2):
                            if hemisphere == 0:
                                z_idx = (z_os_idx + dom_z_os_idx) // oversample
                                theta_idx_ = theta_idx
                            else:
                                z_idx = (-1 - z_os_idx + dom_z_os_idx) // oversample
                                theta_idx_ = ntheta - 1 - theta_idx

                            if z_idx < 0 or z_idx >= nz:
                                continue

                            sp = survival_prob[r_idx, theta_idx_]
                            spv = sp * vol
                            prho_ = prho[r_idx, theta_idx_]
                            pz_ = pz[r_idx, theta_idx_]

                            px_firstquad = px_unnormed * prho_
                            py_firstquad = py_unnormed * prho_

                            for quadrant in range(4):
                                if quadrant == 0:
                                    x_idx = (x_os_idx + dom_x_os_idx) // oversample
                                    y_idx = (y_os_idx + dom_y_os_idx) // oversample
                                    px_ = px_firstquad
                                    py_ = py_firstquad

                                # x -> +y, y -> -x
                                elif quadrant == 1:
                                    x_idx = (-1 - y_os_idx + dom_x_os_idx) // oversample
                                    y_idx = (x_os_idx + dom_y_os_idx) // oversample
                                    px_ = -py_firstquad
                                    py_ = px_firstquad

                                # x -> -x, y -> -y
                                elif quadrant == 2:
                                    x_idx = (-1 - x_os_idx + dom_x_os_idx) // oversample
                                    y_idx = (-1 - y_os_idx + dom_y_os_idx) // oversample
                                    px_ = -px_firstquad
                                    py_ = -py_firstquad

                                # x -> -y, y -> x
                                elif quadrant == 3:
                                    x_idx = (y_os_idx + dom_x_os_idx) // oversample
                                    y_idx = (-1 - x_os_idx + dom_y_os_idx) // oversample
                                    px_ = py_firstquad
                                    py_ = -px_firstquad

                                if x_idx < 0 or x_idx >= nx or y_idx < 0 or y_idx >= ny:
                                    continue

                                # Compute base index; can use directly on typed memoryviews
                                flat_cart_ix = (x_idx * ny*nz) + (y_idx * nz) + z_idx
                                vol_mask[flat_cart_ix] = 1
                                binned_vol[flat_cart_ix] += vol
                                binned_spv[flat_cart_ix] += spv
                                binned_px_spv[flat_cart_ix] += px_ * spv
                                binned_py_spv[flat_cart_ix] += py_ * spv
                                binned_pz_spv[flat_cart_ix] += pz_ * spv

            # Normalize the weighted sum of survival probabilities for this DOM and
            # then include it in the overall survival probability via probabilistic
            # "or" statement:
            #     P(A) or P(B) = 1 - (1 - P(A)) * (1 - P(B))
            # though we stop at just the two factors on the right and more
            # probabilities can be easily combined before being subtracted form one
            # to yield the overall probability.
            for flat_cart_ix in range(nx*ny*nz):
                if vol_mask[flat_cart_ix] == 0:
                    continue
                binned_log_one_minus_sp[flat_cart_ix] += log(
                    1 - binned_spv[flat_cart_ix] / binned_vol[flat_cart_ix]
                )
    finally:
        free(ind_array_ptrs)
        free(vol_array_ptrs)
