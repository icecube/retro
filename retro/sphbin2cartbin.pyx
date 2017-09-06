cimport cython

from libc.math cimport ceil, floor, round, sqrt, cos, sin

import numpy as np
cimport numpy as np


@cython.embedsignature(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def sphbin2cartbin(double r_max, double r_power,
                   int n_rbins, int n_costhetabins, int n_phibins,
                   double x_bw, double y_bw, double z_bw,
                   int x_oversample, int y_oversample,
                   int z_oversample,
                   int antialias_factor=1):
    """
    Parameters
    ----------
    r_max : double
        Maximum r bin edge (note that r binning is assumed to start at 0)

    r_power : double
        Power used for regular power-law binning in radius

    n_rbins, n_costhetabins, n_phibins : int with n_costhetabins % 2 == 0 and n_phibins % 4 == 0
        Number or r, costheta, and phi bins; note that while there are no phi
        bins expected in the retro tables, `n_phi` is used for binning up any
        spherical cells that did not catch at least one Cartesian bin. Note
        also that both n_costhetabins and n_phibins must be even and refer to
        the entire range of costheta and phi, even though we only work with the
        first octant. Therefore, we require each of these to define cells that
        fit exactly within the first octant: n_costhetabins must be even, while
        n_phibins must be an integer multiple of 4.

    x_bw, y_bw, z_bw : double
        Cartesian binwidths in x, y, and z directions

    x_oversample, y_oversample, z_oversample : int >= 1
        Oversampling factors. If oversampling is used, the returned indices
        array will have integer point values from 0 to nx*x_oversample, etc.
        E.g., a bin index with oversampling of 3 would be represented by
        indices 0, 1, and 2. This allows for centering the grid more accurately
        on a DOM without upsampling the entire grid.

    antialias_factor : int from 1 to 50
        The smallest binning unit in each dimension is divided again (i.e.
        after oversampling) by this factor for more accurately computing the
        volume of overlap (and then the sub-binning for antialiasing is
        discarded). This therefore does not add to the memory footprint of the
        final binning, but there will be more partial overlaps found, and so
        `ind_arrays` and `vol_arrays` will be larger. This does increase the
        computational burden for this function _and_ for the `shift_and_bin`
        function (due to the larger ind/vol arrays).

    Returns
    -------
    ind_arrays : list of M numpy.ndarrays each of shape (N, 3), dtype int32
        One array per spherical bin in the first octant. Indices refer to the
        oversampled binning.

    vol_arrays : list of M numpy.ndarrays each of shape (N,), dtype float32
        One array per spherical bin

    See Also
    --------
    retro.shift_and_bin.shift_and_bin

    """
    DEF MAX_AA_FACTOR = 50

    DEF PI = 3.1415926535897932384626433827
    DEF TWO_PI = 6.2831853071795862319959269371
    DEF PI_BY_2 = 1.5707963267948965579989817343

    assert n_rbins >= 1 and n_costhetabins >= 1 and n_phibins >= 1
    assert n_costhetabins % 2 == 0, 'A costheta bin straddles XY-plane since `n_costhetabins` is not even'
    assert n_phibins % 4 == 0, 'Phi bins straddle YZ-plane since `n_phibins` is not evenly divisible by 4'
    assert x_oversample >= 1 and y_oversample >= 1 and z_oversample >= 1
    assert 1 <= antialias_factor <= MAX_AA_FACTOR

    # Logic below about extrapolating from first octant to other octants
    # fails if bin widths are different in different dimensions
    assert x_bw == y_bw == z_bw
    assert x_oversample == y_oversample == z_oversample

    cdef:
        int n_quadrant_costhetabins = <int>ceil(<double>n_costhetabins / 2.0)

        double x_os_dbl = <double>x_oversample
        double y_os_dbl = <double>y_oversample
        double z_os_dbl = <double>z_oversample

        double x_bw_os = x_bw / x_os_dbl
        double y_bw_os = y_bw / y_os_dbl
        double z_bw_os = z_bw / z_os_dbl

        double x_bw_os_aa = x_bw_os / <double>antialias_factor
        double y_bw_os_aa = y_bw_os / <double>antialias_factor
        double z_bw_os_aa = z_bw_os / <double>antialias_factor

        double aa_vol = x_bw_os_aa * y_bw_os_aa * z_bw_os_aa

        double x_halfbw_os_aa = x_bw_os_aa / 2.0
        double y_halfbw_os_aa = y_bw_os_aa / 2.0
        double z_halfbw_os_aa = z_bw_os_aa / 2.0

        int n_xbins_oct_os = <int>ceil(r_max / x_bw) * x_oversample
        int n_ybins_oct_os = <int>ceil(r_max / y_bw) * y_oversample
        int n_zbins_oct_os = <int>ceil(r_max / z_bw) * z_oversample

        double inv_r_power = 1.0 / r_power
        double power_r_bin_scale = <double>n_rbins / r_max**inv_r_power
        double costheta_bin_scale = <double>n_costhetabins / 2.0
        double dphi = TWO_PI / <double>n_phibins

        unsigned int x_os_idx, y_os_idx, z_os_idx
        int xi, yi, zi
        int r_bin_idx, costheta_bin_idx, flat_bin_idx

        double x0, y0, z0

        double x_centers_sq[MAX_AA_FACTOR]
        double rho_squares[MAX_AA_FACTOR][MAX_AA_FACTOR]

        double x_center, y_center, z_center, x_center_sq, y_center_sq,  z_center_sq
        double rho_sq

        double r

        double[:] phi_bin_centers = np.linspace(
            start=TWO_PI / <double>n_phibins / 2.0,
            stop=PI_BY_2 - TWO_PI / <double>n_phibins / 2.0,
            num=n_phibins
        )

        dict d
        list bin_mapping = []
        tuple xyz_idx_q1
        list ind_arrays = []
        list vol_arrays = []

        double r_bmin, r_bmax, r_bcenter
        double costheta_bmin, costheta_bmax, costheta_bcenter
        double rho_center, phi_bin_center

        double total_tabulated_vol, sph_bin_vol
        double dcostheta

    # Note: there is no phi dependence in the source tables

    for costheta_bin_idx in range(n_quadrant_costhetabins):
        for r_bin_idx in range(n_rbins):
            bin_mapping.append(dict())

    for x_os_idx in range(n_xbins_oct_os):
        x0 = <double>x_os_idx * x_bw_os + x_halfbw_os_aa
        for xi in range(antialias_factor):
            x_center = x0 + xi * x_bw_os_aa
            x_center_sq = x_center * x_center
            x_centers_sq[xi] = x_center_sq

        for y_os_idx in range(n_ybins_oct_os):
            y0 = <double>y_os_idx * y_bw_os + y_halfbw_os_aa
            for yi in range(antialias_factor):
                y_center = y0 + yi * y_bw_os_aa
                y_center_sq = y_center * y_center
                for xi in range(antialias_factor):
                    rho_squares[xi][yi] = x_centers_sq[xi] + y_center_sq

            for z_os_idx in range(n_zbins_oct_os):
                # NOTE: populating _only_ first octant values!
                xyz_idx_q1 = (x_os_idx, y_os_idx, z_os_idx)

                z0 = <double>z_os_idx * z_bw_os + z_halfbw_os_aa
                for zi in range(antialias_factor):
                    z_center = z0 + zi * z_bw_os_aa
                    z_center_sq = z_center * z_center
                    for xi in range(antialias_factor):
                        for yi in range(antialias_factor):
                            r = sqrt(rho_squares[xi][yi] + z_center_sq)
                            if r < 0 or r >= r_max:
                                continue

                            r_bin_idx = <int>floor(r**inv_r_power * power_r_bin_scale)
                            costheta_bin_idx = <int>((1.0 - z_center / r) * costheta_bin_scale)
                            if costheta_bin_idx < 0 or costheta_bin_idx >= n_costhetabins:
                                continue

                            d = bin_mapping[costheta_bin_idx + n_quadrant_costhetabins*r_bin_idx]
                            d[xyz_idx_q1] = d.get(xyz_idx_q1, 0.0) + aa_vol

    # Account for any spherical bins that were assigned no corresponding
    # Cartesian bins; otherwise, normalize total binned volume to be no larger
    # than that of each spherical volume element (dr and dcostheta come
    # from the polar bin, while dphi is pi/2 since we only computed one
    # octant)
    for r_bin_idx in range(n_rbins):
        r_bmin = (<double>r_bin_idx / power_r_bin_scale)**r_power
        r_bmax = (<double>(r_bin_idx + 1) / power_r_bin_scale)**r_power

        for costheta_bin_idx in range(n_quadrant_costhetabins):
            costheta_bmin = 1 - costheta_bin_idx / costheta_bin_scale
            costheta_bmax = 1 - (costheta_bin_idx + 1) / costheta_bin_scale
            dcostheta = costheta_bmax - costheta_bmin

            d = bin_mapping[costheta_bin_idx + n_quadrant_costhetabins*r_bin_idx]

            # For spherical volume elements that don't have any assigned
            # Cartesian bins, find spherical element's center and locate it in
            # the closest Cartesian bin
            if len(d) == 0:
                r_bcenter = (r_bmin + r_bmax) / 2.0
                costheta_bcenter = (costheta_bmin + costheta_bmax) / 2.0
                z_center = r_bcenter * costheta_bcenter
                z_os_idx = <unsigned int>round(z_center / z_bw_os)
                rho_center = sqrt(r_bcenter**2 - z_center**2)
                sph_bin_vol = -dcostheta * (r_bmax**3 - r_bmin**3) / 3.0 * dphi

                for phi_idx in range(n_phibins):
                    phi_bin_center = phi_bin_centers[phi_idx]
                    x_center = rho_center * cos(phi_bin_center)
                    y_center = rho_center * sin(phi_bin_center)

                    x_os_idx = <unsigned int>round(x_center / x_bw_os)
                    y_os_idx = <unsigned int>round(y_center / y_bw_os)
                    xyz_idx_q1 = (x_os_idx, y_os_idx, z_os_idx)

                    # NOTE: duplicates are overwritten (i.e., should be at
                    # most one entry for xyz_idx_q1 per spherical bin)
                    d[xyz_idx_q1] = sph_bin_vol

            # Normalize volume weight assigned from this spherical element
            # to all associated Cartesian elements to be exactly the volume of the
            # spherical volume element (i.e., the area of the polar element rotated
            # through the first octant). We can assume this normalization to be correct
            # since we define the Cartesian binning to completely encapsulate the entire spherical
            # volume (so there won't be any "unbinned" parts of the spherical vol)
            sph_bin_vol = -dcostheta * (r_bmax**3 - r_bmin**3) / 3.0 * PI_BY_2
            vols = np.atleast_1d(d.values())
            total_tabulated_vol = np.sum(vols)
            norm_factor = sph_bin_vol / total_tabulated_vol
            vols *= norm_factor

            ind_arrays.append(np.atleast_2d(np.array(d.keys(), dtype=np.uint32)))
            vol_arrays.append(vols.astype(np.float32))

    return ind_arrays, vol_arrays
