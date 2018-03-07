@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def pexp_t_r_theta(pinfo_gen, hit_time, dom_coord, survival_prob,
                   time_indep_survival_prob, t_min, t_max, n_t_bins, r_min,
                   r_max, r_power, n_r_bins, n_costheta_bins):
    """Compute expected photons in a DOM based on the (t,r,theta)-binned
    Retro DOM tables applied to a the generated photon info `pinfo_gen`,
    and the total expected photon count (time integrated) -- the normalization
    of the pdf.

    Parameters
    ----------
    pinfo_gen : shape (N, 8) numpy ndarray, dtype float64
    hit_time : float
    dom_coord : shape (3,) numpy ndarray, dtype float64
    survival_prob
    time_indep_survival_prob
    t_min : float
    t_max : float
    n_t_bins : int
    r_min : float
    r_max : float
    r_power : float
    n_r_bins : int
    n_costheta_bins : int

    Returns
    -------
    total_photon_count, expected_photon_count : (float, float)

    """
    table_dt = (t_max - t_min) / n_t_bins
    table_dcostheta = 2. / n_costheta_bins
    expected_photon_count = 0.
    total_photon_count = 0.
    inv_r_power = 1. / r_power
    table_dr_pwr = (r_max-r_min)**inv_r_power / n_r_bins

    rsquared_max = r_max*r_max
    rsquared_min = r_min*r_min

    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable

        # A photon that starts immediately in the past (before the DOM was hit)
        # will show up in the Retro DOM tables in the _last_ bin.
        # Therefore, invert the sign of the t coordinate and index sequentially
        # via e.g. -1, -2, ....
        dt = t - hit_time
        dx = x - dom_coord[0]
        dy = y - dom_coord[1]
        dz = z - dom_coord[2]

        rsquared = dx**2 + dy**2 + dz**2
        # we can already continue before computing the bin idx
        if rsquared > rsquared_max:
            continue
        if rsquared < rsquared_min:
            continue

        r = math.sqrt(rsquared)

        #spacetime_sep = SPEED_OF_LIGHT_M_PER_NS*dt - r
        #if spacetime_sep < 0 or spacetime_sep >= retro.POL_TABLE_RMAX:
        #    print('spacetime_sep:', spacetime_sep)
        #    print('retro.MAX_POL_TABLE_SPACETIME_SEP:', retro.POL_TABLE_RMAX)
        #    continue

        r_bin_idx = int((r-r_min)**inv_r_power / table_dr_pwr)
        #print('r_bin_idx: ',r_bin_idx)
        #if r_bin_idx < 0 or r_bin_idx >= n_r_bins:
        #    #print('r at ',r,'with idx ',r_bin_idx)
        #    continue

        costheta_bin_idx = int((1 -(dz / r)) / table_dcostheta)
        #print('costheta_bin_idx: ',costheta_bin_idx)
        #if costheta_bin_idx < 0 or costheta_bin_idx >= n_costheta_bins:
        #    print('costheta out of range! This should not happen')
        #    continue

        # time indep.
        time_indep_count = (
            p_count * time_indep_survival_prob[r_bin_idx, costheta_bin_idx]
        )
        total_photon_count += time_indep_count

        # causally impossible
        if hit_time < t:
            continue

        t_bin_idx = int(np.floor((dt - t_min) / table_dt))
        #print('t_bin_idx: ',t_bin_idx)
        #if t_bin_idx < -n_t_bins or t_bin_idx >= 0:
        #if t_bin_idx < 0 or t_bin_idx >= -retro.POL_TABLE_DT:
        if t_bin_idx > n_t_bins or t_bin_idx < 0:
            #print('t')
            #print('t at ',t,'with idx ',t_bin_idx)
            continue

        #print(t_bin_idx, r_bin_idx, thetabin_idx)
        #raise Exception()
        surviving_count = (
            p_count * survival_prob[t_bin_idx, r_bin_idx, costheta_bin_idx]
        )

        #print(surviving_count)

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        expected_photon_count += surviving_count

    return total_photon_count, expected_photon_count
