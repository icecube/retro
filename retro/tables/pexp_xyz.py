@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def pexp_xyz(pinfo_gen, x_min, y_min, z_min, nx, ny, nz, binwidth,
             survival_prob, avg_photon_x, avg_photon_y, avg_photon_z,
             use_directionality):
    """Compute the expected number of detected photons in _all_ DOMs at _all_
    times.

    Parameters
    ----------
    pinfo_gen :
    x_min, y_min, z_min :
    nx, ny, nz :
    binwidth :
    survival_prob :
    avg_photon_x, avg_photon_y, avg_photon_z :
    use_directionality : bool

    """
    expected_photon_count = 0.0
    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable
        x_idx = int(np.round((x - x_min) / binwidth))
        if x_idx < 0 or x_idx >= nx:
            continue
        y_idx = int(np.round((y - y_min) / binwidth))
        if y_idx < 0 or y_idx >= ny:
            continue
        z_idx = int(np.round((z - z_min) / binwidth))
        if z_idx < 0 or z_idx >= nz:
            continue
        sp = survival_prob[x_idx, y_idx, z_idx]
        surviving_count = p_count * sp

        # TODO: Incorporate photon direction info
        if use_directionality:
            raise NotImplementedError('Directionality cannot be used yet')

        expected_photon_count += surviving_count

    return expected_photon_count



