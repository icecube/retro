# Adding and specifying priors

For priors, you can use any distribution found in `scipy.stats.distributions` (you specify the parameter(s) of the distribution) or you can use linearly-interpolated points that describe the prior.

## Using a distribution from scipy

For a distribution in `scipy.stats.distributions`, you set the distribution's parameters when you define your reconstruction in reco.py.
An example can be found in `retro/reco.py`, for the reco method `"stopping_atm_muon_crs"`.

## Adding a custom prior

An example of generating an arbitrary prior from reconstructed Monte Carlo can be found in the jupyter notebook `retro/notebooks/plot_prior_reco_candidates.ipynb`.
This notebook features the use of the function `prior_from_reco` found in `retro/retro/utils/prior_from_reco.py` which uses variable bandwidth kernel density estimation (VBWKDE) from a module found in the PISA project.
That function produces a pickle file containing the priors it produces and plots the results.
The pickle file produced is then used by Retro to set priors.
(Note one could produce the same format of output file using a method other than VBWKDE, but this is currently the only implementation.)

The pickle file produced by the above gets used by the function `define_prior_from_prefit` in `retro/priors.py` and you can see how priors are set up within that same file; if you come up with new prior definitions, you can add another entry to `retro/priors.py` in function `get_prior_func` as an `elif` entry the code block re-casting priors.

# Adding and specifying a reconstruction

Reconstructions are added to class `Reco`, method `_reco_event`.
An example can be seen for method `"stopping_atm_muon_crs"`, reproduced here for reference:

```
elif method == "stopping_atm_muon_crs":
    self.setup_hypo(
        track_kernel="stopping_table_energy_loss", track_time_step=3.0
    )

    self.generate_prior_method(
        x=dict(kind=PRI_UNIFORM, extents=EXT_IC["x"]),
        y=dict(kind=PRI_UNIFORM, extents=EXT_IC["y"]),
        z=dict(kind=PRI_UNIFORM, extents=EXT_IC["z"]),
        time=dict(kind=PRI_TIME_RANGE),
        track_zenith=dict(
            kind=PRI_COSINE, extents=((0, Bound.ABS), (np.pi / 2, Bound.ABS))
        ),
    )

    param_values = []
    log_likelihoods = []
    aux_values = []
    t_start = []

    self.generate_loglike_method(
        param_values=param_values,
        log_likelihoods=log_likelihoods,
        aux_values=aux_values,
        t_start=t_start,
    )

    run_info, fit_meta = self.run_crs(
        n_live=160,
        max_iter=10000,
        max_noimprovement=1000,
        min_llh_std=0.,
        min_vertex_std=dict(x=5, y=5, z=4, time=20),
        use_sobol=True,
        seed=0,
    )

    llhp = self.make_llhp(
        method=method,
        log_likelihoods=log_likelihoods,
        param_values=param_values,
        aux_values=aux_values,
        save=save_llhp,
    )

    self.make_estimate(
        method=method,
        llhp=llhp,
        remove_priors=False,
        run_info=run_info,
        fit_meta=fit_meta,
        save=save_estimate,
    )
```
