The table explains the different reconstruction varables obtained from running retro. Most variables sontain a mapping of 5 values:
* max : max LLH point
* mean : weighted mean over best LLH points
* meadian : weighted median over best LLH points (current standard)
* lower_bound : lower bound on variable from error estimate (-1 sigma)
* upper_bound : upper bound on variable from error estimate (+1 sigma)


| Variable                                 | Type       | Description                                                 |
|------------------------------------------|------------|-------------------------------------------------------------|
| retro_crs_prefit__azimuth                |            | Azimuth                                                     |
| retro_crs_prefit__cascade_energy         |            | Cascade Enegrgy (GeV)                                       |
| retro_crs_prefit__energy                 |            | Total Energy (GeV)                                          |
| retro_crs_prefit__fit_status             |            | 0 : ok -1: not yet run >0: failed                           |
| retro_crs_prefit__iterations             |            | number of minimizer iterations                              |
| retro_crs_prefit__llh_std                |            |                                                             |
| retro_crs_prefit__lower_dllh             |            |                                                             |
| retro_crs_prefit__max_llh                |            | max LLH value                                               |
| retro_crs_prefit__max_postproc_llh       |            | max LLH value after postprocessing (prior reweighting etc.) |
| retro_crs_prefit__median__cascade        | I3Particle |                                                             |
| retro_crs_prefit__median__neutrino       | I3Particle |                                                             |
| retro_crs_prefit__median__track          | I3Particle |                                                             |
| retro_crs_prefit__no_improvement_counter |            | number of minimizer iterations without improvement          |
| retro_crs_prefit__num_failures           |            |                                                             |
| retro_crs_prefit__num_llh                |            | total number of LLH calls                                   |
| retro_crs_prefit__num_mutation_successes |            | number of accepted CRS2 mutations                           |
| retro_crs_prefit__num_simplex_successes  |            | number of accepted CRS2 simplex points                      |
| retro_crs_prefit__run_time               |            | reco time (s)                                               |
| retro_crs_prefit__stopping_flag          |            | minimizer stopping flag                                     |
| retro_crs_prefit__time                   |            | Vertex time                                                 |
| retro_crs_prefit__track_azimuth          |            |                                                             |
| retro_crs_prefit__track_energy           |            |                                                             |
| retro_crs_prefit__track_zenith           |            |                                                             |
| retro_crs_prefit__upper_dllh             |            | LLH difference of bestfit track length vs. endpoint         |
| retro_crs_prefit__vertex_std             |            |                                                             |
| retro_crs_prefit__vertex_std_met_at_iter |            |                                                             |
| retro_crs_prefit__x                      |            |                                                             |
| retro_crs_prefit__y                      |            |                                                             |
| retro_crs_prefit__z                      |            |                                                             |
| retro_crs_prefit__zenith                 |            |                                                             |
| retro_crs_prefit__zero_dllh              |            | LLH difference of bestfit track length vs. cascade only     |
