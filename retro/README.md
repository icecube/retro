# Directory listing

The following modules and scripts are used for generating Retro tables, distilling these into different useful forms, using the tables to evaluate event hypothesis likelihoods, and finally performing event reconstruction.
The listing is in alphabetical order.

* `__init__.py`: Package-wide constants, helper functions, namedtuples, and simple classes
* `discrete_cascade_kernels.py`: Discrete-time cascade photon expectation functions to be used with DiscreteHypo class
* `discrete_hypo.py`: defines `DiscreteHypo`, a simple class for evaluating discrete hypotheses.
* `discrete_muon_kernels.py`: Discrete-time muon photon expectation functions to be used with `DiscreteHypo` class
* `events.py`: Events class for loading and working with events from icetray-produced HDF5 file
* `generate_binmap.py`: 
* `generate_clsim_table.py`: Tabulate the light flux for a DOM
* `generate_tdi_table.py`: aggregate single-DOM tables in time and generate a single time- and DOM-independent Cartesian (x, y, z) table including _all_ IceCube/DeepCore DOMs.
* `generate_t_r_theta_table.py`: 
* `likelihood.py`: function `get_neg_llh` for performing the full likelihood computation. This optionally includes no-hit likelihood term via TDI tables. Note that photon expectations that go into the likelihoods are computed by the table-reader classes, so the handling of average photon directionality is found in `table_readers.py`.
* `particles.py`: Particle and ParticleArray classes for storing and accessing info for neutrinos, tracks, cascades, etc.
* `plot_1d_scan.py`: Plot likelihood scan results
* `plot_slices.ipynb`: Notebook that does too much: test likelihood scans _and_ plot slices and projections of TDI tables
* `plot_tables.py`: ?
* `plot_tdi_table.py`: (deprecated: ugly plots) 2D and 3D visualization of a time- and DOM-independent (TDI) table
* `plot_t_r_theta_tables.py`: Make plots from the 3-d retro tables. The output will be 2-d maps plus the time dimension as the video time
* `shift_and_bin.pyx`: Shift (r, theta) retro tables (i.e., (t, r, theta) tables with time marginalized out) to each DOM location and aggregate their quantities (with appropriate weighting) in (x, y, z) retro tables.
* `sphbin2cartbin.ipynb`: (obsolete: use `sphbin2cartbin.pyx` instead)
* `sphbin2cartbin.pyx`: Generates mapping (including volume of overlap) from spherical to Cartesian bins.
* `table_readers.py`: Classes for reading and getting info from Retro tables.
