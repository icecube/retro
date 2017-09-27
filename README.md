# Retro Reco!

## Software Components

The following modules and scripts are used for generating Retro tables,
distilling these into different useful forms, using the tables to evaluate
event hypothesis likelihoods, and finally performing event reconstruction.

* `__init__.py`: Package-wide constants, helper functions, namedtuples, and
simple classes
* `analytic_hypo.py`: (deprecated, should update to validate against new stuff)
* `angle.py`: ?
* `angular.py`: ?
* `discrete_cascade_kernels.py`: Discrete-time cascade photon expectation functions to be used with DiscreteHypo class
* `discrete_hypo.py`: defines `DiscreteHypo`, a simple class for evaluating discrete hypotheses.
* `discrete_muon_kernels.py`: Discrete-time muon photon expectation functions to be used with `DiscreteHypo` class
* `events.py`: Events class for loading and working with events from icetray-produced HDF5 file
* `generate_table.py`: Tabulate the light flux for a DOM
* `generate_time_and_dom_indep_tables.py`: aggregate single-DOM tables in
time and generate a single time- and DOM-independent Cartesian (x, y, z) table.
This  includes _all_ IceCube/DeepCore DOMs, and is used for finding the total
light expectation for a given event hypothesis, independent of DOM and time,
for purposes of accurately estimating the no-hit likelihood (i.e., all excess
light a hypothesis predicts). Note that this is a very time-consuming task, so
the full volume can be divided into non-overlapping regions and each region can
be computed independently e.g. in a cluster. The `TDICartTables` table reader
is intelligent enough to stitch all of these regions together into the full
detector when intantiated.
* `generate_time_and_dom_indep_tables.obsolete.py`: (obsolete)
* `geo.py`: ?
* `hypo.py`: (deprecated) Define class Hypo 
* `hypo_legacy.py`: (obsolete) Older version of `hypo.py`
* `likelihood.py`: function `get_neg_llh` for performing the full likelihood computation. This optionally includes no-hit likelihood term via TDI tables. Note that photon expectations that go into the likelihoods are computed by the table-reader classes, so the handling of average photon directionality is found in `table_readers.py`.
* `nphotons.py`: Display photon production info for tracks and cascades, as
parameterized in IceCube software
* `particles.py`: Particle and ParticleArray classes for storing and accessing info for
neutrinos, tracks, cascades, etc.
* `plot_1d_scan.py`: Plot likelihood scan results
* `plot_csv.py`: ?
* `plot_hypo_comparisons.py`: (deprecated, needs to be updated) Make plots for a
single hypothesis, comparing segmented to analytic hypotheses.
* `plot_slices.ipynb`: Notebook that does too much: test likelihood scans _and_
plot slices and projections of TDI tables
* `plot_summed_tables.py`: Make plots from the 3-d retro tables. The output
will be 2-d maps plus the time dimension as the video time
* `plot_tables.py`: ?
* `plot_time_and_dom_indep_tables.py`: (deprecated: ugly plots) 2D and 3D
visualization of a time- and DOM-independent (TDI) table
* `print_tables.py`: ?
* `segmented_hypo.py`: (deprecated) SegmentedHypo class for segmented track hypo
and cascade hypo, where each segment is identical for the length of the track.
* `shift_and_bin.pyx`: Shift (r, theta) retro tables (i.e., (t, r, theta)
tables with time marginalized out) to each DOM location and aggregate their
quantities (with appropriate weighting) in (x, y, z) retro tables.
* `smooth_all.sh`: ?
* `smooth_tables.py`: ?
* `solve.py`: ?
* `sparse.py`: (deprecated: simply use dicts instead for compatibility with e.g. Cython) Class to handle sparse elements in n-dimensional array
* `sphbin2cartbin.ipynb`: (obsolete: use `sphbin2cartbin.pyx` instead)
* `sum_tables.py`: ?
* `sum_tables_including_angles.py`: ?
* `table_readers.py`: Classes for reading and getting info from Retro tables.
* `track_hypo.py`: (deprecated)
* `wavelength.py`: ?
