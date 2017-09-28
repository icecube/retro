# Retro Reco

## Introduction

Retro Reco is a method for reconstructing events within the IceCube/DeepCore neutrino detector which seeks to improve on previous methods by striking a pragmatic balance among the trade-offs of table generation time, table size, completeness of interaction physics and ice models, and reconstruction time.

With the previously-developed HybridReco and PegLeg reconstructions, we simulate photons coming from particular physics events (with particular space-time locations, orientations, energies, etc.) in the ice and recorded what the DOMs saw for each these simulations, building up tables connecting physics events directly with DOM detections.
For pragmatic reasons, these tables were only computed for cascades while tracks were modeled as a series of cascades.

Another method currently under development is Direct Reco, which performs MC simulation for _each_ event hypothesis--including the interaction physics, photon generation and propagation, and DOM photon detection.

In contrast to all of these methods, Retro Reco simulates photons as if they _originated from a DOM_ and records in tables where these photons travel in the ice (and their average direction) before they are absorbed by the ice.
With these tables, the count of photons recorded at a given location in the detector (appropriately normalized) can be interpreted as the survival probability for a photon _had it originated at that location_ to make it _back_ to the DOM which was simulated as a light source (think time-reversal symmetry).

With a full set of Retro tables, one can take any event hypothesis which has an expected photon generation profile (as a function of position and time in the detector) and figure out how many photons we expect to make it to each DOM and at what time we expect those photons to be detected.
Comparing these expected detections with the actual detections recorded in an event yields a likelihood that the event came from the hypothesis.

Finally, bolting on an optimizer that modifies the parameters of the event hypothesis so as to maximize this likelihood yields an event reconstruction.

More technical details of how the method works will follow, but now we look at the strengths and weaknesses of Retro Reco as it is currently implemented.

### Strengths and weaknesses of Retro Reco table generation

The table generation process, which took a month or more of human checking/tuning for HybridReco and PegLeg, now can be generated in parallel on a cluster with _no_ human intervention in approximately one day.

* One table is produced for each z-layer for each kind of DOM (IceCube and DeepCore), rather than one table per DOM (which would be more accurate but would take longer to generate and the resulting tables would exceed the memory available on typical cluster nodes).
* The above also forces the ice model to be limited to have no tilt in ice layers.
* We also do not model ice anisotropy in Retro Reco, but it should be straightforward to implement at least a simple azimuthal ice anisotropy in Retro.
* Likewise, we have not implemented DOM orientation; azimuthal orientation would be equally straightforward to implement as azimuthal anisotropy for the ice model, but zenith orientation (tilt) in the DOMs doesn't seem like it would be simple or practical to implement as it would probably require one Retro table per DOM.
* Hole ice is currently implemented only as a DOM acceptance angle (as in the H2 ice model), which in Retro is realized by a light source which produces photons with an angle-dependent probability. More complex hole ice models (that remain azimuthally symmetric) would be straightforward to use with Retro.
* DOM and cable geometries are not considered for photon absorption and reflection, but should be possible to include in the future within `clsim` as (1) a more complicated light source and (2) additional object geometries that photons can interact with. These things might be challenging to implement in `clsim`, but there is nothing about Retro itself that precludes including these.
* Only one DOM is simulated at a time, so the interference (absorption and/or reflection) due to other DOMs is ignored. This is _probably_ not an issue for IceCube since DOMs are small compared to their spacing, but DeepCore might be affected more by this assumption and e.g. Gen2 Phase I would be affected even more.
* Increasing photon statistics for better accuracy/"smoothness" of reconstructions is done at this stage, and so carries a one-time cost, vs. e.g. Direct Reco which incurs this cost upon generation of _each_ hypothesis under consideration.
* The individual-DOM tables needn't be modified for different detector geometries (GCD files), but the whole-detector time- and DOM-independent (TDI) table used for computing no-hit likelihoods does have to be modified for each unique detector geometry. This takes on the order of a few hours to do on a small cluster.

### Strengths and weaknesses of Retro Reco event hypotheses

Event hypotheses in Retro Reco are highly modular and _completely_ independent of the Retro tables (i.e., tables needn't be regenerated even for a completely new type of hypothesis that was never considered previously).
Furthermore, hypotheses can be generated in terms of one single _expected_ behavior which comes from the theory of the interaction physics and does not need to take photon propagation in the detector medium into account.
I.e., a hypothesis is only concerned with the photons at the _instant_ they are generated, and nothing else.
The Retro DOM tables take care of connecting the photons generated by the event hypothesis to the number and timing of expected hits at each DOM, and the Retro TDI table connects the generated photons to a total number of hits expected across the _whole_ detector (independent of time and DOM).

* Tracks needn't be modeled as a series of cascades.
* Cascades can be modeled as simplistically or (almost) as richly as desired.
* Compound hypotheses like track+cascade or double-bangs (two cascades) are as easy to implement as simple hypotheses, like cascade-only.
* The time to generate the hypothesis's generated-photon tuples and then to evaluate a likelihood for that hypothesis are both proportional to the number of samples from the hypothesis, so using the same time step, high energy events will be slower to reconstruct than low energy events. The time step can be modified, though, and it would be straightforward to implement dynamic time steps (different time steps for different parts of the event); no software changes need to be made outside the muon and/or cascade kernel functions to make this work.
* A caveat to the accuracy of a hypothesis model is that generated-photon directionality at a given location is currently summarized by a single _average photon direction_ vector, whereas e.g. for a muon track, the Cherenkov light will be emitted conically and for a low-energy cascade will be almost omnidirectional. How much this affects performance has yet to be measured.

## Summary of the Retro Reco process

Retro Reco currently works as follows, with software components utilized in each step listed beneath each step:

1. `clsim` is used with a custom light source (which emits photons according to DOM acceptance--in both angle and wavelength) to simulate photons coming _from_ a DOM. This then tabulates what time/polar bins the photons traverse before they are absorbed in the ice, along with the average photon directionality within each time/polar bin. One DOM of each "kind" (IceCube or DeepCore) is simulated for each (approximate) z-layer in the ice, and ice anisotropy as well as ice layer tilt is disabled for the simulation. Note that--as of now--when a DOM is simulated, photons do not get absorbed by anything besides the bulk ice. The resulting tables are binned in (t, r, theta, phi) and include (survival probability or photon counts?) and average photon direction within each bin.
    * ?
        * `clsim`
1. Tables are summed to remove phi dimension (?) and normalized such that for each time bin, summing over all spatial bins would yield 1 if absorption were 0. This yields tables of survival probability and average photon direction binned in (t, r, theta), one table for each DOM type (IceCube and DeepCore) and each of 60 z layers.
    * ?
1. All of the DOM tables are aggregated together to form a single (x, y, z)-binned time- and DOM-independent (TDI) table of survival probability and average photon direction. This table is used to compute a total number of photons that _should_ have been detected given a hypothesis which is used to determine the DOMs-should-have-been-hit-but-weren't part of the likelihood.
    * `generate_time_and_dom_indep_tables.py`, executed from command line
        * `sphbin2cartbin.pyx` (Cython), function `sphbin2cartbin`
        * `shift_and_bin.pyx` (Cython), function `shift_and_bin`
1. A hypothesis produces a discrete-time-step sequence of `(t, x, y, z, generated_photon_count, generated_photon_x, generated_photon_y, generated_photon_z)` tuples. These are the time-space locations of photons that are expected to be generated by the event hypothesis (note that this is _not_ the number of photons we expect to detect for the hypothesis).
    * `discrete_hypo.py`, class `DiscreteHypo`
        * `discrete_muon_kernels.py`, function `const_energy_loss_muon`
        * `discrete_cascade_kernels.py`, function `point_cascade`
1. The likelihood that an actual event in the detector came from the hypothesis is computed. To do this, we first take the generated photons from above and, using the single-DOM time/polar tables and the whole-detector TDI table, compute the number of photons we _expect_ to detect given the hypothesis. Comparing the expected-to-be-detected photons (and adding in expected noise hits) with the actually-detected photons is done with a Poisson likelihood function.
    * `likelihood.py`, function `get_neg_llh`
        * `table_readers.py`, class `DOMTimePolarTables` which uses function `pexp_t_r_theta` and class `TDICartTables` which uses function `pexp_xyz`.
1. Reconstruction is performed by scanning or minimizing the negative log likelihood over the space of possible hypotheses.
    * `likelihood.py`, executed from command line, calls function `main` which performs scanning and/or minimization given an events HDF5 file

## Software components listing

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
