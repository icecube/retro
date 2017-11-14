# Retro Reco

## Introduction

Retro Reco is a method for reconstructing events within the IceCube/DeepCore neutrino detector which seeks to improve on previous methods by striking a pragmatic balance among the trade-offs of table generation time, table size, completeness of interaction & ice model physics, and reconstruction time.

The previously-developed HybridReco and PegLeg reconstructions simulate photons coming from particular physics events (with particular space-time locations, orientations, energies, etc.) in the ice and record what the DOMs see for each of these simulations, building up tables connecting physics events directly with DOM detections.
For pragmatic reasons, these tables are only computed for cascades, while tracks are modeled as a series of cascades.
(Differences between HybridReco and PegLeg are beyond the scope of the discussion here.)

Another method currently under development is Direct Reco, which performs MC simulation for _each_ event hypothesis--including the interaction physics, photon generation and propagation in the ice, and DOM photon detection.

In contrast to all of these methods, Retro Reco simulates photons as if they _originated from a DOM_ and records in tables where these photons travel in the ice (taking into account their average directionality) before they are absorbed by the ice.
With Retro tables, the count of photons recorded at a given location in the detector (appropriately normalized) and average directionality can be interpreted as the survival probability for a photon _had it originated at that location_ (traveling in a particular direction) to make it _back_ to the DOM which was simulated as a light source.

With a full set of Retro tables, one can take any event hypothesis which has an expected photon generation profile (as a function of position and time in the detector) and figure out how many photons we expect to make it to each DOM and at what time we expect those photons to be detected.
Comparing these expected detections with the actual detections recorded in an event yields a likelihood that the event came from the hypothesis.

Finally, bolting on an optimizer that modifies the parameters of the event hypothesis so as to maximize this likelihood yields an event reconstruction.

More technical details of how the method works will follow, but now we look at the strengths and weaknesses of Retro Reco as it is currently implemented.

### Strengths and weaknesses of Retro Reco table generation

The table generation process, which took a month or more of human checking/tuning for HybridReco and PegLeg, now can be generated in parallel on a cluster with _no_ human intervention in approximately one day.

* One table is produced for each z-layer for each kind of DOM (IceCube and DeepCore); while one table could be produced per DOM and this would be more accurate, it would take much longer to generate and the resulting tables would exceed the RAM available on typical cluster nodes. This choice also forces the ice model to be limited to have no tilt in ice layers.
* We currently do not model ice anisotropy in Retro Reco, but it should be straightforward to implement at least a simple azimuthally-dependent ice anisotropy.
* Due to the above, we do not implement DOM orientations. And as above, azimuthal orientation would be equally straightforward to implement as azimuthal anisotropy for the ice model, but zenith orientation (tilt) in the DOMs doesn't seem like it would be simple or practical to implement as it would probably require one Retro table per DOM.
* Hole ice is currently implemented only as a DOM acceptance angle (as in the H2 ice model), which in Retro is realized by a light source which produces photons with an angle-dependent (zenith angle only for now) probability. More complex hole ice models (that remain azimuthally symmetric) would be straightforward to use with Retro.
* DOM and cable geometries are not considered for photon absorption and reflection, but should be possible to include in the future within `clsim` as (1) a more complicated light source and (2) additional object geometries that photons can interact with. These things might be challenging to implement in `clsim`, but there is nothing about Retro itself that precludes including these.
* Only one DOM is simulated at a time, so the interference (absorption and/or reflection) due to other DOMs is ignored. This is _probably_ not an issue for IceCube since DOMs are small compared to their spacing, but DeepCore might be affected more by this assumption and e.g. Gen2 Phase I would be affected even more.
* Increasing photon statistics for better accuracy/"smoothness" of reconstructions is done at this stage, and so carries a one-time cost, vs. e.g. Direct Reco which incurs this cost upon generation of _each_ hypothesis under consideration.
* The individual-DOM tables needn't be modified for different detector geometries (GCD files), but the whole-detector time- and DOM-independent (TDI) table used for computing no-hit likelihoods does have to be modified for each unique detector geometry. The TDI table takes on the order of a few hundred core hours to regenerate.
* Directionality of photons in the tables is summarized by a single vector with direction and length, and a simple model of "how directionally dependent" is built based on these parameters. This is obviously simplistic, but this choice was made to keep table size down so tables will fit in RAM.

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

1. `clsim` is used with a custom light source (which emits photons according to DOM acceptance--in both angle and wavelength) to simulate photons coming _from_ a DOM. This then tabulates what time/polar bins the photons traverse before they are absorbed in the ice; the photon directionality (in zenith (`theta_dir`) and relative-azimuth (`delta_phi_dir`) bins) is also binned. One DOM of each "kind" (IceCube or DeepCore) is simulated for each (approximated) z-layer in the ice, and ice anisotropy as well as ice layer tilt is disabled for the simulation. Note that--as of now--when a DOM is simulated, photons do not get absorbed by anything besides the bulk ice. The resulting tables are binned in (`theta`, `r`, `t`, `theta_dir`, `delta_phi_dir`) and each bin contains a single photon count in it.
    * `generate_clsim_table.py`, which uses the `clsim` IceTray module
1. The clsim tables are averaged together, averaged over the phi dimension, and photon directionality is reduced from two binning dimensions to a single average directionality vector. Finally, the counts are normalized to be detection probabilities. This produces tables binned in (t, r, theta) where each bin contains a detection probability and average photon vector. The normalization is such that that for each time and (theta, phi) bin, summing over all other bins would yield a detection probability of one if there was no absorption in the ice. There is one such table for each DOM type (IceCube and DeepCore) and for each of 60 z layers.
    * `table_readers.py`, class `DOMRawTable` method `export_dom_time_polar_table` which calls `generate_t_r_theta_table.py` function `generate_t_r_theta_table`.
1. All of the DOM tables are aggregated together to form a single (x, y, z)-binned time- and DOM-independent (TDI) table of survival probability and average photon direction. This table is used to compute a total number of photons that _should_ have been detected given a hypothesis which is used to determine the DOMs-should-have-been-hit-but-weren't part of the likelihood.
    * `generate_tdi_table.py`, executed from command line calls function `generate_tdi_table`
        * `generate_binmap.py` function `generate_binmap` uses `sphbin2cartbin.pyx` (Cython) function `sphbin2cartbin` to map spherical to Cartesian bins
        * `shift_and_bin.pyx` (Cython), function `shift_and_bin` does some of the low-level heavy lifting for `generate_tdi_table`
1. A hypothesis produces a discrete-time-step sequence of `(t, x, y, z, generated_photon_count, generated_photon_x, generated_photon_y, generated_photon_z)` sequences. These are the time-space locations of "photons" that are expected to be generated by the event hypothesis (note that this is _not_ the number of photons we expect to detect for the hypothesis).
    * `discrete_hypo.py`, class `DiscreteHypo`
        * `discrete_muon_kernels.py`, function `const_energy_loss_muon`
        * `discrete_cascade_kernels.py`, function `point_cascade`
1. The likelihood that an actual event in the detector came from the hypothesis is computed. To do this, we first take the generated photons from above and, using the single-DOM time/polar tables and the whole-detector TDI table, compute the number of photons we _expect_ to detect given the hypothesis. Comparing the expected-to-be-detected photons (and adding in expected noise hits) with the actually-detected photons is done with a Poisson likelihood function.
    * `likelihood.py`, function `get_neg_llh`
        * `table_readers.py`, class `DOMTimePolarTables` which uses function `pexp_t_r_theta` and class `TDICartTables` which uses function `pexp_xyz`.
1. Reconstruction is performed by scanning or minimizing the negative log likelihood over the space of possible hypotheses.
    * `likelihood.py`, executed from command line, calls function `main` which performs scanning and/or minimization given an events HDF5 file

## Directory listing

* `doc/`: documentation directory; Sphinx documention skeleton and built docs go in this directory.
* `icetray_info_scripts/`: scripts to extract simple info coded into IceTray modules
* `retro/`: The Retro Reco sourcecode 
* `scripts/`: Driver scripts for running Retro Reco and other miscellaneous scripts
* `.gitignore`: Files for git to ignore
* `README.md`: This file, an overview of Retro Reco and directory listing / description
* `setup.py`: Python install script, for use e.g. by ``python setup.py build_ext --inplace``
