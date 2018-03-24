#!/bin/bash

start_evt=$1
n_events=1

timestamp="$( date +%Y-%m-%dT%H%M%z )"

end_evt=$(( start_evt + n_events - 1 ))
outdir="/data/justin/retro/recos/upmuckvcompr_uni_ins_mm1_lv160_et0.5_ef0.8_cz_cd8"
mkdir -p "$outdir"

#hits="/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1/photon_series/photons.pkl"
hits="~justin/src/retro/data/MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1_photon_series_photons0-9.pkl"
hits_are_photons="--hits-are-photons"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""
#tblkind="ckv_uncompr"

proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
tmpl_lib="--template-library /data/icecube/retro_tables/ckv_dir_templates.npy"
tblkind="ckv_templ_compr"

#noise="--no-noise"
noise=""

gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl"

importance_sampling="--importance-sampling"
consteff=""


~/src/retro/retro/reco.py \
	--outdir "$outdir" \
	--spatial-lims ic \
	--energy-lims 1,1000  \
	--energy-prior uniform \
	$importance_sampling \
	--max-modes 1 \
	$consteff \
	--n-live 160 \
	--evidence-tol 0.5 \
	--sampling-eff 0.8 \
	--max-iter 100000 \
	--seed 0 \
	--start-idx $start_evt \
	--num-events $n_events \
	--hits-file "$hits" \
	--hits-are-photons \
	--angsens-model "h2-50cm" \
	--cascade-kernel "point" \
	--track-kernel "const_e_loss" \
	--track-time-step 1.0 \
	--dom-tables-fname-proto "$proto" \
	--step-length 1.0 \
	--dom-tables-kind "$tblkind" \
	$tmpl_lib \
	--strs-doms "dc_subdust" \
	--gcd "$gcd" \
	$noise \
	--norm-version "binvol2"

wait
