#!/bin/bash

start_evt=0
n_events=1

base="/data/justin/retro/scans/upgoing_muon_evtidx${start_evt}_mie_notilt_tables_vs_lea_sim"
mkdir -p "$base"

events_root="/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_lea_holeice_as.9_gcd_md5_14bd15d0_geant_false_nsims100_step1"

#hits="/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1/photon_series/photons.pkl"
#hits="~justin/src/retro/data/MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_lea_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1_photon_series_photons0-9.pkl"
hits="$events_root/photons/photons.pkl"
hits_are_photons="--hits-are-photons"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""
#tblkind="ckv_uncompr"

#proto="/home/icecube/retro/tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
proto="/home/icecube/retro/tables/large_5d_notilt_combined/" #large_5d_notilt_string_{subdet}_depth_{depth_idx}"
tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
tblkind="ckv_templ_compr"

#sdoms="dc_subdust"
sdoms="all"

noise="--no-noise"
#noise=""

gcd="~/src/retro/data/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl"
angsens="9"
ttru="0"
xtru="0"
ytru="0"
ztru="-400"
zentru="3.141592653589793"
aztru="0"
tetru="20"
cetru="0"
norm="binvol2"
twindow="4000"
tkernel="table_e_loss"
tktimestep="1.0"
ckernel="aligned_one_dim"

tscan="-100-100:1"
xscan="-100-100:1"
zscan="-500--300:1"
zenscan="0-pi:0.031415926535897934"
azscan="-pi-pi:0.06283185307179587"
trckescan="0-100:1"
cscdescan="0-100:1"

~/src/retro/retro/scan_llh.py \
	--outdir="$base/t_${tscan}_z_${zscan}" \
	--events-root="$events_root" \
	--start-idx=$start_evt \
	--num-events=$n_events \
	--time="$tscan" \
	--x="$xtru" \
	--y="$ytru" \
	--z="$zscan" \
	--track-zenith="$zentru" \
	--track-azimuth="$aztru" \
	--track-energy="$tetru" \
	--cascade-energy="$cetru" \
	--hits="photons/photons" \
	--angsens-model="$angsens" \
	--cascade-kernel="$ckernel" \
	--track-kernel="$tkernel" \
	--track-time-step="$tktimestep" \
	--dom-tables-fname-proto="$proto" \
	--dom-tables-kind="$tblkind" \
	$tmpl_lib \
	--use-doms="$sdoms" \
	--gcd="$gcd" \
	$noise \
	--norm-version="$norm" &

#~/src/retro/retro/scan_llh.py \
#	--outdir="$base/x_${xscan}_z_${zscan}" \
#	--start-event-idx=$start_evt \
#	--num-events=$n_events \
#	--time="$ttru" \
#	--x="$xscan" \
#	--y="$ytru" \
#	--z="$zscan" \
#	--track-zenith="$zentru" \
#	--track-azimuth="$aztru" \
#	--track-energy="$tetru" \
#	--cascade-energy="$cetru" \
#	--hits-file="$hits" \
#	--hits-are-photons \
#	--angsens-model="$angsens" \
#	--cascade-kernel="$ckernel" \
#	--track-kernel="$tkernel" \
#	--track-time-step="$tktimestep" \
#	--dom-tables-fname-proto="$proto" \
#	--dom-tables-kind="$tblkind" \
#	$tmpl_lib \
#	--use-doms="$sdoms" \
#	--gcd="$gcd" \
#	$noise \
#	--norm-version="$norm" &
#
#	--norm-version="$norm" &
#
#~/src/retro/retro/scan_llh.py \
#	--outdir="$base/zenith_${zenscan}_azimuth_${azscan}" \
#	--start-event-idx=$start_evt \
#	--num-events=$n_events \
#	--time="$ttru" \
#	--x="$xtru" \
#	--y="$ytru" \
#	--z="$ztru" \
#	--track-zenith="$zenscan" \
#	--track-azimuth="$azscan" \
#	--track-energy="$tetru" \
#	--cascade-energy="$cetru" \
#	--hits-file="$hits" \
#	--hits-are-photons \
#	--angsens-model="$angsens" \
#	--cascade-kernel="$ckernel" \
#	--track-kernel="$tkernel" \
#	--track-time-step="$tktimestep" \
#	--dom-tables-fname-proto="$proto" \
#	--dom-tables-kind="$tblkind" \
#	$tmpl_lib \
#	--use-doms="$sdoms" \
#	--gcd="$gcd" \
#	$noise \
#	--norm-version="$norm" &
#
#~/src/retro/retro/scan_llh.py \
#	--outdir="$base/t_${tscan}_zenith_${zenscan}" \
#	--start-event-idx=$start_evt \
#	--num-events=$n_events \
#	--time="$tscan" \
#	--x="$xtru" \
#	--y="$ytru" \
#	--z="$ztru" \
#	--track-zenith="$zenscan" \
#	--track-azimuth="$aztru" \
#	--track-energy="$tetru" \
#	--cascade-energy="$cetru" \
#	--hits-file="$hits" \
#	--hits-are-photons \
#	--angsens-model="$angsens" \
#	--cascade-kernel="$ckernel" \
#	--track-kernel="$tkernel" \
#	--track-time-step="$tktimestep" \
#	--dom-tables-fname-proto="$proto" \
#	--dom-tables-kind="$tblkind" \
#	$tmpl_lib \
#	--use-doms="$sdoms" \
#	--gcd="$gcd" \
#	$noise \
#	--norm-version="$norm" &
#
#~/src/retro/retro/scan_llh.py \
#	--outdir="$base/cascade_energy_${cscdescan}_track_energy_${trckescan}" \
#	--start-event-idx=$start_evt \
#	--num-events=$n_events \
#	--time="$ttru" \
#	--x="$xtru" \
#	--y="$ytru" \
#	--z="$ztru" \
#	--track-zenith="$zentru" \
#	--track-azimuth="$aztru" \
#	--track-energy="${trckescan}" \
#	--cascade-energy="${cscdescan}" \
#	--hits-file="$hits" \
#	--hits-are-photons \
#	--angsens-model="$angsens" \
#	--cascade-kernel="$ckernel" \
#	--track-kernel="$tkernel" \
#	--track-time-step="$tktimestep" \
#	--dom-tables-fname-proto="$proto" \
#	--dom-tables-kind="$tblkind" \
#	$tmpl_lib \
#	--use-doms="$sdoms" \
#	--gcd="$gcd" \
#	$noise \
#	--norm-version="$norm" &

wait
