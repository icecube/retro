#!/bin/bash

start_evt=0
n_events=1

#base="/data/justin/retro/scans/downgoing_muon_evtidx${start_evt}_stacked_numballh_reordered"
base="/tmp/delme"
mkdir -p "$base"

#hits="/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1/photon_series/photons.pkl"
#hits="~justin/src/retro/data/MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1_photon_series_photons0-9.pkl"
#hits_are_photons="--hits-are-photons"

events_base="/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000_step1"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""
#tblkind="ckv_uncompr"

#proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib="--template-library /data/icecube/retro_tables/ckv_dir_templates.npy"
#tblkind="ckv_templ_compr"

proto="/fastio/icecube/retro/tables/large_5d_notilt_combined/stacked"
#proto="/data/icecube/retro_tables/large_5d_notilt_combined/stacked_reordered"
tmpl_lib="--template-library /fastio/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
tblkind="ckv_templ_compr"

#use_doms="dc_subdust"
use_doms="all"

#no_noise="--no-noise"
no_noise=""

gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.bz2"
angsens="h2-50cm"
ttru="0"
xtru="0"
ytru="0"
ztru="-300"
zentru="0"
aztru="0"
tetru="20"
cetru="0"
norm="binvol2"
tkernel="const_e_loss"
tktimestep="1.0"
ckernel="point"

tscan="-100-100:1"
xscan="-100-100:1"
zscan="-400--200:1"
zenscan="0-pi:0.031415926535897934"
azscan="-pi-pi:0.06283185307179587"
trckescan="0-100:1"
cscdescan="0-100:1"

#~/src/retro/retro/scan_llh.py \
#    --outdir="$base/t_${tscan}_z_${zscan}" \
#    --start-idx=$start_evt \
#    --num-events=$n_events \
#    --t="$tscan" \
#    --x="$xtru" \
#    --y="$ytru" \
#    --z="$zscan" \
#    --track-zenith="$zentru" \
#    --track-azimuth="$aztru" \
#    --track-energy="$tetru" \
#    --cascade-energy="$cetru" \
#    --hits-file="$hits" \
#    --hits-are-photons \
#    --angsens-model="$angsens" \
#    --cascade-kernel="$ckernel" \
#    --track-kernel="$tkernel" \
#    --track-time-step="$tktimestep" \
#    --dom-tables-fname-proto="$proto" \
#    --dom-tables-kind="$tblkind" \
#    $tmpl_lib \
#    --use-doms="$use_doms" \
#    --gcd="$gcd" \
#    $no_noise \
#    --norm-version="$norm" &
#
#~/src/retro/retro/scan_llh.py \
#    --outdir="$base/x_${xscan}_z_${zscan}" \
#    --start-idx=$start_evt \
#    --num-events=$n_events \
#    --t="$ttru" \
#    --x="$xscan" \
#    --y="$ytru" \
#    --z="$zscan" \
#    --track-zenith="$zentru" \
#    --track-azimuth="$aztru" \
#    --track-energy="$tetru" \
#    --cascade-energy="$cetru" \
#    --hits-file="$hits" \
#    --hits-are-photons \
#    --angsens-model="$angsens" \
#    --cascade-kernel="$ckernel" \
#    --track-kernel="$tkernel" \
#    --track-time-step="$tktimestep" \
#    --dom-tables-fname-proto="$proto" \
#    --dom-tables-kind="$tblkind" \
#    $tmpl_lib \
#    --use-doms="$use_doms" \
#    --gcd="$gcd" \
#    $no_noise \
#    --norm-version="$norm" &

~/src/retro/retro/scan_llh.py \
    --outdir="$base/zenith_${zenscan}_azimuth_${azscan}" \
    --time="$ttru" \
    --x="$xtru" \
    --y="$ytru" \
    --z="$ztru" \
    --track-zenith="$zenscan" \
    --track-azimuth="$azscan" \
    --track-energy="$tetru" \
    --cascade-energy="$cetru" \
    \
    --events-base="$events_base" \
    --start-idx=$start_evt \
    --num-events=$n_events \
    --photons "photons" \
    --hits "photons/photons" \
    \
    --angsens-model="$angsens" \
    \
    --dom-tables-kind "ckv_templ_compr" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "$use_doms" \
    --gcd "GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl" \
    $no_noise \
    --norm-version "binvol2" \
    $tmpl_lib \
    --step-length 1.0 \
    \
    --cascade-kernel="$ckernel" \
    --track-kernel="$tkernel" \
    --track-time-step="$tktimestep"

#~/src/retro/retro/scan_llh.py \
#    --outdir="$base/t_${tscan}_zenith_${zenscan}" \
#    --start-idx=$start_evt \
#    --num-events=$n_events \
#    --t="$tscan" \
#    --x="$xtru" \
#    --y="$ytru" \
#    --z="$ztru" \
#    --track-zenith="$zenscan" \
#    --track-azimuth="$aztru" \
#    --track-energy="$tetru" \
#    --cascade-energy="$cetru" \
#    --hits-file="$hits" \
#    --hits-are-photons \
#    --angsens-model="$angsens" \
#    --cascade-kernel="$ckernel" \
#    --track-kernel="$tkernel" \
#    --track-time-step="$tktimestep" \
#    --dom-tables-fname-proto="$proto" \
#    --dom-tables-kind="$tblkind" \
#    $tmpl_lib \
#    --use-doms="$use_doms" \
#    --gcd="$gcd" \
#    $no_noise \
#    --norm-version="$norm" &
#
#~/src/retro/retro/scan_llh.py \
#    --outdir="$base/cascade_energy_${cscdescan}_track_energy_${trckescan}" \
#    --start-idx=$start_evt \
#    --num-events=$n_events \
#    --t="$ttru" \
#    --x="$xtru" \
#    --y="$ytru" \
#    --z="$ztru" \
#    --track-zenith="$zentru" \
#    --track-azimuth="$aztru" \
#    --track-energy="${trckescan}" \
#    --cascade-energy="${cscdescan}" \
#    --hits-file="$hits" \
#    --hits-are-photons \
#    --angsens-model="$angsens" \
#    --cascade-kernel="$ckernel" \
#    --track-kernel="$tkernel" \
#    --track-time-step="$tktimestep" \
#    --dom-tables-fname-proto="$proto" \
#    --dom-tables-kind="$tblkind" \
#    $tmpl_lib \
#    --use-doms="$use_doms" \
#    --gcd="$gcd" \
#    $no_noise \
#    --norm-version="$norm" &

wait
