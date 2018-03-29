#!/bin/bash

base="/data/justin/retro/scans/downgoing_muon_evtidx${start_evt}_newcode_ckv_compr_speedup"
mkdir -p "$base"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""
#tblkind="ckv_uncompr"

proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"
tblkind="ckv_templ_compr"

sdoms="dc_subdust"
#sdoms="all"

#noise="--no-noise"
noise=""

#GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl
gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.bz2"
angsens="h2-50cm"
norm="binvol2"
tkernel="const_e_loss"
tktimestep="1.0"
ckernel="point"

~/projects/retro/retro/retro_dom_pdfs.py \
    --outdir="~/projects/misc_retro/dom_pdfs" \
    --sim-to-test="upgoing_em_cascade" \
    --angsens-model="$angsens" \
    --cascade-kernel="$ckernel" \
    --track-kernel="$tkernel" \
    --track-time-step="$tktimestep" \
    --dom-tables-fname-proto="$proto" \
    --dom-tables-kind="$tblkind" \
    $tmpl_lib \
    --strs-doms="$sdoms" \
    --gcd="$gcd" \
    $noise \
    --norm-version="$norm"
