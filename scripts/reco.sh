#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

events_base="$1"
start_idx="$2"
outdir="$3"

mkdir -p "$outdir"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""

#proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"

proto="/data/icecube/retro_tables/large_5d_notilt_combined/stacked"
tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"

#noise="--no-noise"
noise=""

importance_sampling="--importance-sampling"
consteff=""


#kernprof -l -v ~/src/retro/retro/reco.py \
~/src/retro/retro/reco.py \
    --outdir "$outdir" \
    --spatial-prior SPEFit2 \
    --temporal-prior SPEFit2 \
    --energy-prior log_uniform \
    --energy-lims 0.1,1000  \
    \
    $importance_sampling \
    --max-modes 1 \
    $consteff \
    --n-live 160 \
    --evidence-tol 0.5 \
    --sampling-eff 0.3 \
    --max-iter 10000 \
    --seed 0 \
    \
    --dom-tables-kind "ckv_templ_compr" \
    --dom-tables-fname-proto "$proto" \
    --strs-doms "all" \
    --gcd "GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl" \
    --norm-version "binvol2" \
    $tmpl_lib \
    --step-length 1.0 \
    $noise \
    \
    --cascade-kernel "point" \
    --track-kernel "table_e_loss" \
    --track-time-step 1.0 \
    \
    --events-base "$events_base" \
    --start-idx "$start_idx" \
    --num-events 1 \
    --truth \
    --pulses "OfflinePulses" \
    --recos "SPEFit2" \
    --triggers "I3TriggerHierarchy" \
    --hits "pulses/OfflinePulses" \
    --angsens-model "h2-50cm"

wait
