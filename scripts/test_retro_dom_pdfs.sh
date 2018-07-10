#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

basedir="/data/justin/retro/dom_distributions"

outdir="$basedir/$sim_to_test"
mkdir -p "$outdir"

sdoms="dc_subdust"
#sdoms="all"

sim_to_test="mie_upgoing_muon"
angsens="h2-50cm"
norm="binvol2"
tkernel="table_e_loss"
tktimestep="1"
ckernel="aligned_one_dim"

gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.bz2"

#no_noise="--no-noise"
no_noise=""


# -- Tables -- #

if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/stacked"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    proto="/fastio/icecube/retro/tables/large_5d_notilt_combined/stacked/"
    tmpl_lib="--template-library /fastio/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tblkind="ckv_templ_compr"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
else
    proto="/gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/stacked/"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
fi

mkdir -p "$outdir"

~/src/retro/retro/retro_dom_pdfs.py \
    --outdir="$outdir" \
    --sim-to-test="$sim_to_test" \
    \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    --gcd "$gcd" \
    --norm-version "$norm" \
    $tmpl_lib \
    --step-length 1.0 \
    $no_noise \
    \
    --angsens-model="$angsens" \
    --cascade-kernel="$ckernel" \
    --track-kernel="$tkernel" \
    --track-time-step="$tktimestep"
