#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

extraname=$1

basedir="/data/justin/retro/dom_distributions"

use_doms="all"
sim_to_test="lea_upgoing_muon"
angsens="9"
#sim_to_test="mie_upgoing_muon"
#angsens="h2-50cm"
norm="binvol2"
tkernel="table_e_loss"
tktimestep="1.0"
ckernel="aligned_one_dim"

gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.bz2"

#no_noise="--no-noise"
no_noise=""

# -- Tables -- #

if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/stacked"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"

    #proto="/fastio/icecube/retro/tables/large_5d_notilt_combined/stacked/"
    #tmpl_lib="--template-library /fastio/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tblkind="ckv_templ_compr"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    #tblkind="ckv_uncompr"
    #tmpl_lib=""

    proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    tblkind="ckv_templ_compr"
    tmpl_lib="--template-library /data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
else
    proto="/gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/stacked/"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
fi

# TODO: change outdir based on whether simulating muon or cascade
[ "$no_noise" ] && noisebool="true" || noisebool="false"
[ "$extraname" ] && extraname=",${extraname}"

outdir="$basedir/sim=${sim_to_test},as=${angsens},n=${norm},tk=${tkernel},ts=${tktimestep},tbl=${tblkind},noise=${noisebool}${extraname}"
mkdir -p "$outdir"

~/src/retro/retro/retro_dom_pdfs.py \
    --outdir="$outdir" \
    --sim-to-test="$sim_to_test" \
    \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "$use_doms" \
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
