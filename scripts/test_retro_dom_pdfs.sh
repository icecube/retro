#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

extraname=$1

#basedir="/data/justin/retro/dom_distributions"
basedir="/data/peller/retro/dom_distributions"

use_doms="all"
#sim_to_test="lea_upgoing_muon"
#angsens="9"
#sim_to_test="mie_upgoing_muon"
#sim_to_test="mie_horizontal_muon"
sim_to_test="lea_horizontal_muon"
angsens="h2-50cm"
norm="binvol2.5"
tkernel="table_e_loss"
tktimestep="1.0"
ckernel="aligned_one_dim"

gcd="GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.bz2"

#no_noise="--no-noise"
no_noise=""

# -- Tables -- #

if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    # -- Mie tables: stacked, template compressed -- #

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/stacked"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Mie tables: separate, template compressed -- #

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Lea tables: 80 clusters, uncompressed -- #

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    #tmpl_lib=""
    #tblkind="ckv_uncompr"

    # -- Lea tables: 80 clusters, template compressed -- #

    proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    tmpl_lib="--template-library /data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"

    # -- Lea tables: 80 clusters plus string 81 DOMs 29-60 are single-DOM tables (not clustered w/ other DOMs) -- #

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80+str81_29-60/cl{cluster_idx}"
    #tmpl_lib=""
    #tblkind="ckv_uncompr"

    # -- Lea tables: 1 table used for all DOMs (cluster 0 from above) -- #

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_1/cl{cluster_idx}"
    #tmpl_lib="--template-library /data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

else
    proto="/gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/stacked/"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
fi

# TODO: change outdir based on whether simulating muon or cascade
[ "$no_noise" ] && noisebool="true" || noisebool="false"
[ "$extraname" ] && extraname=",${extraname}"

outdir="$basedir/sim=${sim_to_test},as=${angsens},n=${norm},tk=${tkernel},ts=${tktimestep},tbl=${tblkind},noise=${noisebool}${extraname}"
mkdir -p "$outdir"

~/retro/retro/retro_dom_pdfs.py \
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
