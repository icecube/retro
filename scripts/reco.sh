#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

events_base="$1"
start_idx="$2"
outdir="$3"

mkdir -p "$outdir"

#proto="/fastio2/icecube/retro/tables/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib=""

#proto="/gpfs/scratch/jll1062/retro_tables/stacked"
#tmpl_lib="--template-library /gpfs/scratch/jll1062/retro_tables/ckv_dir_templates.npy"

#proto="/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
#tmpl_lib="--template-library /data/icecube/retro_tables/large_5d_notilt_combined/ckv_dir_templates.npy"

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

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; uncompressed (low stats) -- #

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60/cl{cluster_idx}"
    #tmpl_lib=""
    #tblkind="ckv_uncompr"

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; uncompressed (high stats) -- #

    #proto="/home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    #tmpl_lib=""
    #tblkind="ckv_uncompr"

else
    # -- Lea tables: 80 clusters, template compressed -- #

    proto="/gpfs/group/dfc13/xv/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    tmpl_lib="--template-library /gpfs/group/dfc13/xv/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"

    # -- Mie tables: separate, template compressed -- #

    #proto="/gpfs/scratch/pde3/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tmpl_lib="--template-library /gpfs/scratch/pde3/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"
fi

#no_noise="--no-noise"
no_noise=""

importance_sampling="--importance-sampling"
#importance_sampling=""

#consteff="--const-eff"
consteff=""


#kernprof -l -v ~/retro/retro/reco.py \
~/retro/retro/reco.py \
    --outdir "$outdir" \
    --spatial-prior SPEFit2 \
    --temporal-prior SPEFit2 \
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
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    --gcd "GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl" \
    --norm-version "binvol2.5" \
    $tmpl_lib \
    --step-length 1.0 \
    $no_noise \
    \
    --cascade-kernel "scaling_aligned_one_dim" \
    --cascade-angle-prior "log_normal" \
    --track-kernel "pegleg" \
    --track-time-step 1.0 \
    \
    --events-base "$events_base" \
    --start-idx "$start_idx" \
    --num-events 100 \
    --pulses "OfflinePulses" \
    --recos "SPEFit2" \
    --triggers "I3TriggerHierarchy" \
    --hits "pulses/OfflinePulses" \
    --angsens-model "h2-50cm" \
    --truth

wait
