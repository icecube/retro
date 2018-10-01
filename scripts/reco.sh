#!/bin/bash
export PATH=~/anaconda2/bin:$PATH

timestamp="$( date +%Y-%m-%dT%H%M%z )"

events_base="$1"
modulo="$2"
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
    tdi0=""
    #tdi0="--tdi /data/icecube/retro/tables/tdi/tdi_table_873a6a13_tilt_on_anisotropy_off"
    tdi1=""

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

    #proto="/data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    #tmpl_lib="--template-library /data/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

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

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #

    proto="/home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"

else
    tdi0=""
    #tdi0="--tdi /gpfs/group/dfc13/default/retro/tables/tdi_table_873a6a13_tilt_on_anisotropy_off"
    #tdi0="--tdi /gpfs/group/dfc13/default/retro/tables/tdi_table_873a6a13_tilt_on_anisotropy_on"

    tdi1=""

    # -- Lea tables: 80 clusters, template compressed -- #

    #proto="/gpfs/group/dfc13/xv/retro/tables/tilt_on_anisotropy_on_noazimuth_80/cl{cluster_idx}"
    #tmpl_lib="--template-library /gpfs/group/dfc13/xv/retro/tables/tilt_on_anisotropy_on_noazimuth_80/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Mie tables: separate, template compressed -- #

    #proto="/gpfs/scratch/pde3/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tmpl_lib="--template-library /gpfs/scratch/pde3/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #

    proto="/gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"

fi


#python -m cProfile  \
#kernprof -l -v \
~/retro/retro/reco.py \
    --outdir "$outdir" \
    --method "crs_prefit_mn" \
    \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --gcd "GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl" \
    --norm-version "binvol2.5" \
    $tdi0 \
    $tdi1 \
    $tmpl_lib \
    --step-length 1.0 \
    --use-doms "all" \
    \
    --events-base "$events_base" \
    --start "$modulo" \
    --step 10 \
    --pulses "OfflinePulses" \
    --recos "SPEFit2" \
    --triggers "I3TriggerHierarchy" \
    --hits "pulses/OfflinePulses" \
    --angsens-model "h2-50cm" \
    --truth

#    --pulses "InIcePulses" \
#    --hits "pulses/InIcePulses" \
wait
