#!/bin/bash

timestamp="$( date +%Y-%m-%dT%H%M%z )"

scripts_dir="$( dirname $0 )"
retro_dir="$( dirname $scripts_dir )"
START="$1"
shift
STEP="$1"
shift
events_root="$*"

# -- Tables -- #

case $HOSTNAME in
    schwyz)
        myhostname=ET ;;
    uri)
        myhostname=ET ;;
    unterwalden)
        myhostname=ET ;;
    luzern)
        myhostname=ET ;;
    *)
        myhostname=$HOSTNAME ;;
esac

if [ "$myhostname" = "ET" ] ; then
    conda activate
    tdi0=""
    #tdi0="--tdi /data/icecube/retro/tables/tdi/tdi_table_873a6a13_tilt_on_anisotropy_off"
    #tdi0="--tdi /data/icecube/retro/tables/tdi/tdi_table_873a6a13_tilt_on_anisotropy_on"
    tdi1=""

    # -- Mie tables: stacked, template compressed -- #

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/stacked"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Mie tables: separate, template compressed -- #

    #proto="/home/icecube/retro/tables/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tmpl_lib="--template-library /home/icecube/retro/tables/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; uncompressed (high stats) -- #

    #proto="/home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    #tmpl_lib=""
    #tblkind="ckv_uncompr"

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #

    proto="/home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"
    gcd="/data/icecube/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl"

else
    conda activate
    tdi0=""
    #tdi0="--tdi /gpfs/group/dfc13/default/retro/tables/tdi_table_873a6a13_tilt_on_anisotropy_off"
    #tdi0="--tdi /gpfs/group/dfc13/default/retro/tables/tdi_table_873a6a13_tilt_on_anisotropy_on"
    tdi1=""

    # -- Mie tables: separate, template compressed -- #

    #proto="/gpfs/scratch/pde3/large_5d_notilt_combined/large_5d_notilt_string_{subdet}_depth_{depth_idx}"
    #tmpl_lib="--template-library /gpfs/scratch/pde3/large_5d_notilt_combined/ckv_dir_templates.npy"
    #tblkind="ckv_templ_compr"

    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #

    proto="/gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"
    gcd="/gpfs/group/dfc13/default/gcd/mc/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl"

fi

#    --gcd "GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl" \

# NOTE:
#
# for DRAGON MC, use
#   --pulses "InIcePulses" \
#   --hits "pulses/InIcePulses" \
# and for GRECO MC, use
#   --pulses "OfflinePulses" \
#   --hits "pulses/OfflinePulses" \
#
# If pulses provided are photons, then specify e.g.
#   --angsens-model 9
# or
#   --angsens-model h2-50cm
# etc.

#python -m cProfile  \
#kernprof -l -v \
$retro_dir/retro/reco.py \
    --method crs_prefit \
    --filter 'event["header"]["L5_oscNext_bool"]' \
    \
    --gcd $gcd \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    $tmpl_lib \
    $tdi0 \
    $tdi1 \
    \
    --events-root $events_root \
    --agg-start "$START" \
    --agg-step $STEP \
    \
    --pulses "SplitInIcePulses" \
    --triggers "I3TriggerHierarchy" \
    --hits "pulses/SplitInIcePulses"

wait
