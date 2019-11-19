#!/bin/bash

scripts_dir="$( pwd $0 )"
retro_dir="$( dirname $scripts_dir )"

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
    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #
    proto="/home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /home/icecube/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"
    gcd="/data/icecube/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl"

elif [[ $HOSTNAME = cobalt*.icecube.wisc.edu ]] ; then
    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #
    proto="/data/user/peller/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /data/user/peller/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"
    gcd="/data/sim/DeepCore/2018/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"

else
    # -- Lea tables: 80 IceCube-only clusters, 60 DeepCore-only clusters; template compressed (high stats) -- #
    proto="/gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
    tmpl_lib="--template-library /gpfs/group/dfc13/default/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
    tblkind="ckv_templ_compr"
    gcd="/gpfs/group/dfc13/default/gcd/mc/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl"

fi

# -- Reco -- #

$retro_dir/retro/i3reco.py \
    --gcd $gcd \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    $tmpl_lib \
    --input-i3-file "$1" \
    --output-i3-file "$2" \

wait
