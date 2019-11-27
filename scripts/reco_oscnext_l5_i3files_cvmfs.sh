#!/bin/bash

export PYTHONPATH=/cvmfs/icecube.opensciencegrid.org/users/peller/retro/build:$PYTHONPATH
export PATH=/cvmfs/icecube.opensciencegrid.org/users/peller/retro/build/bin:$PATH
#export PYTHONPATH=/net/cvmfs_users/peller/retro/build:$PYTHONPATH
#export PATH=/net/cvmfs_users/peller/retro/build/bin:$PATH

proto="/cvmfs/icecube.opensciencegrid.org/users/peller/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/cl{cluster_idx}"
tmpl_lib="--template-library /cvmfs/icecube.opensciencegrid.org/users/peller/retro/tables/tilt_on_anisotropy_on_noazimuth_ic80_dc60_histats/ckv_dir_templates.npy"
tblkind="ckv_templ_compr"
gcd="/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"

#python /net/cvmfs_users/peller/retro/build/retro/i3reco.py \
python /cvmfs/icecube.opensciencegrid.org/users/peller/retro/build/retro/i3reco.py \
    --gcd $gcd \
    --dom-tables-kind "$tblkind" \
    --dom-tables-fname-proto "$proto" \
    --use-doms "all" \
    $tmpl_lib \
    --input-i3-file "$1" \
    --output-i3-file "$2" \

wait
