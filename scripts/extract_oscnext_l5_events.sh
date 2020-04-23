#!/bin/bash

# Provide the root directory in which to search for all *.i3* files (or provide
# a single file) as the first argument to this script.
#
# If a second argument is provided, it is the number of processes to spawn for
# extraction. If not provided, default number of processes spawned is found by
#   $ nproc --all
#
# All extracted events are placed in the same directory as the source i3 file
# and a subdirectory with the same name as the source i3 file but with the .i3*
# extension(s) stripped.
if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    retro_gcd_dir=/data/icecube/retro_gcd
    simulation_gcd="/data/icecube/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
elif [[ "$HOSTNAME" == *".aci.ics.psu.edu" ]]; then
    retro_gcd_dir=/gpfs/group/dfc13/default/retro_gcd
    simulation_gcd="/gpfs/group/dfc13/default/gcd/mc/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
else
    retro_gcd_dir=~/retro_gcd
    simulation_gcd="/data/sim/DeepCore/2018/pass2/gcd/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
fi

start_dt=$( date +'%Y-%m-%dT%H%M%z' )

if [ -d "$1" ] ; then
    root_dir="$1"
    echo "Searching for .i3* files in: $root_dir"
    if [ -n "$2" ] ; then
        num_subprocs=$2
    else
        num_subprocs=$( nproc --all )
    fi
    echo "Number of subprocesses: $num_subprocs"
else
    full_i3_filepath="$1"
    num_subprocs=1
fi

mydir=$( dirname "$0" )

function runit () {
    full_i3_filepath="$1"
    i3_basename=$( basename $full_i3_filepath )
    if $( echo "$i3_basename" | grep -i "genie" >/dev/null 2>&1 ) ; then
        truth_flag="--truth"
        gcd=$simulation_gcd
    elif $( echo "$i3_basename" | grep -i "muongun" >/dev/null 2>&1 ) ; then
        truth_flag="--truth"
        gcd=$simulation_gcd
    elif $( echo "$i3_basename" | grep -i "oscNext_noise" >/dev/null 2>&1 ) ; then
        truth_flag="--truth"
        gcd=$simulation_gcd
    elif $( echo "$i3_basename" | grep -i "oscNext_data" >/dev/null 2>&1 ) ; then
        truth_flag=""
        gcd=$( find $( dirname "$full_i3_filepath" ) -mindepth 1 -maxdepth 1 -type f -iname "*GCD*.i3*" )
    else
        echo "dunno what to do with $full_i3_filepath"
        exit 1
    fi
    echo "i3_basename=$i3_basename gcd=$gcd"

    echo "Extracting from file --> to dir: \"$full_i3_filepath\""
    "$mydir"/../retro/i3processing/extract_events.py \
        --i3-files "$full_i3_filepath" \
        --retro-gcd-dir $retro_gcd_dir \
        --gcd $gcd \
        --additional-keys "L5_oscNext_bool" \
        --pulses SplitInIcePulses SRTTWOfflinePulsesDC \
        --triggers I3TriggerHierarchy \
        --recos LineFit_DC L5_SPEFit11 \
        $truth_flag \
        &
        #    SplitInIcePulses \
        #    SRTInIcePulses \
        #    InIcePulses \
        #    L4_TWPulses \
        #    L4_TWPulses_DCFid_DTW200 \
        #    L5_WideCorridorCutPulses \
        #    RTTWOfflinePulses_FR_WIMP \
        #    TWOfflinePulsesDC \
        #    TWOfflinePulses_FR_WIMP \
        #    TWRTVetoSeries \
}

if [ -n "$root_dir" ] ; then
    if [ -d "$root_dir" ] ; then
        find "$root_dir" -iregex ".*oscNext_.*[0-9]\.i3.*" | sort -V | while read full_i3_filepath ; do
            while (( $( jobs -r | wc -l ) >= $num_subprocs )) ; do
                sleep 0.2
            done
            runit "$full_i3_filepath"
        done
    else
        runit "$root_dir"
    fi
else
    runit "$full_i3_filepath"
fi

while (( $( jobs -r | wc -l ) > 0 )) ; do
    sleep 1
done

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
