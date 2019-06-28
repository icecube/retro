#!/bin/bash

# Find any retro_*.npy recos and populate to the i3 files epxected to be in the
# directory above the corresponding Retro dir containing the events.npy file
# directory.
#
# arg 1
#     directory in which to search for reco files
#
# arg 2 (optional)
#     number of jobs to run in parallel; if not specified, number of cores is
#     auto-detected and used for that value

rootdir="$1"
if [ -n "$2" ] ; then
    num_jobs=$2
else
    num_jobs=$( grep -c ^processor /proc/cpuinfo )
fi

mydir=$( dirname "$0" )

tmpf=$( tempfile )

find "$rootdir" -type f -name "retro_*.npy" | while read recofile ; do
    recodir=$( dirname "$recofile" )
    if [ "$recodir" != "recos" ]; then
        continue
    fi
    eventsdir=$( dirname "$recodir" )
    grep -e "^${eventsdir}$" "$tmpf" && continue

    while (( $( jobs | wc -l ) >= $num_jobs )) ; do
        sleep 1
    done

    echo "$eventsdir" | tee -a "$tmpf"
    "$mydir"/../retro/i3processing/retro_recos_to_i3files.py \
        --eventsdir "$eventsdir" \
        --point-estimator median \
        --overwrite \
        &
done
wait

rm -f "$tmpf"
