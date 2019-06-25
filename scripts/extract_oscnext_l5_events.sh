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
    echo "Extracting from file --> to dir: \"$full_i3_filepath\" -->  \"$outdir\""
    outdir=$( echo "$full_i3_filepath" | sed 's/\.i3.*//' )
    "$mydir"/../retro/i3processing/extract_events.py \
        --fpath "$full_i3_filepath" \
        --additional-keys "L5_oscNext_bool" \
        --outdir "$outdir" \
        --truth \
        --triggers I3TriggerHierarchy \
        --pulses SplitInIcePulses \
        --recos LineFit_DC L5_SPEFit11 \
        &
}

if [ -n "$root_dir" ] ; then
    find "$root_dir" -name "oscNext*.i3*" | sort -V | while read full_i3_filepath ; do
        while (( $( jobs -r | wc -l ) >= $num_subprocs )) ; do
            sleep 0.2
        done
        runit "$full_i3_filepath"
    done
else
    runit "$full_i3_filepath"
fi

wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
