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
    echo "Extracting file: \"$full_i3_filepath\""
    outdir="/tmp/."$( echo "$full_i3_filepath" | sed 's/\.i3.*//' )
    "$mydir"/../retro/i3processing/extract_events.py \
        --triggers I3TriggerHierarchy \
        --truth \
        --pulses SplitInIcePulses \
        --recos Pegleg_Fit_MN CascadeLast_DC FiniteRecoFit L4_ToIEval2 L4_iLineFit LineFit MM_DC_LineFitI_MM_DC_Pulses_1P_C05 MPEFit PoleMuonLinefit SPEFit2 SPEFit2_DC SPEFitSingle_DC DipoleFit_DC L4_ToI L4_ToIEval3 L5_SPEFit11 LineFit_DC MM_IC_LineFitI MPEFitMuEX PoleMuonLlhFit SPEFit2MuEX_FSS SPEFitSingle ToI_DC \
        --fpath "$full_i3_filepath" \
        --outdir "$outdir" &
}

if [ -n "$root_dir" ] ; then
    find "$root_dir" -name "*.i3*" | sort -V | while read full_i3_filepath ; do
        while (( 1 )) ; do
            (( $( jobs -r | wc -l ) < $num_subprocs )) && break
            sleep 3
        done
        runit "$full_i3_filepath"
    done
else
    runit "$full_i3_filepath"
fi

wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
