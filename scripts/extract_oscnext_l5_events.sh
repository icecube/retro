#!/bin/bash

start_dt=$( date +'%Y-%m-%dT%H%M%z' )

chunksize=$( nproc --all )
based="/data/icecube/sim/ic86/i3/oscNext/pass2/genie/level5/"
mydir=$( dirname "$0" )

i=0
for full_i3_filepath in $( find "$based" -name "*.i3*" | sort -V )
do
    (( i++ )) ; (( i % chunksize == 0 )) && wait

    echo "$full_i3_filepath"

    outdir=$( echo "$full_i3_filepath" | sed 's/\.i3.*//' )

    "$mydir"/../retro/i3processing/extract_events.py \
        --triggers I3TriggerHierarchy \
        --truth \
        --pulses SplitInIcePulses \
        --recos L5_SPEFit11 LineFit_DC \
        --fpath "$full_i3_filepath" \
        --outdir "$outdir" &

done

wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
