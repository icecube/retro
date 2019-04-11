#!/bin/bash

start_dt=$( date +'%Y-%m-%dT%H%M%z' )

chunksize=$( nproc --all )
based="$1"
mydir=$( dirname "$0" )

i=0
find "$based" -name "*.i3*" | sort -V | while read full_i3_filepath
do
    (( i++ )) ; (( i % chunksize == 0 )) && wait

    echo "$full_i3_filepath"

    outdir=$( echo "$full_i3_filepath" | sed 's/\.i3.*//' )

    "$mydir"/../retro/i3processing/extract_events.py \
        --triggers I3TriggerHierarchy \
        --truth \
        --pulses SplitInIcePulses \
        --recos CascadeLast_DC FiniteRecoFit L4_ToIEval2 L4_iLineFit LineFit MM_DC_LineFitI_MM_DC_Pulses_1P_C05 MPEFit PoleMuonLinefit SPEFit2 SPEFit2_DC SPEFitSingle_DC DipoleFit_DC L4_ToI L4_ToIEval3 L5_SPEFit11 LineFit_DC MM_IC_LineFitI MPEFitMuEX PoleMuonLlhFit SPEFit2MuEX_FSS SPEFitSingle ToI_DC \
        --fpath "$full_i3_filepath" \
        --outdir "$outdir" &
done

wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
