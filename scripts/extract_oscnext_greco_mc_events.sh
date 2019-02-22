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

    outdir=$( echo "$full_i3_filepath" | sed 's/\.i3.*/test/' )

    "$mydir"/../retro/i3processing/extract_events.py \
        --triggers I3TriggerHierarchy \
        --truth \
        --pulses SplitInIcePulses \
        --recos CascadeLast_DC DipoleFit_DC FiniteRecoFit L4_ToI L4_ToIEval2 L4_ToIEval3 L4_iLineFit L5_SPEFit11 LineFit LineFit_DC MM_DC_LineFitI_MM_DC_Pulses_1P_C05 MM_IC_LineFitI MPEFit MPEFitMuEX PoleMuonLinefit PoleMuonLlhFit SPEFit2 SPEFit2MuEX_FSS SPEFit2_DC SPEFitSingle SPEFitSingle_DC ToI_DC \
        --fpath "$full_i3_filepath" \
        --outdir "$outdir" &

done

wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
