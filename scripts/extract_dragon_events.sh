#!/bin/bash

chunksize=100
based="/data/icecube/sim/ic86/i3/l5b_dragon_hybridreco"
baseoutd="/data/icecube/sim/ic86/retro"
start_dt=$( date +'%Y-%m-%dT%H%M%z' )

for flavdir in 1260 1460 1660
do
	if [ "$flavdir" -eq 1260 ]
	then
		flav="nue"
	elif [ "$flavdir" -eq 1460 ]
	then
		flav="numu"
	else
		flav="nutau"
	fi

	num_files=$( ls ${based}/${flav}/${flavdir}/*${flavdir}.*.i3.bz2 | wc -l )
	for offset in $( seq 0 $chunksize $num_files )
	do
		#for f in $( ls -v ${based}/${flav}/${flavdir}/*${flavdir}.*.i3.bz2 | tail -1 )
		for f in $( ls -v ${based}/${flav}/${flavdir}/*${flavdir}.*.i3.bz2 | head -$(( $offset + $chunksize )) | tail -$chunksize )
		do
			fbase=$( basename "$f" )
			printf "$fbase  "
			i=$( echo "$fbase" | sed -e 's/.*\.[0]*'"${flavdir}"'\.[0]*//' -e 's/\.*i3\.bz2//' )
			[ -z "$i" ] && i=0
			printf "${i}\n"
			~/src/retro/retro/i3processing/extract_events.py \
				--truth \
				--triggers I3TriggerHierarchy \
				--pulses OfflinePulses \
				--recos SPEFit2 IC86_Dunkman_L6_MultiNest7D IC86_Dunkman_L6_MultiNest8D \
				--fpath "$f" \
				--outdir ${baseoutd}/${flavdir}/${i} &
		done
		wait
	done
done

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
