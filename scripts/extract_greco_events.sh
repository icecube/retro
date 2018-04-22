#!/bin/bash

chunksize=100
based="/data/icecube/sim/ic86/i3/l7_greco"
start_dt=$( date +'%Y-%m-%dT%H%M%z' )

for flavdir in 12600 14600 16600
do
	num_files=$( ls ${based}/${flavdir}/*${flavdir}.*.i3.bz2 | wc -l )
	for offset in $( seq 0 $chunksize $num_files )
	do
		for f in $( ls -v ${based}/${flavdir}/*${flavdir}.*.i3.bz2 | head -$(( $offset + $chunksize )) | tail -$chunksize )
		do
			f=$( basename "$f" )
			printf "$f  "
			i=$( echo "$f" | sed -e 's/.*'"${flavdir}"'\.//' -e 's/\.[0-9]*\.i3\.bz2//' -e 's/[0]*//' )
			[ -z "$i" ] && i=0
			j=$( echo "$f" | sed -e 's/.*'"${flavdir}"'\.[0-9]*\.[0]*//' -e 's/\.i3\.bz2//' )
			[ -z "$j" ] && j=0
			printf "${i}.${j}\n"
			~/src/retro/retro/i3processing/extract_events.py \
				--truth \
				--triggers I3TriggerHierarchy \
				--pulses OfflinePulses \
				--recos SPEFit2 Pegleg_Fit_LB_3Iter Pegleg_Fit_MN Pegleg_Fit_SP_3Iter \
				--fpath "$based/$flavdir/$f" \
				--outdir /data/icecube/sim/ic86/retro/${flavdir}/${i}.${j} &
		done
		wait
	done
done
wait

stop_dt=$( date +'%Y-%m-%dT%H%M%z' )
echo "Started at ${start_dt}, ended at ${stop_dt}"
