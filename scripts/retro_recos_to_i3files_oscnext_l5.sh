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

# Since we're search for _all_ npy files, we can find the same "recos" (and
# hence the associated "events") dir multiple times since there can be multiple
# retro_*.npy files in each such dir. This file stores which dirs we've
# processed
tmpf=$( tempfile )

find "$rootdir" -type f -name "retro_*.npy" | while read recofile ; do
    recos_dir=$( dirname "$recofile" )
    if [ $( basename "$recos_dir" )  != "recos" ]; then
        continue
    fi
    eventsdir=$( dirname "$recos_dir" )

    # Continue if we've already found this directory
    grep -e "^${eventsdir}$" "$tmpf" && continue

    # NOTE: following is not needed: if --recos is not supplied as an arg to
    # the python script, it automatically grabs all retro_*.npy files in the
    # recos dir

    ## Search for all retro recos in the recos dir to populate to the i3 file
    #recos=$(
    #    find "$recos_dir" -mindepth 1 -maxdepth 1 -type f -name "retro_*.npy" | while read rfile ; do
    #        basename "$rfile" | sed 's/\.npy$//'
    #    done
    #)
    ## Concatenate reco name with \n characters to allow `sort` to work
    #recos_=""
    #for reco in $recos ; do
    #    recos_="$reco\n$recos_"
    #done
    ## Sort and then combine reco names onto a single line
    #recos=$( printf $recos | sort -V | xargs echo )

    while (( $( jobs -r | wc -l ) >= $num_jobs )) ; do
        sleep 1
    done

    echo "$eventsdir" | tee -a "$tmpf"
    echo "running..."
    "$mydir"/../retro/i3processing/retro_recos_to_i3files.py \
        --eventsdir "$eventsdir" \
        --point-estimator median \
        --overwrite \
        &
done

while (( $( jobs -r | wc -l ) >= $num_jobs )) ; do
    sleep 1
done

rm -f "$tmpf"
