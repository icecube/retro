#!/bin/bash

data_rootdir="/data/icecube"
pass=2
level="level5_v01.03"
pulse_series=SplitInIcePulses

mydir=$( dirname $0 )
script="$mydir"/../retro/utils/data_mc_agreement__extract_pulses.py

# -- Monte Carlo -- #

for d in \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/muongun/${level}/139011" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/noise/${level}/888003" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/120000" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/140000" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/160000" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/120001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/120002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/120003" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/120004" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/140001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/140002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/140003" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/140004" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/160001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/160002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/${level}/160003" \
    ;
do
    echo "Extracting pulses from \"$d\""
    time $script --pulse-series $pulse_series --indirs "$d" --outdir "$d" --serial
    sleep 0.2
done

# -- Data -- #

for y in {12..18} ; do
    d="${data_rootdir}/ana/LE/oscNext/pass${pass}/data/${level}/IC86.${y}/"
    echo "Extracting pulses from \"$d\""
    time $script --pulse-series $pulse_series --indirs "$d" --outdir "$d"
    sleep 0.2
done
