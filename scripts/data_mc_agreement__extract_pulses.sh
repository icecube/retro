#!/bin/bash

retro_scripts_dir=$( dirname $0 )
script="$retro_scripts_dir"/../retro/utils/data_mc_agreement__extract_pulses.py
script_base=$( basename "$script" )

usage="$(basename "$0")
    [-h|--help]
    --data-rootdir DATA_ROOTDIR
    --pass PASS_NUM
    --level LEVEL
    --proc-ver PROC_VER
    --pulse-series PULSE_SERIES

Extract pulses from each data & MC set into numpy arrays and save to files
named as <PULSE_SERIES>__{events,doms,pulses}_array.npy at the head directory
for each set.

See $script_base for more details of the arrays that script produces.

Arguments:
    -h|--help    show this help text

    --data-rootdir DATA_ROOTDIR   Dir at which \"ana\/LE/oscnext\" lives. E.g.,
                                  \"/data\" at Wisconsin, \"/data/icecube\" on
                                  ET workstations

    --pass PASS                   Processing pass. E.g., specify \"2\" for \"pass2\"

    --level LEVEL                 Processing level. E.g., specify \"5\" for \"level5\"

    --proc-ver PROC_VER           Processing version. E.g., \"01.03\"

    --pulse-series PULSE_SERIES   Pulse series name to extract. E.g.,
                                  \"SRTTWOfflinePulsesDC\"
"

# -- Parse command line args -- #

# https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
while [[ $# -gt 0 ]] ; do
    key="$1"
    case $key in
        --data-rootdir)
        data_rootdir="$2"
        shift  # past argument
        shift  # past value
        ;;
        --pass)
        pass="$2"
        shift  # past argument
        shift  # past value
        ;;
        --level)
        level="$2"
        shift  # past argument
        shift  # past value
        ;;
        --proc-ver)
        proc_ver="$2"
        shift  # past argument
        shift  # past value
        ;;
        --pulse-series)
        pulse_series="$2"
        shift  # past argument
        shift  # past value
        ;;
        -h|--help)
        echo "$usage"
        exit
        ;;
        *)  # unknown option
        echo "$usage" >&2
        exit 1
        ;;
    esac
done

# -- Monte Carlo -- #

for d in \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/muongun/level${level}_v${proc_ver}/139011" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/noise/level${level}_v${proc_ver}/888003" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/120000" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/140000" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/160000" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/120001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/120002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/120003" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/120004" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/140001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/140002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/140003" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/140004" \
    \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/160001" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/160002" \
    "${data_rootdir}/ana/LE/oscNext/pass${pass}/genie/level${level}_v${proc_ver}/160003" \
    ;
do
    echo "Extracting \"${pulse_series}\" pulses from \"$d\""
    time $script --pulse-series $pulse_series --indir "$d" --outdir "$d" --processes=1 &
    sleep 0.2
done

# -- Data -- #

for y in {12..18} ; do
    d="${data_rootdir}/ana/LE/oscNext/pass${pass}/data/level${level}_v${proc_ver}/IC86.${y}/"
    echo "Extracting \"${pulse_series}\" pulses from \"$d\""
    time $script --pulse-series $pulse_series --indir "$d" --outdir "$d" --processes=1 &
    sleep 0.2
done

wait
