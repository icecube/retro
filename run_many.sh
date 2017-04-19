#!/usr/bin/env bash
if [ "$HOSTNAME" = schwyz ]; then
    offset="10"
fi
if [ "$HOSTNAME" = uri ]; then
    offset="26"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    offset="42"
fi
for i in `seq 0 15`; do
	num=$(($offset + $i))
	nohup python likelihood/likelihood.py -f /fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.0000${num}.hdf5 -i 300 &
done
