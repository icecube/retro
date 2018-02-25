#!/bin/bash

# GCD file to be used
GCD=$I3_TESTDATA/sim/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz

#NAME=downgoing_track
NAME=horizontal_track

# simulation-V05

# SIMULATE
# horizontal
#./sim.py -x=0 -y=0 -z=-350 --particle-type=EMinus --energy=20 --coszen=0 --azimuth=0 --no-hole-ice --outbase=${NAME}_step1 --num-events=100000 -g $GCD --device 0
# downgoing
#./sim.py -x=0 -y=0 -z=-300 --particle-type=EMinus --energy=20 --coszen=1 --azimuth=0 --no-hole-ice --outbase=${NAME}_step1 --num-events=100000 -g $GCD --device 0

# DAQ (needs simulation-V05)
#./photons_to_pe.py -i ${NAME}_step1.i3.bz2 -o ${NAME}_step2.i3.bz2 --holeice $I3_SRC/ice-models/resources/models/angsens/as.h2-50cm -g $GCD

# combo_stable ?

GCD=$I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz
# Processing and Filtering
$I3_BUILD/filterscripts/resources/scripts/SimulationFiltering.py --disable-gfu -i ${NAME}_step2.i3.bz2 -g $GCD -o ${NAME}_step3.i3.bz2

# SRT hit cleaning
./hit_cleaning.py -i ${NAME}_step3.i3.bz2 -o ${NAME}_step4.i3.bz2 -g $GCD


# shit that didn't work
#./deepcoreL2example.py -s -i ${NAME}_step2.i3.bz2 -g $I3_TESTDATA/sim/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz -o ${NAME}_step3.i3.bz2
#./dc_fit.py -i ${NAME}_step3.i3.bz2 -o ${NAME}_step4.i3.bz2 -g $I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz
#$I3_BUILD/filterscripts/resources/scripts//offlineL2/process.py -i ${NAME}_step3.i3.bz2 -g $I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz -o ${NAME}_step4.i3.bz2
