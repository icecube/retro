#!/bin/bash

MYDIR=`dirname $0`
PROCDIR="${MYDIR}/../retro/i3processing"

#==============================================================================
# User-defined simulation parameters
#==============================================================================

#GCD="$I3_TESTDATA/sim/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz"
GCD="$I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz"

ICE_MODEL="$I3_SRC/clsim/resources/ice/spice_mie"
#ICE_MODEL="$I3_SRC/clsim/resources/ice/spice_lea"

HOLE_ICE_MODEL="$I3_SRC/ice-models/resources/models/angsens/as.h2-50cm"

USE_GEANT4=false

PARTICLE="MuMinus"
E=20
X=0
Y=0
Z=-400
CZ=-1.0
AZ=0

NUM_EVENTS=100

#==============================================================================
# Derive things from above
#==============================================================================

ICE="$( basename $ICE_MODEL )"
HOLEICE="$( basename $HOLE_ICE_MODEL )"
GCD_MD5=$( cat $GCD | gunzip | md5sum | sed 's/ .*//' )
if $USE_GEANT4
then
    GEANT4="--use-geant4"
else
    GEANT4=""
fi

NAME="${PARTICLE}_energy${E}_x${X}_y${Y}_z${Z}_cz${CZ}_az${AZ}_ice_${ICE}_holeice_${HOLEICE}_gcd_md5_${GCD_MD5:0:8}_geant_${USE_GEANT4}_nsims${NUM_EVENTS}"

echo "NAME: ${NAME}"

#==============================================================================
# Scripty bits
#==============================================================================

# Step 1: SIMULATE
if [ ! -e "${NAME}_step1.i3.bz2" ]
then
    echo "Running Step 1..."
    time ${PROCDIR}/simulate_particle.py \
        --particle-type=$PARTICLE \
        -x=$X -y=$Y -z=$Z --energy=$E --coszen=$CZ --azimuth=$AZ \
        --ice-model $ICE_MODEL \
        --hole-ice-model $HOLE_ICE_MODEL \
        --num-events=$NUM_EVENTS \
        --gcd $GCD \
        $GEANT4 \
        --run-num 1 \
        --outfile=${NAME}_step1
	echo ""
fi

# Step 2: DAQ (needs simulation-V05)
[ -e "${NAME}_step1.i3.bz2" -a ! -e "${NAME}_step2.i3.bz2" ] && \
    echo "Running Step 2..." && \
    time ${PROCDIR}/photons_to_pe.py \
        --holeice "$HOLE_ICE_PARAM" \
        -i "${NAME}_step1.i3.bz2" \
        -g "$GCD" \
        -o "${NAME}_step2.i3.bz2"
echo ""

# Step 3: Processing and Filtering
[ -e "${NAME}_step2.i3.bz2" -a ! -e "${NAME}_step3.i3.bz2" ] && \
    echo "Running Step 3..." && \
    time $I3_BUILD/filterscripts/resources/scripts/SimulationFiltering.py \
        --disable-gfu \
        -i "${NAME}_step2.i3.bz2" \
        -g "$GCD" \
        -o "${NAME}_step3.i3.bz2"
echo ""

## Step 4: SRT hit cleaning
#[ -e "${NAME}_step3.i3.bz2" -a ! -e "${NAME}_step4.i3.bz2" ] && \
#    echo "Running Step 4..." && \
#    time ${PROCDIR}/hit_cleaning.py \
#        -i "${NAME}_step3.i3.bz2" \
#        -g "$GCD" \
#        -o "${NAME}_step4.i3.bz2"
#echo ""

# shit that didn't work
#${PROCDIR}/deepcoreL2example.py -s -i ${NAME}_step2.i3.bz2 -g $I3_TESTDATA/sim/GeoCalibDetectorStatus_IC86.55697_corrected_V2.i3.gz -o ${NAME}_step3.i3.bz2
#${PROCDIR}/dc_fit.py -i ${NAME}_step3.i3.bz2 -o ${NAME}_step4.i3.bz2 -g $I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz
#$I3_BUILD/filterscripts/resources/scripts//offlineL2/process.py -i ${NAME}_step3.i3.bz2 -g $I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz -o ${NAME}_step4.i3.bz2
