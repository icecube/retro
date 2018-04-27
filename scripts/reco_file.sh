export PATH=/storage/home/pde3/anaconda2/bin:$PATH
dir=$1
file=$2
# ACI:
#~/retro/scripts/reco.sh /gpfs/group/dfc13/default/sim/retro/14600/$dir.$file 0 /gpfs/scratch/pde3/retro/recos/2018.04.26_pegleg/14600/$dir.$file
# ET:
~/retro/scripts/reco.sh /data/icecube/sim/ic86/retro/14600/$dir.$file 0 /data/peller/retro/recos/2018.04.21_reproduce/14600/$dir.$file
