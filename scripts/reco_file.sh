export PATH=/storage/home/pde3/anaconda2/bin:$PATH
dir=$1
file=$2
modulo=$3
mc=14600
if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    ~/retro/scripts/reco.sh /data/icecube/sim/ic86/retro/$mc/$dir.$file $modulo /data/peller/retro/recos/2018.07.18_lea_test/$mc/$dir.$file
else
    ~/retro/scripts/reco.sh /gpfs/group/dfc13/default/sim/retro/$mc/$dir.$file $modulo /gpfs/scratch/pde3/retro/recos/2018.09.07_finerMN/$mc/$dir.$file
fi
