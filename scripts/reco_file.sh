hostname
dir=$1
sidx=$2
mc=$3
if [ "$HOSTNAME" = "schwyz" ] || [ "$HOSTNAME" = "uri" ] || [ "$HOSTNAME" = "unterwalden" ] || [ "$HOSTNAME" = "luzern" ]; then
    ~/retro/scripts/reco_oscnext_greco.sh /data/icecube/sim/ic86/i3/oscNext/pass2/genie/level5/$mc/oscNext_genie_level5_pass2.$mc.$dir $sidx /data/peller/retro/recos/2019.03.01/oscNext/pass2/genie/level5/$mc/oscNext_genie_level5_pass2.$mc.$dir
else
    ~/retro/scripts/reco_oscnext_greco.sh /gpfs/group/dfc13/default/sim/icecube/oscNext/pass2/genie/level5/$mc/oscNext_genie_level5_pass2.$mc.$dir $sidx /gpfs/scratch/pde3/retro/recos/2019.03.04/oscNext/pass2/genie/level5/$mc/oscNext_genie_level5_pass2.$mc.$dir
fi
