#!/usr/bin/env bash
for i in {0601..0999}
do
    echo "~/retro/retro/reco_event_batch.sh $i" | qsub -A cyberlamp \
-l nodes=1:ppn=1 \
-l mem=5000mb \
-l walltime=8:00:00 \
-N reco_$i \
-o ~/retro/retro/log/run_$i.log \
-e ~/retro/retro/log/run_$i.err
done
#    echo "~/retro/retro/reco_event_batch.sh $i" | qsub -A open \
#    echo "~/retro/retro/reco_event_batch.sh $i" | qsub -A dfc13_a_t_bc_default \
