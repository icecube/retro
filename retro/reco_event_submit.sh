#!/usr/bin/env bash
for i in {0002..0999}
do
    echo "~/retro/retro/reco_event_batch.sh $i 2" | qsub -A cyberlamp \
-l qos=cl_open \
-l nodes=1:ppn=1 \
-l mem=8000mb \
-l walltime=16:00:00 \
-N reco_$i \
-o /gpfs/scratch/pde3/retro/test_event2/log/run_$i.log \
-e /gpfs/scratch/pde3/retro/test_event2/log/run_$i.err
done
#    echo "~/retro/retro/reco_event_batch.sh $i" | qsub -A open \
#    echo "~/retro/retro/reco_event_batch.sh $i" | qsub -A dfc13_a_t_bc_default \
