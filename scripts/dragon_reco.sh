
# submitted:
# mod 12 14 16
#  0  x  x  x
#  1        x (only 100)
#  2  x  x  x (only 100)
#  3  x  x  x (stale file handles)
#  4

modulo=4
simdir=/gpfs/group/dfc13/default/sim/retro/level5p_merged
outdir=/gpfs/scratch/pde3/retro/recos/level5pt_merged
#queue=cyberlamp
#queue=open
queue=dfc13_b_g_lc_default
#for flav in 16600
for flav in 12600 14600 16600
do
    for dir in $simdir/$flav/*
    do
        basename=`basename $dir`
        echo "/storage/home/pde3/retro/scripts/reco.sh $dir $modulo $outdir/$flav/$basename/" \
            | qsub -A $queue \
            -l nodes=1:ppn=1 \
            -l pmem=4000mb \
            -l walltime=48:00:00 \
            -N r$flav.$basename \
            -o /gpfs/scratch/pde3/retro/log/$flav.$basename.log \
            -e /gpfs/scratch/pde3/retro/log/$flav.$basename.err
done
done
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A cyberlamp \
#-l qos=cl_open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A dfc13_b_g_sc_default \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A open \
#-l qos=cl_open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A cyberlamp \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A dfc13_a_g_sc_default \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A cyberlamp -l qos=cl_open \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A dfc13_b_g_lc_default \
#    echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $file" | qsub -A open \
