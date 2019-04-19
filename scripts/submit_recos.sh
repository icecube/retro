mc=149002 
mkdir -p /gpfs/scratch/pde3/retro/log/
for dir in {000009..000011}
do
    for sidx in {0..99}
    do
        echo "/storage/home/pde3/retro/scripts/reco_file.sh $dir $sidx $mc" | qsub -A cyberlamp \
-l nodes=1:ppn=1 \
-l pmem=8000mb \
-l walltime=24:00:00 \
-N r$dir.$sidx \
-o /gpfs/scratch/pde3/retro/log/r$mc:$dir.$sidx.log \
-e /gpfs/scratch/pde3/retro/log/r$mc:$dir.$sidx.err \
-m n
    done
done
