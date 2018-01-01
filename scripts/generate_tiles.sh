#!/bin/bash

xmin=-700
xmax=700

ymin=-700
ymax=700

zmin=-800
zmax=600

dx=100
dy=100
dz=100

nxtiles=$(( (xmax - xmin) / dx ))
nytiles=$(( (ymax - ymin) / dy ))
nztiles=$(( (zmax - zmin) / dz ))

echo "nxtiles=$nxtiles, nytiles=$nytiles, nztiles=$nztiles"

for xtile in {13..0}
do
	x0=$(( xmin + xtile * dx ))
	x1=$(( x0 + dx ))
	for ytile in {13..0}
	do
		y0=$(( ymin + ytile * dy ))
		y1=$(( y0 + dy ))

		for ztile in {13..0}
		do
			z0=$(( zmin + ztile*dz ))
			z1=$(( z0 + dz ))
			name=$( printf "tdi01a%02d_%02d_%02d" ${xtile} ${ytile} ${ztile} )
			echo "/storage/home/jll1062/src/retro/retro/test_tindep.py --tables-dir /gpfs/scratch/jll1062/retro_tables --geom-fpath /storage/home/jll1062/src/retro/retro/data/geo_array.npy --n-phibins 80 --x-lims $x0 $x1 --y-lims $y0 $y1 --z-lims $z0 $z1 --binwidth 1 --oversample 1 --antialias 1" | qsub -A dfc13_a_t_bc_default -l nodes=1:ppn=1 -l mem=5gb -l walltime=8:00:00 -d /storage/home/jll1062/src/retro/retro -j oe -o "/gpfs/scratch/jll1062/logs-retro_tables/${name}.log"
			#echo "/storage/home/jll1062/src/retro/retro/test_tindep.py --tables-dir /gpfs/scratch/jll1062/retro_tables --geom-fpath /storage/home/jll1062/src/retro/retro/data/geo_array.npy --n-phibins 80 --x-lims $x0 $x1 --y-lims $y0 $y1 --z-lims $z0 $z1 --binwidth 1 --oversample 1 --antialias 1" | qsub -A cyberlamp -l qos=cl_open -l nodes=1:ppn=1 -l mem=5gb -l walltime=8:00:00 -d /storage/home/jll1062/src/retro/retro -j oe -o "/gpfs/scratch/jll1062/logs-retro_tables/${name}.log"
		done
	done
done
