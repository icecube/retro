#!/usr/bin/env bash
for dom in `seq 0 59`; do
    python smooth_tables.py tables/full1000/retro_nevts1000_IC_DOM${dom}_r_cz_t.fits
done
