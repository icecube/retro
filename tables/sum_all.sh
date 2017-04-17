#!/usr/bin/env bash
for dom in `seq 0 59`; do
    python sum_tables.py tables/full1000/retro_nevts1000_DC_DOM$dom.fits
done
