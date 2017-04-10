#!/usr/bin/env bash
for dom in `seq 0 59`; do
    python sum_tables.py tables/full/retro_nevts1000_DC_DOM$dom.fits
done
