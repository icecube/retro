#!/usr/bin/env bash
if [ "$HOSTNAME" = schwyz ]; then
    dom="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
fi
if [ "$HOSTNAME" = uri ]; then
    dom="20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"
fi
if [ "$HOSTNAME" = unterwalden ]; then
    dom="40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59"
fi
for run in $dom
do
    nohup python generate_table.py --nevts 10000 --subdet IC --dom $dom > /dev/null 2>&1 &
    nohup python generate_table.py --nevts 10000 --subdet DC --dom $dom > /dev/null 2>&1 &
done
