#!/usr/bin/env bash
if [ "$HOSTNAME" = schwyz ]; then
    dom="0 10 20 30 40 50"
    det='DC'
fi
if [ "$HOSTNAME" = uri ]; then
    dom="0 10 20 30 40 50"
    det='IC'
fi
if [ "$HOSTNAME" = unterwalden ]; then
    dom=""
fi
for d in $dom
do
    for i in `seq 0 9`; do
        DOM=$(($d + $i))
        nohup python generate_table.py --nevts 1000 --subdet $det --dom $DOM > /dev/null 2>&1 &
    done
    wait
done
