#!/bin/bash
for i in {0..124}
do
    python step6_assign_indices.py --cluster-idx $i --dir /data/icecube/retro/tables/spice_3.2.1/no_phi_absdpdir/
done
