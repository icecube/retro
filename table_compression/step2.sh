for i in {0..125}
do
    python step2_pca_eval.py --cluster-idx $i --dir /data/icecube/retro/tables/spice_3.2.1/no_phi_absdpdir/ --pca /data/icecube/retro/tables/spice_3.2.1/no_phi_absdpdir/cl59/pca_increment_124
done
