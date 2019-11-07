for i in {0..124}
do
    python step4_cluster_eval.py --cluster-idx $i \
        --dir /data/icecube/retro/tables/spice_3.2.1/no_phi_absdpdir/ \
        --centroids /data/icecube/retro/tables/spice_3.2.1/no_phi_absdpdir/kmcuda_4000_clusters_centroids_rand.npy --overwrite
done
