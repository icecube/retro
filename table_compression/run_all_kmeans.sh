string=$1
start_idx=8
stop_idx=59
for idx in $(seq $start_idx $stop_idx)
do
	echo $string $idx
	python ~/retro/table_compression/kmeans_eval.py --string $string --depth-idx $idx --nfs --overwrite
done
