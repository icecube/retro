string=$1
start_idx=0
stop_idx=59
for idx in $(seq $start_idx $stop_idx)
do
	echo $string $idx
	python ~/retro/table_compression/pca_eval.py --string $string --depth-idx $idx --nfs
done
