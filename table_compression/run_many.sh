string=$1
start_idx=$2
stop_idx=$((start_idx + 9))
for idx in $(seq $start_idx $stop_idx)
do
	echo $string $idx
	python ~/retro/table_compression/pca_eval.py --string $string --depth-idx $idx
done
