string=$1
start_idx=$2
export CUDA_VISIBLE_DEVICES=$3
stop_idx=$((start_idx + 9))
for idx in $(seq $start_idx $stop_idx)
do
	echo $string $idx
	python ~/retro/table_compression/assign_idices.py --string $string --depth-idx $idx --nfs > /dev/null
done
