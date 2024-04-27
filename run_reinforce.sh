set -x

NUM_DEVICES=`nvidia-smi  | grep 'NVIDIA A100 80GB PCIe' | wc -l`

echo "Found $NUM_DEVICES GPUs on this node."

count=0;

for nn_type in shallow deep; do
    for lr in 0.001 0.01 0.1 0.5; do
        if [ "$count" == "$NUM_DEVICES" ]; then
            wait
            count=0
        fi
        CUDA_VISIBLE_DEVICES=$count python3 train.py -c 50 -n $nn_type --lr $lr > outputs/REINFORCE_50_${nn_type}_${lr}.log 2>&1 & 
        count=$((count+1))
    done
done

# after best config is found, 
# BEST_NN=
# BEST_LR=
# for cache_size in 5 10 50 100; do
#     python train.py -c $cache_size -n $BEST_NN --lr $BEST_LR > outputs/REINFORCE_${cache_size}_${nn_type}_${lr}.log
# done