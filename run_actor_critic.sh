set -x

NUM_DEVICES=`nvidia-smi  | grep 'NVIDIA A100 80GB PCIe' | wc -l`

echo "Found $NUM_DEVICES GPUs on this node."

count=0;

for nn_type in shallow deep attention; do
    for lr in 0.00001 0.00005 0.0001 0.0005; do
        if [ "$count" == "$NUM_DEVICES" ]; then
            wait
            count=0
        fi
        CUDA_VISIBLE_DEVICES=$count python3 train.py -c 50 -n $nn_type --lr $lr -a ActorCritic > outputs/ActorCritic_50_${nn_type}_${lr}.log 2>&1 & 
        count=$((count+1))
    done
done

# after best config is found, 
# BEST_NN=
# BEST_LR=
# for cache_size in 5 10 50 100; do
#     python train.py -c $cache_size -n $BEST_NN --lr $BEST_LR -a ActorCritic > outputs/ActorCritic_${cache_size}_${BEST_NN}_${BEST_LR}.log
# done