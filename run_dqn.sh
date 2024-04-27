set -x

NUM_DEVICES=`nvidia-smi  | grep 'NVIDIA A100 80GB PCIe' | wc -l`

echo "Found $NUM_DEVICES GPUs on this node."

count=0;

for lr in 0.01 0.05 0.1 0.5 1.00; do
    if [ "$count" == "$NUM_DEVICES" ]; then
        wait
        count=0
    fi
    CUDA_VISIBLE_DEVICES=$count python3 train.py -c 50 --lr $lr -a DQN > outputs/DQN_50_${lr}.log 2>&1 & 
    count=$((count+1))
done

# after best config is found, 
# BEST_LR=
# for cache_size in 5 10 50 100; do
#     python3 train.py -c $cache_size --lr $BEST_LR -a DQN > outputs/DQN_${cache_size}_${BEST_LR}.log
# done