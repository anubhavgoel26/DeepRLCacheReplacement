set -x

for cache_size in 5 10 50 100; do
    for agent in LRU LFU MRU Random; do
        python3 train.py -c $cache_size -a $agent > outputs/${agent}_${cache_size}.log 2>&1
    done
done