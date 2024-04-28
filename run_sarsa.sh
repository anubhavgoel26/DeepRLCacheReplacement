# set -x

for cache_size in 5; do
    for num_tilings in 2 5 7 10; do
        for tile_width in 1.0; do
            for lam in 0 0.2 0.4 0.6 0.8 1.0; do
                echo python3 train.py -f Base -c $cache_size --num_tilings $num_tilings --tile_width $tile_width --lam $lam -a SarsaLambda # > outputs/sarsa_tile_coding_${cache_size}_${num_tilings}_${tile_width}_${lam}.log 2>&1
            done
        done
    done
done