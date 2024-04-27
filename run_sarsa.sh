set -x

for cache_size in 5 10; do
    for num_tilings in 5 10; do
        for tile_width in 0.5 1.0; do
            for lam in 0 0.4 0.8 1.0; do
                python3 train.py -c $cache_size --num_tilings $num_tilings --tile_width $tile_width --lam $lam > outputs/sarsa_tile_coding_${cache_size}_${num_tilings}_${tile_width}_${lam}.log 2>&1
            done
        done
    done
done

# python3 train.py -c 5 --num_tilings 5 --tile_width 0.5 --lam 0.0 > outputs/sarsa_tile_coding0 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 0.5 --lam 0.4 > outputs/sarsa_tile_coding1 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 0.5 --lam 0.8 > outputs/sarsa_tile_coding2 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 0.5 --lam 1.0 > outputs/sarsa_tile_coding3 2>&1

# python3 train.py -c 5 --num_tilings 5 --tile_width 1.0 --lam 0.0 > outputs/sarsa_tile_coding4 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 1.0 --lam 0.4 > outputs/sarsa_tile_coding5 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 1.0 --lam 0.8 > outputs/sarsa_tile_coding6 2>&1
# python3 train.py -c 5 --num_tilings 5 --tile_width 1.0 --lam 1.0 > outputs/sarsa_tile_coding7 2>&1

# python3 train.py -c 5 --num_tilings 10 --tile_width 0.5 --lam 0.0 > outputs/sarsa_tile_coding8 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 0.5 --lam 0.4 > outputs/sarsa_tile_coding9 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 0.5 --lam 0.8 > outputs/sarsa_tile_coding10 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 0.5 --lam 1.0 > outputs/sarsa_tile_coding11 2>&1


# python3 train.py -c 5 --num_tilings 10 --tile_width 1.0 --lam 0.0 > outputs/sarsa_tile_coding12 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 1.0 --lam 0.4 > outputs/sarsa_tile_coding13 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 1.0 --lam 0.8 > outputs/sarsa_tile_coding14 2>&1
# python3 train.py -c 5 --num_tilings 10 --tile_width 1.0 --lam 1.0 > outputs/sarsa_tile_coding15 2>&1