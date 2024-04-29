# Deep Reinforcement Learning for Cache Replacement Policy

## Data Generation
```
mkdir data
python3 utils/gen_zipf.py
```

## Reproducing Results
To reproduce the results in the report, run one of the scripts ending in `run_<agent>.sh` which correspond to different agents. For more specific results, run `python3 train.py` with the right command line arguments.

## References

1. (https://dl.acm.org/doi/abs/10.1145/986537.986601)[Al-Zoubi, Hussein, Aleksandar Milenkovic, and Milena Milenkovic. "Performance evaluation of cache replacement policies for the SPEC CPU2000 benchmark suite." Proceedings of the 42nd annual Southeast regional conference. 2004.]
2. 