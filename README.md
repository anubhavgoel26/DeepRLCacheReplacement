# Deep Reinforcement Learning for Cache Replacement Policy

## Requirements
```
torch
numpy
pandas
matplotlib
```

## Data Generation
```
mkdir data
python3 utils/gen_zipf.py ./data/zipf_10k.csv 10000 50000 1.5 10
```

## Reproducing Results
To reproduce the results in the report, run one of the scripts ending in `run_<agent>.sh` which correspond to different agents. For more specific results, run `python3 train.py` with the right command line arguments. Every option has a default argument for an easier start.

## Running Code
The cache environment simulator has been adopted from [DRLCache](https://github.com/peihaowang/DRLCache/tree/master/cache). The file `outputs.zip` contains all the train logs from running the `run_<agent>.sh` commands. Unzip this folder to a convenient location and open the `parse_logs.ipynb` jupyter notebook. In this notebook, set `OUTPUT_DIR = "/path/to/where/you/unzipped"` and then set the `Model` variable on Line 9 of the notebook to one of `['Baselines', 'SarsaLambda', 'ActorCritic', 'ActorCriticQ', 'PPO_shallow', 'PPO_deep', 'DQN', 'REINFORCE (shallow)', 'REINFORCE (deep)', 'REINFORCE (attention)', 'FINAL']` to re-create plots from the paper.


## References

1. [Al-Zoubi, Hussein, Aleksandar Milenkovic, and Milena Milenkovic. "Performance evaluation of cache replacement policies for the SPEC CPU2000 benchmark suite." Proceedings of the 42nd annual Southeast regional conference. 2004.](https://dl.acm.org/doi/abs/10.1145/986537.986601)
2. [Belady, Laszlo A. "A study of replacement algorithms for a virtual-storage computer." IBM Systems journal 5.2 (1966): 78-101.](https://ieeexplore.ieee.org/abstract/document/5388441)
3. [Berger, Daniel S. "Towards lightweight and robust machine learning for cdn caching." Proceedings of the 17th ACM Workshop on Hot Topics in Networks. 2018.](https://dl.acm.org/doi/abs/10.1145/3286062.3286082)
4. [Berger, Daniel S., Nathan Beckmann, and Mor Harchol-Balter. "Practical bounds on optimal caching with variable object sizes." Proceedings of the ACM on Measurement and Analysis of Computing Systems 2.2 (2018): 1-38.](https://dl.acm.org/doi/abs/10.1145/3224427)
5. [BreslauL, CaoP, and ShenkerS PhilipsG. "Webcachingand Zipf likedistributions: Evidenceandimplications." Proceedingsofthe IEEEInternationalConferenceonComputerCommunications. NewYork, USA 126 (1999): 134.](https://ieeexplore.ieee.org/abstract/document/749260/)
6. [Brownlee, Jason. "A gentle introduction to the rectified linear unit (ReLU)." Machine learning mastery 6 (2019).](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/)
7. [Chen, Jiayi, et al. "Darwin: Flexible learning-based cdn caching." Proceedings of the ACM SIGCOMM 2023 Conference. 2023.](https://dl.acm.org/doi/abs/10.1145/3603269.3604863)
8. [What is a content delivery network (cdn)?](https://www.cloudflare.com/learning/cdn/what-is-a-cdn/)
9. [Gober, Nathan, et al. "The championship simulator: Architectural simulation for education and competition." arXiv preprint arXiv:2210.14324 (2022).](https://arxiv.org/abs/2210.14324)
10. [Hackenberg, Daniel, Daniel Molka, and Wolfgang E. Nagel. "Comparing cache architectures and coherency protocols on x86-64 multicore SMP systems." Proceedings of the 42Nd Annual IEEE/ACM International Symposium on microarchitecture. 2009.](https://dl.acm.org/doi/abs/10.1145/1669112.1669165)
11. [Henning, John L. "SPEC CPU2006 benchmark descriptions." ACM SIGARCH Computer Architecture News 34.4 (2006): 1-17.](https://dl.acm.org/doi/pdf/10.1145/1186736.1186737?casa_token=vVmSlSq_8lkAAAAA:idYw8qHhaZ6BxFX3bl_sQ0cHGOu1p8qULkh3iXVytlBnkEzzCK4wWvdD00Hwsvhw9e6nW0crPD2MiQ)
12. [Liu, Evan, et al. "An imitation learning approach for cache replacement." International Conference on Machine Learning. PMLR, 2020.](http://proceedings.mlr.press/v119/liu20f.html)
13. [Puzak, Thomas Roberts. Analysis of cache replacement-algorithms. University of Massachusetts Amherst, 1985.](https://search.proquest.com/openview/3821c43e2783d0c0a7043212ea833abf/1?pq-origsite=gscholar&cbl=18750&diss=y&casa_token=ZtRrhAbYH-AAAAAA:kp3uvkDHSfeokpxiuyGCslayI5Xer3lrCnhBulWM_bzbgZsdomPBzCfNxFbVJgHHCO6Azu-n)
14. [Rummery, Gavin A., and Mahesan Niranjan. On-line Q-learning using connectionist systems. Vol. 37. Cambridge, UK: University of Cambridge, Department of Engineering, 1994.](https://www.researchgate.net/profile/Mahesan-Niranjan/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems/links/5438d5db0cf204cab1d6db0f/On-Line-Q-Learning-Using-Connectionist-Systems.pdf?_sg%5B0%5D=HYd0h230b7WOR6m4hj5yx01K97aS61Z0DufUURMQr9ZqMqcEVZ0dNpG84h6uCfRl_M40FNkXgRX-GnpnxH31Ww.jBF3fgrlhaJYs3bDEaHQU22nRpKP0zKeF_oOsqh7WddL8pfxAomPSbeANzdmLP9YPB26HbLeSaEJqhFgzIxvWQ&_sg%5B1%5D=CZtZhHTEMgSwBZrpZU_7BACd8RH04JUKiITdXRQJ6MQ9SFS27jreZmcsuNcqYYWRoxcwBE-xBMbrfl1QobmEZ65bmkmpzonq5JoLRIIUKXne.jBF3fgrlhaJYs3bDEaHQU22nRpKP0zKeF_oOsqh7WddL8pfxAomPSbeANzdmLP9YPB26HbLeSaEJqhFgzIxvWQ&_iepl=)
15. [Sadeghi, Alireza, et al. "Reinforcement learning for adaptive caching with dynamic storage pricing." IEEE Journal on Selected Areas in Communications 37.10 (2019): 2267-2281.](https://ieeexplore.ieee.org/abstract/document/8790766/?casa_token=PK7zPZIzqtwAAAAA:frTNb29ehZqPncFXzMaaYAP1cnK-i1fO_TXIaqXlbfYzX-EBMJeKvByveylguu47eH2bixwwzA)
16. [Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).](https://arxiv.org/abs/1707.06347)
17. [Sethumurugan, Subhash, Jieming Yin, and John Sartori. "Designing a cost-effective cache replacement policy using machine learning." 2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA). IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9407137/?casa_token=igWUC0xRuFEAAAAA:gTMv1KO6Po70BPbd2beH62EEcaBOPV3CUHEulqAo-YRVnpy03VXsMVgwC-5yoYXOcN60SEcixA)
18. [Shi, Zhan, et al. "Applying deep learning to the cache replacement problem." Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture. 2019.](https://dl.acm.org/doi/abs/10.1145/3352460.3358319)
19. [Singh, Inderpreet, et al. "Cache coherence for GPU architectures." 2013 IEEE 19th International Symposium on High Performance Computer Architecture (HPCA). IEEE, 2013.](https://ieeexplore.ieee.org/abstract/document/6522351/?casa_token=2TVhOJupAlwAAAAA:mJkQ8d5QGFgEp4zqUhmWpEkuwPwg6syi_gHj0OfAb_umkh5vJbUyKBULOG6MTlY1bRafR34MnQ)
20. [Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018.](https://books.google.com/books?hl=en&lr=&id=uWV0DwAAQBAJ&oi=fnd&pg=PR7&dq=Reinforcement+learning:+An+introduction.&ots=mjoIq2_3k9&sig=2Hh98nKFYCDRvelDziXIC-FvNbw)
21. [Peihao Wang, Yuehao Wang, and Rui Wang. Drlcache: Deep reinforcement learning-based cache replacement policy.](https://github.com/peihaowang/DRLCache)
22. [Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8 (1992): 229-256.](https://link.springer.com/article/10.1007/BF00992696)
23. [Wu, Carole-Jean, et al. "SHiP: Signature-based hit predictor for high performance caching." Proceedings of the 44th Annual IEEE/ACM International Symposium on Microarchitecture. 2011.](https://dl.acm.org/doi/abs/10.1145/2155620.2155671)
24. [Young, Vinson, et al. "Ship++: Enhancing signature-based hit predictor for improved cache performance." The 2nd Cache Replacement Championship (CRC-2 Workshop in ISCA 2017). 2017.](https://www.semanticscholar.org/paper/SHiP-%2B-%2B-%3A-Enhancing-Signature-Based-Hit-Predictor-Young-Chou/d1fb26e6fc2c71d984bb1213af1d5b2a57f04b6f)
25. [Zhong, Chen, M. Cenk Gursoy, and Senem Velipasalar. "Deep reinforcement learning-based edge caching in wireless networks." IEEE Transactions on Cognitive Communications and Networking 6.1 (2020): 48-61.](https://ieeexplore.ieee.org/abstract/document/8964499/?casa_token=O4ErV8-RzzAAAAAA:Tj28qIMPq7oUzozANE2hlA0s6zspX7Z1Gny3coMHLxec7jyVf532L81AtJve6ECdt4QqTrhSDw)
26. [Zhou, Yang, et al. "An end-to-end automatic cache replacement policy using deep reinforcement learning." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 32. 2022.](https://ojs.aaai.org/index.php/ICAPS/article/view/19840)
27. [Zhou, Yang, et al. "An Efficient Deep Reinforcement Learning-based Automatic Cache Replacement Policy in Cloud Block Storage Systems." IEEE Transactions on Computers (2023).](https://ieeexplore.ieee.org/abstract/document/10288208/?casa_token=nf6AS0eXpXkAAAAA:WxecuOT1_iz3_ex2nhy2tqpf0S_yaf7TNxF-Z6z-gB84_6-xoVCP19v-vMNl995T4NrQP2M-ig)
28. [Zhu, Hao, et al. "Caching transient data for Internet of Things: A deep reinforcement learning approach." IEEE Internet of Things Journal 6.2 (2018): 2074-2083.](https://ieeexplore.ieee.org/abstract/document/8542696/?casa_token=14naOS4QrWAAAAAA:gYZFmy3HP_cOU_8OJJ46xy6PgvLMSLAIbt_-eG-R7SfvshaT2QuBPIc0wucpW4AZndQDQMzeiA)
