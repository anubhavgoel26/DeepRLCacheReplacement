import argparse
import time 

from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ActorCriticAgent import ActorCriticAgent
from agents.ActorCriticQAgent import ActorCriticQAgent
from agents.PPOAgent import PPOAgent
from agents.REINFORCEAgent import REINFORCEAgent
from agents.SarsaLambdaAgent import SarsaLambdaAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos

def main():
    parser = argparse.ArgumentParser(description='Train RL approaches for cache replacement problem')
    parser.add_argument('-c', '--cachesize', default=50, type=int, choices=[5, 10, 50, 100])
    parser.add_argument('-d', '--data', default="zipf_10k.csv", type=str, choices=["zipf.csv", "zipf_10k.csv"])
    parser.add_argument('-n', '--network', default='shallow', choices=['deep', 'shallow', 'attention'], type=str)
    parser.add_argument('-a', '--agent', default='DQN', choices=['SarsaLambda', 'DQN', 'ActorCritic', 'ActorCriticQ', 'PPO', 'REINFORCE', 'LRU', 'LFU', 'MRU', 'Random'], type=str)
    
    # common args for all NN models
    parser.add_argument('--lr', default=1e-3, type=float)

    # arguments for SARSA LAMBDA
    parser.add_argument('--num_tilings', default=10, type=int)
    parser.add_argument('--lam', default=0.8, type=float)
    parser.add_argument('--tile_width', default=1.0, type=float)
    
    # args for PPO
    parser.add_argument('--ppo_epochs', default=10, type=int)

    args = parser.parse_args()
    

    print(args)

    dataloader = DataLoaderPintos([f"data/{args.data}"])
    env = Cache(dataloader, args.cachesize, 
        feature_selection=('Base', 'UT', 'CT'), 
        reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3), 
        allow_skip=False, normalize=True
    )
    
    if args.agent == 'SarsaLambda':
        agent = SarsaLambdaAgent(
            env.n_actions, env.n_features,
            num_tilings = args.num_tilings,
            tile_width = args.tile_width,
            lam = args.lam
        )
    elif args.agent == 'REINFORCE':
        agent = REINFORCEAgent(env.n_actions, env.n_features, nn_type = args.network, learning_rate = args.lr)
    elif args.agent == 'ActorCritic':
        agent = ActorCriticAgent(env.n_actions, env.n_features,
            actor_learning_rate=0.0001,
            critic_learning_rate=0.01,
            reward_decay=0.99,
            batch_size=16,
            architecture='deep'
        )
    elif args.agent == 'ActorCriticQ':
        agent = ActorCriticQAgent(env.n_actions, env.n_features,
            actor_learning_rate=args.lr,
            critic_learning_rate=args.lr,
            reward_decay=0.95,
            batch_size=32,
            architecture=args.network
        )
    elif args.agent == 'PPO':
        agent = PPOAgent(env.n_actions, env.n_features,
            actor_learning_rate=args.lr,
            critic_learning_rate=args.lr,
            ppo_epochs=args.ppo_epochs
        )
    elif args.agent == 'Random':
        agent = RandomAgent(env.n_actions)
    elif args.agent == 'LRU':
        agent = LRUAgent(env.n_actions)
    elif args.agent == 'LFU':
        agent = LFUAgent(env.n_actions)
    elif args.agent == 'MRU':
        agent = MRUAgent(env.n_actions)
    elif args.agent == 'DQN':
        agent = DQNAgent(env.n_actions, env.n_features,
            learning_rate=args.lr,
            reward_decay=0.9,        
            e_greedy_min=(0.0, 0.1),
            e_greedy_max=(0.2, 0.8),
            e_greedy_init=(0.1, 0.5),
            e_greedy_increment=(0.005, 0.01),
            e_greedy_decrement=(0.005, 0.001),

            history_size=50,
            dynamic_e_greedy_iter=25,
            reward_threshold=3,
            explore_mentor = 'LRU',

            replace_target_iter=100,
            memory_size=10000,
            batch_size=128,

            output_graph=False,
            verbose=0
        )
    
    step = 0
    episodes = 10 if isinstance(agent, LearnerAgent) else 1
    
    start_time_step = time.time()
    start_time_episode = time.time()

    for episode in range(episodes):
        observation = env.reset()

        while True:
            if args.agent=='PPO':
                action, old_log_prob = agent.choose_action(observation)
            else:
                action = agent.choose_action(observation)

            observation_, reward = env.step(action)

            if env.hasDone():
                break

            if args.agent=='PPO':
                agent.store_transition(observation, action, reward, observation_, old_log_prob)
            else:
                agent.store_transition(observation, action, reward, observation_)

            if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                agent.learn()

            observation = observation_

            if step % 10 == 0:
                mr = env.miss_rate()
                print(f"### Time={time.time() - start_time_step} Agent={args.agent}, CacheSize={args.cachesize} Episode={episode}, Step={step}: Accesses={env.total_count}, Misses={env.miss_count}, MissRate={mr}")
                start_time_step = time.time()

            step += 1
    mr = env.miss_rate()
    print(f"=== Time={time.time() - start_time_episode} Agent={args.agent}, CacheSize={args.cachesize} Episode={episode}: Accesses={env.total_count}, Misses={env.miss_count}, MissRate={mr}")
    start_time_episode = time.time()

if __name__ == "__main__":
    main()