import argparse

from cache.Cache import Cache
from agents.CacheAgent import *
from agents.DQNAgent import DQNAgent
from agents.ActorCriticAgent import ActorCriticAgent
from agents.ActorCriticQAgent import ActorCriticQAgent
from agents.A2CAgent import A2CAgent
from agents.PPOAgent import PPOAgent
from agents.REINFORCEAgent import REINFORCEAgent
from agents.ReflexAgent import *
from cache.DataLoader import DataLoaderPintos

def main():
    parser = argparse.ArgumentParser(description='Train RL approaches for cache replacement problem')
    parser.add_argument('-c', '--cachesize', default=50, type=int, choices=[5, 25, 50, 100, 300])
    args = parser.parse_args()

    dataloader = DataLoaderPintos(["data/zipf.csv"])
    env = Cache(dataloader, args.cachesize, 
        feature_selection=('Base', 'UT', 'CT'), 
        reward_params = dict(name='our', alpha=0.5, psi=10, mu=1, beta=0.3), 
        allow_skip=False
    )

    agents = {}
    # agents['DQN'] = DQNAgent(env.n_actions, env.n_features,
    #     learning_rate=0.01,
    #     reward_decay=0.9,
        
    #     e_greedy_min=(0.0, 0.1),
    #     e_greedy_max=(0.2, 0.8),
    #     e_greedy_init=(0.1, 0.5),
    #     e_greedy_increment=(0.005, 0.01),
    #     e_greedy_decrement=(0.005, 0.001),

    #     history_size=50,
    #     dynamic_e_greedy_iter=25,
    #     reward_threshold=3,
    #     explore_mentor = 'LRU',

    #     replace_target_iter=100,
    #     memory_size=10000,
    #     batch_size=128,

    #     output_graph=False,
    #     verbose=0
    # )
    # agents['A2C'] = A2CAgent(env.n_actions, env.n_features,
    #     actor_learning_rate=0.0001,
    #     critic_learning_rate=0.001,
    #     reward_decay=0.9,
    #     batch_size=128
    # )
    agents['ActorCriticQ'] = ActorCriticQAgent(env.n_actions, env.n_features,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        reward_decay=0.99,
        batch_size=128
    )
    agents['PPO'] = PPOAgent(env.n_actions, env.n_features,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.0001,
    )
    agents['REINFORCE'] = REINFORCEAgent(env.n_actions, env.n_features)
    agents['Random'] = RandomAgent(env.n_actions)
    agents['LRU'] = LRUAgent(env.n_actions)
    agents['LFU'] = LFUAgent(env.n_actions)
    agents['MRU'] = MRUAgent(env.n_actions)

    for (name, agent) in agents.items():
        print("-------------------- %s --------------------" % name)

        step = 0
        episodes = 100 if isinstance(agent, LearnerAgent) else 1
        
        for episode in range(episodes):
            observation = env.reset()

            while True:
                if name=='PPO':
                    action, old_log_prob = agent.choose_action(observation)
                else:
                    action = agent.choose_action(observation)

                observation_, reward = env.step(action)

                if env.hasDone():
                    break

                if name=='PPO':
                    agent.store_transition(observation, action, reward, observation_, old_log_prob)
                else:
                    agent.store_transition(observation, action, reward, observation_)

                if isinstance(agent, LearnerAgent) and (step > 20) and (step % 5 == 0):
                    agent.learn()

                observation = observation_

                if step % 100 == 0:
                    mr = env.miss_rate()
                    print(f"### Agent={name}, CacheSize={args.cachesize} Episode={episode}, Step={step}: Accesses={env.total_count}, Misses={env.miss_count}, MissRate={mr}")

                step += 1
        mr = env.miss_rate()
        print(f"=== Agent={name}, CacheSize={args.cachesize} Episode={episode}: Accesses={env.total_count}, Misses={env.miss_count}, MissRate={mr}")

if __name__ == "__main__":
    main()