import functools 
import numpy as np
import time
import operator
import math

from agents.CacheAgent import LearnerAgent

def dot(a, b):
    return np.sum(a * b)

def natural_multiply(a, b):
    return int(a) * int(b)

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.n_dim = self.state_low.shape[0]
        assert self.state_low.shape[0] == self.state_high.shape[0]
        assert self.state_low.shape[0] == self.tile_width.shape[0]
        
        self.num_tiles = (np.ceil((self.state_high - self.state_low)/self.tile_width) + 1).astype(np.int32)
        self.feature_vector_size = self.num_tilings * self.num_actions * functools.reduce(natural_multiply, self.num_tiles, 1)

    def get_idx(self, tiling_index, dim, val):
        tile_start = self.state_low[dim] - ((tiling_index/self.num_tilings) * self.tile_width[dim])
        return min(
            int(np.floor((val - tile_start)/self.tile_width[dim])),
            self.num_tiles[dim] - 1
        )

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.feature_vector_size

    def __call__(self, s, a) -> np.array:
        feature_vec = np.zeros((self.num_tilings, self.num_actions, *self.num_tiles))
        for i in range(self.num_tilings):
            idx = [i, a]
            for j in range(self.n_dim):
                idx.append(self.get_idx(i, j, s[j]))
            feature_vec[tuple(idx)] = 1.00
        return feature_vec.flatten()

class SarsaLambdaAgent(LearnerAgent):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        lam=0.8, #lambda
        num_tilings=5,
        tile_width=1.0
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.lam = lam
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.X = StateActionFeatureVectorWithTile(
            np.asarray([0] * self.n_features),
            np.asarray([1] * self.n_features), 
            self.n_actions, self.num_tilings,
            np.asarray([self.tile_width] * self.n_features)
        )
        self.w = np.zeros(self.X.feature_vector_len())
        self.reset()
    
    def reset(self):
        self.Q_old = 0
        self.z = np.zeros(self.X.feature_vector_len())
        self.cached_action = None
        self.cached_observation_for_ac = None
    
    def epsilon_greedy_policy(self, s, epsilon=.0):
        nA = self.n_actions
        Q = [dot(self.w, self.X(s['features'], a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    def choose_action(self, observation):
        if self.cached_observation_for_ac is not None and self.cached_observation_for_ac == observation:
            returnval = self.cached_action
        else:
            returnval = self.epsilon_greedy_policy(observation)
        
        self.cached_observation_for_ac = None
        self.cached_action = None
        
        return returnval

    def store_transition(self, observation, action, reward, observation_):
        action_ = self.epsilon_greedy_policy(observation_)
        
        self.cached_action = action_
        self.cached_observation_for_ac = observation_

        x =  self.X(observation['features'], action)
        x_ = self.X(observation_['features'], action_)
        
        Q = dot(self.w.T, x)
        Q_dash = dot(self.w.T, x_)
        
        delta = reward + (self.gamma * Q_dash) - Q
        
        tmp = 1 - (self.alpha * self.gamma * self.lam * dot(self.z.T, x))
        
        self.z = (self.gamma * self.lam * self.z) + (tmp * x)

        self.w = self.w + ((self.alpha * (delta + Q - self.Q_old)) * self.z) - ((self.alpha * (Q - self.Q_old)) * x)
        self.Q_old =  Q_dash
    
    def learn(self):
        pass # we are learning with every transition so don't need to do anything

# env.reset()
# while True:
# 	agent.choose_action(observation)
# 	observation_, reward = env.step(action)
# 	agent.store_transition(observation, action, reward, observation_))
# 	if step % 5 == 0:
# 		agent.learn()