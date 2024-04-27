import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.CacheAgent import LearnerAgent

from typing import Iterable
import numpy as np
import torch
import numpy as np

class PiApproximationWithNN(torch.nn.Module):
    def __init__(self, state_dims, num_actions, alpha, nn_type='shallow'):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        super(PiApproximationWithNN, self).__init__()

        assert nn_type in ['shallow', 'deep']
        self.state_dims = state_dims
        self.num_actions = num_actions

        self.alpha = alpha

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        
        if nn_type == 'shallow':
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dims, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions),
                torch.nn.Softmax()
            )
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dims, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, self.num_actions),
                torch.nn.Softmax()
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha, betas=(0.9, 0.999))

    def forward(self, states, return_prob=False):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        self.model.eval()
        output = self.model(states)
        
        if return_prob:
            return output
        else:
            probabilities = torch.distributions.Categorical(output)
            action = probabilities.sample()
            return action.item()

    def update(self, states, actions_taken, gamma_t, delta):
        """
        states: states
        actions_taken: actions_taken
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # theta = theta + (alpha * gamma^t * delta) * grad(ln(pi(A|S, theta)))
        self.model.train()
        self.optimizer.zero_grad()

        output = self.forward(torch.from_numpy(states).float().to(self.device), return_prob=True)
        logprob = torch.distributions.Categorical(output).log_prob(torch.tensor(actions_taken).to(self.device))
        loss = -1 * logprob * gamma_t * delta
        loss.backward()
        self.optimizer.step()

class VApproximationWithNN(torch.nn.Module):
    def __init__(self, state_dims, alpha, nn_type = 'shallow',):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        super(VApproximationWithNN, self).__init__()

        assert nn_type in ['shallow', 'deep']
        
        self.state_dims = state_dims
        self.alpha = alpha
        self.loss_fn = torch.nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if nn_type == 'shallow':
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dims, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(self.state_dims, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
            )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha, betas=(0.9, 0.999))

    def forward(self, states) -> float:
        self.model.eval()
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        op = self.model.forward(states)
        return op.data

    def update(self, states, G):
        # w = w + alpha^w * delta * grad(v(S, w))
        # Convert this into: θ=θ−α∇Loss
        self.model.train()
        self.optimizer.zero_grad()
        
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        predicted_val = self.model.forward(states.to(self.device))
        loss = self.loss_fn(predicted_val, torch.tensor([G], device=self.device))
        loss.backward()

        self.optimizer.step()
        self.model.eval()


class REINFORCEAgent(LearnerAgent):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.1,
        reward_decay=0.9,
        nn_type='shallow',
        cuda=False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.pi = PiApproximationWithNN(self.n_features, self.n_actions, self.lr, nn_type=nn_type)
        self.V = VApproximationWithNN(self.n_features, self.lr, nn_type=nn_type)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pi.to(self.device)
        self.V.to(self.device)
        
        self.state_idx = 0
        self.action_idx = 0
        self.reward_idx = 0

        self.states = dict()
        self.actions = dict()
        self.rewards = dict()

    def choose_action(self, observation):
        self.states[self.state_idx] = observation['features']
        ac = self.pi.forward(torch.from_numpy(self.states[self.state_idx]).to(self.pi.device).float(), return_prob=False)
        self.actions[self.action_idx] = ac

        self.state_idx += 1
        self.action_idx += 1

        return ac
    
    def store_transition(self, observation, action, reward, observation_):
        self.states[self.state_idx] = observation_['features']
        self.rewards[self.reward_idx] = reward
        self.reward_idx += 1
    
    def learn(self):
        for t in range(0, self.state_idx):
            G = 0.00
            for k in range(t+1, self.state_idx):
                G = G + self.gamma**(k-t-1) * self.rewards[k]

            delta = G - self.V.forward(torch.from_numpy(self.states[t]).float().to(self.device))

            self.V.update(self.states[t], G)
            self.pi.update(self.states[t], self.actions[t], self.gamma ** t, delta)

        self.state_idx = 0
        self.action_idx = 0
        self.reward_idx = 0

        self.states = dict()
        self.actions = dict()
        self.rewards = dict()    

# env.reset()
# while True:
# 	agent.choose_action(observation)
# 	observation_, reward = env.step(action)
# 	agent.store_transition(observation, action, reward, observation_))
# 	if step % 5 == 0:
# 		agent.learn()