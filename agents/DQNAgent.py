import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.CacheAgent import LearnerAgent
from agents.ReflexAgent import RandomAgent, LRUAgent, LFUAgent

torch.manual_seed(1)

class Net(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(LearnerAgent):
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy_min=(0.1, 0.1),
        e_greedy_max=(0.1, 0.1),
        e_greedy_init=None,
        e_greedy_increment=None,
        e_greedy_decrement=None,
        reward_threshold=None,
        history_size=10,
        dynamic_e_greedy_iter=5,
        explore_mentor='LRU',
        replace_target_iter=300,
        memory_size=500,
        batch_size=32,
        output_graph=False,
        verbose=0
    ):
        super().__init__(n_actions)
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.history_size = history_size
        self.dynamic_e_greedy_iter = dynamic_e_greedy_iter
        self.reward_threshold = reward_threshold
        self.verbose = verbose

        self.epsilons_min = e_greedy_min
        self.epsilons_max = e_greedy_max
        self.epsilons_increment = e_greedy_increment
        self.epsilons_decrement = e_greedy_decrement
        
        self.epsilons = list(e_greedy_init) if e_greedy_init is not None else list(self.epsilons_min)

        self.explore_mentor = LRUAgent if explore_mentor.upper() == 'LRU' else LFUAgent
        
        self.eval_net = Net(n_features, n_actions)
        self.target_net = Net(n_features, n_actions)
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.reward_history = []

        if output_graph:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir="logs/")
        
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        s, s_ = s['features'], s_['features']
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
        if len(self.reward_history) == self.history_size:
            self.reward_history.pop(0)
        self.reward_history.append(r)

    def choose_action(self, observation):
        self.eval_net.eval()

        if np.random.uniform() < self.epsilons[0]:
            action = RandomAgent._choose_action(self.n_actions)
        elif self.epsilons[0] <= np.random.uniform() < sum(self.epsilons[:2]):
            if isinstance(observation, torch.Tensor):
                observation = observation.numpy()
            if isinstance(observation, dict):
                for key, value in observation.items():
                    if isinstance(value, torch.Tensor):
                        observation[key] = value.numpy()
            action = self.explore_mentor._choose_action(observation)
        else:
            if not isinstance(observation, torch.Tensor):
                observation = torch.FloatTensor(observation['features'])
            observation = torch.unsqueeze(observation, 0)

            with torch.no_grad():
                actions_value = self.eval_net(observation)
                action = actions_value.max(1)[1].item()

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            if self.verbose >= 1:
                print('Target DQN params replaced')

        if self.memory_counter < self.batch_size:
            return
        
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]
        
        batch_state = torch.FloatTensor(batch_memory[:, :self.n_features])
        batch_action = torch.LongTensor(batch_memory[:, self.n_features].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, self.n_features + 1])
        batch_next_state = torch.FloatTensor(batch_memory[:, -self.n_features:])
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0]
        
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.item())
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
