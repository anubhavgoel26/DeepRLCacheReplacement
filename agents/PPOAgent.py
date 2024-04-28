import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from agents.CacheAgent import LearnerAgent

from agents.ReflexAgent import RandomAgent, LRUAgent, LFUAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShallowActor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(ShallowActor, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class ShallowCritic(nn.Module):
    def __init__(self, n_features):
        super(ShallowCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepActor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DeepActor, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

class DeepCritic(nn.Module):
    def __init__(self, n_features):
        super(DeepCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def normalize_features(features):
    return (features - np.mean(features)) / (np.std(features) + 1e-8)

class PPOAgent(LearnerAgent):
    def __init__(self, n_actions, n_features, architecture='deep', actor_learning_rate=0.001, critic_learning_rate=0.01, gamma=0.99, clip_param=0.2, ppo_epochs=10, batch_size=32, verbose=0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.learn_step_counter = 0
        self.memory = []
        if architecture == 'shallow':
            self.actor = ShallowActor(n_features, n_actions).to(device)
            self.critic = ShallowCritic(n_features).to(device)
        elif architecture == 'deep':
            self.actor = DeepActor(n_features, n_actions).to(device)
            self.critic = DeepCritic(n_features).to(device)
        else:
            raise ValueError("Invalid architecture type")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_, old_log_prob):
        s_features = normalize_features(np.array(s['features']))
        s_next_features = normalize_features(np.array(s_['features']))
        self.memory.append((s_features, a, r, s_next_features, old_log_prob))

    def choose_action(self, observation, return_log_prob=True):
        features = normalize_features(np.array(observation['features']))
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
        probabilities = self.actor(features_tensor)
        dist = torch.distributions.Categorical(probabilities)
        if np.random.uniform() < 0.2:
            action = LRUAgent._choose_action(observation)
            log_prob = dist.log_prob(torch.tensor(action).to(device))
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = action.item()
        return (action, log_prob.item()) if return_log_prob else action.item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        sample_indices = random.sample(range(len(self.memory)), self.batch_size)
        sample = [self.memory[i] for i in sample_indices]
        self.memory = []

        states, actions, rewards, next_states, old_log_probs = zip(*sample)
        states = torch.tensor(np.vstack(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float).to(device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float).to(device)

        for _ in range(self.ppo_epochs):
            state_values = self.critic(states).squeeze()
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = (rewards + self.gamma * self.critic(next_states).detach().squeeze() - state_values).detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            if _ == self.ppo_epochs - 1:
                critic_loss = self.loss_func(state_values, rewards + self.gamma * self.critic(next_states).detach().squeeze())
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            if self.verbose:
                print(f'PPO Step: {self.learn_step_counter}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}')

        self.learn_step_counter += 1

    def plot_cost(self):
        pass
