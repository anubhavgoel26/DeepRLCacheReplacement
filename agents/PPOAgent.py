import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from agents.CacheAgent import LearnerAgent

class ShallowActor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(ShallowActor, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class ShallowCritic(nn.Module):
    def __init__(self, n_features):
        super(ShallowCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepActor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(DeepActor, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class DeepCritic(nn.Module):
    def __init__(self, n_features):
        super(DeepCritic, self).__init__()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def pad_features(x, num_heads):
    n_features = x.size(-1)
    padding_size = (num_heads - (n_features % num_heads)) % num_heads
    if padding_size > 0:
        padding = torch.zeros((x.size(0), padding_size), device=x.device)
        x = torch.cat([x, padding], dim=-1)
    return x

class AttentionActor(nn.Module):
    def __init__(self, n_features, n_actions, num_heads=4):
        super(AttentionActor, self).__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(n_features + (num_heads - n_features % num_heads) % num_heads, num_heads, batch_first=True)
        self.fc1 = nn.Linear(n_features + (num_heads - n_features % num_heads) % num_heads, n_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = pad_features(x, self.num_heads)
        x, _ = self.attention(x, x, x)
        x = self.fc1(x)
        return self.softmax(x)

class AttentionCritic(nn.Module):
    def __init__(self, n_features, num_heads=4):
        super(AttentionCritic, self).__init__()
        self.n_features = n_features
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(n_features + (num_heads - n_features % num_heads) % num_heads, num_heads, batch_first=True)
        self.fc1 = nn.Linear(n_features + (num_heads - n_features % num_heads) % num_heads, 1)

    def forward(self, x):
        x = pad_features(x, self.num_heads)
        x, _ = self.attention(x, x, x)
        x = self.fc1(x)
        return x

def normalize_features(features):
    return (features - np.mean(features)) / (np.std(features) + 1e-8)

class PPOAgent(LearnerAgent):
    def __init__(self, n_actions, n_features, architecture='attention', actor_learning_rate=0.001, critic_learning_rate=0.01, gamma=0.99, clip_param=0.2, ppo_epochs=10, batch_size=32, verbose=0):
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
            self.actor = ShallowActor(n_features, n_actions)
            self.critic = ShallowCritic(n_features)
        elif architecture == 'deep':
            self.actor = DeepActor(n_features, n_actions)
            self.critic = DeepCritic(n_features)
        elif architecture == 'attention':
            self.actor = AttentionActor(n_features, n_actions)
            self.critic = AttentionCritic(n_features)
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
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)
        probabilities = self.actor(features_tensor)
        dist = torch.distributions.Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return (action.item(), log_prob.item()) if return_log_prob else action.item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        sample_indices = random.sample(range(len(self.memory)), self.batch_size)
        sample = [self.memory[i] for i in sample_indices]
        self.memory = []

        states, actions, rewards, next_states, old_log_probs = zip(*sample)
        states = torch.tensor(np.vstack(states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float)


        for _ in range(self.ppo_epochs):
            state_values = self.critic(states).squeeze()
            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = (rewards + self.gamma * self.critic(next_states).detach().squeeze() - state_values).detach()  # Detach old value calculations
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

    def evaluate_actions(self, states, actions):
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        state_values = self.critic(states)
        return log_probs, state_values, entropy

    def plot_cost(self):
        pass
