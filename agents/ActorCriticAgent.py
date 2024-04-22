import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.CacheAgent import LearnerAgent

class Actor(nn.Module):
    def __init__(self, n_features, n_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, n_features):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

def normalize_features(features):
    return (features - np.mean(features)) / (np.std(features) + 1e-8)

class ActorCriticAgent(LearnerAgent):
    def __init__(self, n_actions, n_features, actor_learning_rate=0.001, critic_learning_rate=0.01, reward_decay=0.9, replace_target_iter=300, memory_size=500, batch_size=32, output_graph=False, verbose=0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.memory_counter = 0
        self.actor = Actor(n_features, n_actions)
        self.critic = Critic(n_features)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        s_features = normalize_features(np.array(s['features']))
        s_next_features = normalize_features(np.array(s_['features']))
        transition = np.hstack((s_features.flatten(), [a, r], s_next_features.flatten()))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        features = normalize_features(np.array(observation['features']))
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            probabilities = self.actor(features_tensor)
        self.actor.train()

        action_probs = probabilities.detach().numpy().squeeze()
        action = np.random.choice(range(self.n_actions), p=action_probs)
        return action


    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        sample_index = np.random.choice(min(self.memory_counter, self.memory_size), size=self.batch_size, replace=False)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float)
        batch_action = torch.tensor(batch_memory[:, self.n_features], dtype=torch.long)
        batch_reward = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float)
        batch_next_state = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float)
        q_values = self.critic(batch_state).squeeze()
        q_values_next = self.critic(batch_next_state).detach().squeeze()
        q_target = batch_reward + self.gamma * q_values_next
        critic_loss = self.loss_func(q_values, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        log_probs = torch.log(self.actor(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze())
        advantage = (q_target - q_values).detach()  # Detach to stop gradients
        actor_loss = -(log_probs * advantage).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        if self.verbose:
            print(f'Step: {self.learn_step_counter}, Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}')
        self.learn_step_counter += 1

    def plot_cost(self):
        pass
