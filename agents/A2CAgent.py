import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

class A2CAgent(LearnerAgent):
    def __init__(self, n_actions, n_features, architecture='attention', actor_learning_rate=0.001, critic_learning_rate=0.01, reward_decay=0.9, memory_size=500, batch_size=32, verbose=0):
        self.n_actions = n_actions
        self.n_features = n_features
        self.gamma = reward_decay
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
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.learn_step_counter = 0
        self.memory = []

    def store_transition(self, s, a, r, s_):
        s_features = normalize_features(np.array(s['features']))
        s_next_features = normalize_features(np.array(s_['features']))
        transition = np.hstack((s_features.flatten(), [a, r], s_next_features.flatten()))
        self.memory.append(transition)  # Adjust to append to a list

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
        if len(self.memory) < self.batch_size:
            return
        sample_index = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch_memory = np.array(self.memory)[sample_index]
        batch_state = torch.tensor(batch_memory[:, :self.n_features], dtype=torch.float)
        batch_action = torch.tensor(batch_memory[:, self.n_features].astype(int), dtype=torch.long)
        batch_reward = torch.tensor(batch_memory[:, self.n_features + 1], dtype=torch.float)
        batch_next_state = torch.tensor(batch_memory[:, -self.n_features:], dtype=torch.float)

        v_s = self.critic(batch_state).squeeze()
        v_s_next = self.critic(batch_next_state).detach().squeeze()
        
        q_target = batch_reward + self.gamma * v_s_next
        advantage = q_target - v_s

        critic_loss = self.loss_func(v_s, q_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        log_probs = torch.log(self.actor(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.verbose:
            print(f'Step: {self.learn_step_counter}, Critic Loss: {critic_loss.item()}, Actor Loss: {actor_loss.item()}')

        self.learn_step_counter += 1
        self.memory = []

    def plot_cost(self):
        pass
