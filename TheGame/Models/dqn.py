# # https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import collections
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class DQNAgent:
    def __init__(self, input_dim, learning_rate=0.0005, load_model=False):
        self.agent = Qnet(input_dim)
        self.target = Qnet(input_dim)
        self.target.load_state_dict(self.agent.state_dict())
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.method = 'DQN'

        if load_model:
            self.agent.load_state_dict(torch.load(load_model))
            self.agent.eval()

    def get_action(self, s, epsilon=0):
        return self.agent.sample_action(s, epsilon)

    def memorize(self, s, a, r, s_prime, done):
        if done:
            done_mask = 0.0
        else:
            done_mask = 1.0
        self.memory.put((s, a, r, s_prime, done_mask))

    def train(self):
        if self.memory.size() > 1000:
            train(self.agent, self.target, self.memory, self.optimizer)
        self.target.load_state_dict(self.agent.state_dict())


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, input_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float()
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()