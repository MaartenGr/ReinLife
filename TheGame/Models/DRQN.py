import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class DRQNAgent:
    def __init__(self, observation_dim=153, learning_rate=1e-3, capacity=10000, epsilon_init=0.9, gamma=0.99,
                 soft_update_freq=100, load_model=False, training=True):
        self.observation_dim = observation_dim
        self.action_dim = 8
        self.target_net = drqn_net(self.observation_dim, self.action_dim)
        self.eval_net = drqn_net(self.observation_dim, self.action_dim)
        self.eval_net.load_state_dict(self.target_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.buffer = recurrent_replay_buffer(capacity)
        self.loss_fn = nn.MSELoss()
        self.epsilon = epsilon_init
        self.count = 0
        self.hidden = None
        self.gamma = gamma
        self.soft_update_freq = soft_update_freq
        self.method = "DRQN"
        self.training = training
        self.epsilon_min = 0.05
        self.decay = 0.99
        self.n_epi = 0

        if load_model:
            self.eval_net.load_state_dict(torch.load(load_model))
            self.eval_net.eval()
            self.epsilon = 0

    def get_action(self, obs, n_epi):
        if self.training:
            if n_epi > self.n_epi:
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon * self.decay
                self.n_epi = n_epi

        action, hidden = self.eval_net.act(torch.FloatTensor(np.expand_dims(np.expand_dims(obs, 0), 0)),
                                      self.epsilon,
                                      self.hidden)
        self.hidden = hidden
        return action

    def memorize(self, obs, action, reward, next_obs, done):
        self.buffer.store(obs, action, reward, next_obs, done)

    def learn(self, age, dead, action, state, reward, state_prime, done, n_epi):
        self.memorize(state, action, reward / 200.0, state_prime, done)
        if age % 20 == 0 or dead:
            self.train()

        if n_epi % self.soft_update_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def train(self):
        observation, action, reward, next_observation, done = self.buffer.sample()

        observation = torch.FloatTensor(observation)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done)

        q_values, _ = self.eval_net.forward(observation)
        next_q_values, _ = self.target_net.forward(next_observation)
        argmax_actions = self.eval_net.forward(next_observation)[0].max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        # loss = loss_fn(q_value, expected_q_value.detach())
        loss = (expected_q_value.detach() - q_value).pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class drqn_net(nn.Module):
    def __init__(self, observation_dim, action_dim, time_step=1, layer_num=1, hidden_num=64):
        super(drqn_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.time_step = time_step
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.lstm = nn.LSTM(self.observation_dim, self.hidden_num, self.layer_num, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_num, 128)
        # nn.LSTM(input_size, hidden_num, layer_num, batch_first=True)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, observation, hidden=None):
        if not hidden:
            h0 = torch.zeros([self.layer_num, observation.size(0), self.hidden_num])
            c0 = torch.zeros([self.layer_num, observation.size(0), self.hidden_num])
            hidden = (h0, c0)
            # hidden [layer_num, batch_size, hidden_num]
        # lstm-x-input [batch_size, time_step, input_size]
        # lstm-x-output [batch_size, time_step, input_size]
        x, new_hidden = self.lstm(observation, hidden)
        x = self.fc1(x[:, -1, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x, new_hidden

    def act(self, observation, epsilon, hidden):
        q_values, new_hidden = self.forward(observation, hidden)
        if random.random() > epsilon:
            action = q_values.max(1)[1].detach()[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action, new_hidden


class recurrent_replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.memory.append([])

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(np.expand_dims(observation, 0), 0)
        next_observation = np.expand_dims(np.expand_dims(next_observation, 0), 0)

        self.memory[-1].append([observation, action, reward, next_observation, done])

        if done:
            self.memory.append([])

    def sample(self):
        idx = random.choice(list(range(len(self.memory) - 1)))
        observation, action, reward, next_observation, done = zip(* self.memory[idx])
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)
