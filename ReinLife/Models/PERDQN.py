# https://github.com/rlcode/per

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Any
from .utils import BasicBrain


class PERDQNAgent(BasicBrain):
    """ Prioritized Experience Replay Deep Q Network

    Parameters:
    -----------
    input_dim : int
        The input dimension

    output_dim : int
        The output dimension

    explore_step : int, default 5_000
        The number of epochs until which to decrease epsilon

    train_freq : int, default 20
        The frequency at which to train the agent

    learning_rate : float, default 0.001
        Learning rate

    batch_size : int
        The number of training samples to work through before the model's internal parameters are updated.

    gamma : float, default 0.98
        Discount factor. How far out should rewards in the future influence the policy?

    capacity : int, default 10_000
        Capacity of replay buffer

    load_model : str, default False
        Path to an existing model

    training : bool, default True,
        Whether to continue training or not
    """
    def __init__(self, input_dim, output_dim, explore_step=5_000, train_freq=20, learning_rate=0.001, batch_size=64,
                 gamma=0.99, capacity=20000, load_model=False, training=True):
        super().__init__(input_dim, output_dim, "PERDQN")

        # get size of state and action
        self.state_size = input_dim
        self.action_size = output_dim

        # These are hyper parameters for the DQN
        self.discount_factor = gamma
        self.learning_rate = learning_rate
        self.memory_size = capacity
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = explore_step
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = batch_size
        self.train_start = 1000
        self.training = training
        self.train_freq = train_freq

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = DQN(input_dim, output_dim)
        self.model.apply(self.weights_init)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        self.training = training
        if not self.training:
            self.epsilon = 0

        if load_model:
            self.model.load_state_dict(torch.load(load_model))
            self.model.eval()

    def weights_init(self, m):
        """ Weight xavier initialize """
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)

    def update_target_model(self):
        """ After some time interval update the target model to be same with model """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        """ Get action from model using epsilon-greedy policy """
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    def append_sample(self, state, action, reward, next_state, done):
        """ Save sample (error,<s,a,r,s'>) to the replay memory """
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        target = self.model(Variable(torch.FloatTensor(state))).data
        old_val = target[0][action]
        target_val = self.target_model(Variable(torch.FloatTensor(next_state))).data
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * torch.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))

    def train_model(self):
        """ Pick samples from prioritized replay memory (with batch_size) """
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        mini_batch = np.array(mini_batch).transpose()

        states = np.vstack(mini_batch[0])
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.vstack(mini_batch[3])
        dones = mini_batch[4]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        errors = torch.abs(pred - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()

    def learn(self, age, dead, action, state, reward, state_prime, done):
        self.append_sample(state, action, reward, state_prime, done)

        if age % self.train_freq == 0 or dead:
            if self.memory.tree.n_entries >= self.train_start:
                self.train_model()

            self.update_target_model()


class SumTree:
    """ A binary tree data structure where the parent’s value is the sum of its children """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class Memory:
    """ Stored as ( s, a, r, s_ ) in SumTree"""
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            while True:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
                if not isinstance(data, int):
                    break
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

    def __call__(self, *args, **kwargs) -> Any:
        """ Necessary to remove linting problem in class above: https://github.com/pytorch/pytorch/issues/24326 """
        return super().__call__(*args, **kwargs)


