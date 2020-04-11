# https://github.com/marload/deep-rl-tf2/blob/master/DuelingDoubleDQN/DuelingDoubleDQN_Discrete.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.optimizers import Adam

import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')

gamma = 0.95
learning_rate = 0.005
batch_size = 32
eps = 1.0
eps_decay = 0.995
eps_min = 0.01


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = eps

        self.model = self.create_model()

    def create_model(self):
        backbone = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu')
        ])
        state_input = Input((self.state_dim,))
        backbone_1 = Dense(32, activation='relu')(state_input)
        backbone_2 = Dense(16, activation='relu')(backbone_1)
        value_output = Dense(1)(backbone_2)
        advantage_output = Dense(self.action_dim)(backbone_2)
        output = Add()([value_output, advantage_output])
        model = tf.keras.Model(state_input, output)
        model.compile(loss='mse', optimizer=Adam(learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= eps_decay
        self.epsilon = max(self.epsilon, eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)


class DDQNAgent:
    def __init__(self, input_dim, output_dim):
        self.state_dim = input_dim
        self.action_dim = output_dim

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[
                range(batch_size), np.argmax(self.model.predict(next_states), axis=1)]
            targets[range(batch_size), actions] = rewards + (1 - done) * next_q_values * gamma
            self.model.train(states, targets)

    def get_action(self, state):
        return self.model.get_action(state)

    def memorize(self, state, action, reward, next_state, done):
        if done:
            done_mask = 1.0
        else:
            done_mask = 0.0
        self.buffer.put(state, action, reward * 0.01, next_state, done_mask)

    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward * 0.01, next_state, done)
                total_reward += reward
                state = next_state

            if self.buffer.size() >= batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))

