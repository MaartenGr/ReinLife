# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from .utils import BasicBrain


class A2CAgent(BasicBrain):
    """ Actor 2 Critic

    Parameters:
    -----------
    input_dim : int
        The input dimension

    output_dim : int
        The output dimension

    train_freq : int, default 20
        The frequency at which to train the agent

    actor_learning_rate : float, default 0.001
        Learning rate

    critic_learning_rate : float, default 0.005
        Learning rate

    gamma : float, default 0.98
        Discount factor. How far out should rewards in the future influence the policy?

    load_model : str, default False
        Path to an existing model

    training : bool, default True,
        Whether to continue training or not
    """
    def __init__(self, input_dim, output_dim, train_freq=20, gamma=0.99, actor_learning_rate=0.001,
                 critic_learning_rate=0.005,
                 load_model=False):
        super().__init__(input_dim, output_dim, "A2C")
        self.load_model = load_model
        self.state_size = input_dim
        self.action_size = output_dim
        self.value_size = 1
        self.train_freq = train_freq

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = gamma
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights(load_model)
            # self.actor.load_weights("./save_model/cartpole_actor.h5")
            # self.critic.load_weights("./save_model/cartpole_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(128, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, state, n_epi):
        state = np.reshape(state, [1, self.state_size])
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def learn(self, age, dead, state, action, reward, state_prime, done):

        if age % self.train_freq == 0 or dead:
            self.train_model(state, action, reward, state_prime, done)