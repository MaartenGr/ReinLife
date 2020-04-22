import os
from TheGame.World.utils import EntityTypes

entities = EntityTypes


class Entity:
    """ An Entity """
    def __init__(self, coordinates, entity_type):

        # Current coordinates
        self.i, self.j = coordinates
        self.coordinates = list(coordinates)

        # Coordinates of target location
        self.i_target, self.j_target = None, None
        self.target_coordinates = list(coordinates)
        self.entity_type = entity_type

    def move(self):
        """ Move to target location """
        self.i = self.i_target
        self.j = self.j_target
        self.coordinates = [self.i, self.j]

    def update_target_location(self, i, j):
        """ Update coordinates of target location """
        self.i_target = i
        self.j_target = j
        self.target_coordinates = [i, j]


class Empty(Entity):
    """ An Agent with several trackers and movement options """
    def __init__(self, coordinates):
        super().__init__(coordinates, entities.empty)
        self.nutrition = 40


class Food(Entity):
    """ An Agent with several trackers and movement options """
    def __init__(self, coordinates):
        super().__init__(coordinates, entities.food)
        self.nutrition = 40


class Poison(Entity):
    """ An Agent with several trackers and movement options """
    def __init__(self, coordinates):
        super().__init__(coordinates, entities.poison)
        self.nutrition = -40


class SuperFood(Entity):
    """ An Agent with several trackers and movement options """
    def __init__(self, coordinates):
        super().__init__(coordinates, entities.super_food)
        self.nutrition = 40
        self.age_multiplier = 1.2


class Agent(Entity):
    """ An Agent with several trackers and movement options """
    def __init__(self, coordinates=(None, None), entity_type=None, brain=None, gen=None):
        super().__init__(coordinates, entity_type)

        # Agent-based stats
        self.health = 200
        self.age = 0
        self.max_age = 50
        self.brain = brain
        self.reproduced = False
        self.gen = gen
        self.fitness = 0
        self.action = -1
        self.killed = 0
        self.ate_berry = -1
        self.dead = False

        # Reinforcement Learning Stats
        self.state = None
        self.state_prime = None
        self.reward = None
        self.done = False
        self.info = None
        self.prob = None

    def execute_attack(self):
        """ The agent executes an attack """
        self.health = min(200, self.health + 100)
        self.killed = 1

    def is_attacked(self):
        """ The agent is attacked """
        self.health = 0

    def update_rl_stats(self, reward, done, info):
        self.fitness += reward
        self.reward = reward
        self.done = done
        self.info = info

    def learn(self, **kwargs):
        if self.age > 1:
            if self.brain.method == "PPO":
                self.brain.learn(age=self.age, dead=self.dead, action=self.action, state=self.state, reward=self.reward,
                                 state_prime=self.state_prime, done=self.done, prob=self.prob)
            elif self.brain.method == "DRQN":
                self.brain.learn(age=self.age, dead=self.dead, action=self.action, state=self.state, reward=self.reward,
                                 state_prime=self.state_prime, done=self.done, **kwargs)
            elif self.brain.method in ["DQN", "A2C", "PERDQN"]:
                self.brain.learn(age=self.age, dead=self.dead, action=self.action, state=self.state, reward=self.reward,
                                 state_prime=self.state_prime, done=self.done)
            else:
                self.brain.learn(age=self.age, dead=self.dead, action=self.action, state=self.state, reward=self.reward,
                                 state_prime=self.state_prime, done=self.done, **kwargs)

    def scramble_brain(self):
        if self.brain.method == "PERD3QN":
            self.brain.apply_gaussian_noise()

    def get_action(self, n_epi):
        if self.brain.method == "PPO":
            self.action, self.prob = self.brain.get_action(self.state, n_epi)
        else:
            self.action = self.brain.get_action(self.state, n_epi)

    def save_brain(self, path):
        """ Save the best brain for further use """
        if self.brain.method == "A2C":
            self.brain.actor.save_weights(f"{path}.h5")

        elif self.brain.method == "DQN":
            import torch
            torch.save(self.brain.agent.state_dict(), f"{path}.pt")

        elif self.brain.method == "PERDQN":
            import torch
            torch.save(self.brain.model.state_dict(), f"{path}.pt")

        elif self.brain.method in ["PERD3QN", "DRQN", "D3QN"]:
            import torch
            torch.save(self.brain.eval_net.state_dict(), f"{path}.pt")

        elif self.brain.method == "PPO":
            import torch
            torch.save(self.brain.agent.state_dict(), f"{path}.pt")



