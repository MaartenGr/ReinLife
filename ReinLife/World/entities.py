from typing import Collection
from .utils import EntityTypes
from ReinLife.Models.utils import BasicBrain

entities = EntityTypes


class Entity:
    """ Represents a basic Entity that can target a new location and move to it

    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the entity.

    entity_type : int
        The type of entity which follows .utils.EntityTypes

    """
    def __init__(self, coordinates: Collection[int], entity_type: int):
        assert len(coordinates) == 2

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

    def update_target_location(self, i: int, j: int):
        """ Update coordinates of target location """
        self.i_target = i
        self.j_target = j
        self.target_coordinates = [i, j]


class Empty(Entity):
    """ Represents an Empty Space

    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the empty space.
    """
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, entities.empty)
        self.nutrition = 40

    def move(self):
        ...

    def update_target_location(self, i: int, j: int):
        ...


class BasicFood(Entity):
    """ Represents the basic food entity which has some nutritional value

    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the entity.

    entity_type : int
        The type of entity which follows .utils.EntityTypes

    """
    def __init__(self, coordinates: Collection[int], entity_type: int):
        super().__init__(coordinates, entity_type)
        self._nutrition = 0

    @property
    def nutrition(self) -> int:
        return self._nutrition

    @nutrition.setter
    def nutrition(self, val):
        self._nutrition = val


class Food(BasicFood):
    """ Represents the food entity which has a positive nutritional  value

    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the empty space.
    """
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, entities.food)
        self.nutrition = 40


class Poison(BasicFood):
    """ Represents the poison entity which has a positive nutritional  value


    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the empty space.
    """
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, entities.poison)
        self.nutrition = -40


class SuperFood(BasicFood):
    """ Represents the super food entity which has a positive nutritional value and has a max_age multiplier


    Parameters:
    -----------
    coordinates : Collection[int]
        The i and j coordinates (2d) of the empty space.
    """
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, entities.super_food)
        self.nutrition = 40
        self.age_multiplier = 1.2


class Agent(Entity):
    """ Represents an Agent that learns and thinks for itself

    Parameters:
    -----------
    coordinates : Collection[int], default (None, None)
        The i and j coordinates (2d) of the empty space.

    brain : BasicBrain, default None
        The brain of the agent.

    gene : int, default None
        The gene of the Agent which represents to which family it belongs
    """
    def __init__(self, coordinates: Collection[int] = (None, None), brain: BasicBrain = None, gene: int = None):
        super().__init__(coordinates, entities.agent)

        # Agent-based stats
        self.health = 200
        self.max_health = 200
        self.age = 0
        self.max_age = 50
        self.brain = brain
        self.reproduced = False
        self.gene = gene
        self.action = -1
        self.killed = 0
        self.inter_killed = 0
        self.intra_killed = 0
        self.ate_super_food = -1
        self.dead = False

        # Reinforcement Learning Stats
        self.state = None
        self.state_prime = None
        self.reward = None
        self.done = False
        self.info = None
        self.prob = None
        self.fitness = 0

    def reset_killed(self):
        """ Reset all killing stats """
        self.killed = 0
        self.inter_killed = 0
        self.intra_killed = 0

    def execute_attack(self):
        """ The agent executes an attack """
        self.health = min(200, self.health + 100)
        self.killed = 1

    def is_attacked(self):
        """ The agent is attacked """
        self.health = 0

    def update_rl_stats(self, reward: float, done: bool, info: str):
        """ Update the typical RL-based stats """
        self.fitness += reward
        self.reward = reward
        self.done = done
        self.info = info

    def learn(self, **kwargs):
        """ Make sure to """
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

    def mutate_brain(self):
        """ Applies gaussian noise to all layers in a model """
        if self.brain.method == "PERD3QN":
            self.brain.apply_gaussian_noise()

    def get_action(self, n_epi: int):
        """ Extracts the action """
        if self.brain.method == "PPO":
            self.action, self.prob = self.brain.get_action(self.state)
        elif self.brain.method == "PERDQN":
            self.action = self.brain.get_action(self.state)
        else:
            self.action = self.brain.get_action(self.state, n_epi)

    def save_brain(self, path: str):
        """ Save the best brain for further use """
        if self.brain.method == "A2C":
            self.brain.actor.save_weights(f"{path}.h5")

        else:
            import torch

            if self.brain.method == "DQN":
                torch.save(self.brain.agent.state_dict(), f"{path}.pt")

            elif self.brain.method == "PERDQN":
                torch.save(self.brain.model.state_dict(), f"{path}.pt")

            elif self.brain.method in ["PERD3QN", "DRQN", "D3QN"]:
                torch.save(self.brain.eval_net.state_dict(), f"{path}.pt")

            elif self.brain.method == "PPO":
                torch.save(self.brain.model.state_dict(), f"{path}.pt")

    def can_reproduce(self) -> bool:
        """ Returns whether the entity can reproduce """
        if not self.dead and not self.reproduced and self.age > 5:
            return True
        return False
