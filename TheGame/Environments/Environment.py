import random
import pygame
import numpy as np
import copy
import gym
from gym import spaces

from TheGame.utils import check_pygame_exit
from TheGame.Entities.Agent import Agent
from TheGame.Environments.Grid import Grid
from TheGame.Environments.utils import Actions, Entities
from TheGame.Environments.Render import Visualize


class GridWorld(gym.Env):
    def __init__(self, width=30, height=30, mode='computer', nr_agents=10, evolution=False, fps=60, brains=None,
                 grid_size=16, prepare_render=True):

        # Coordinate information
        self.width = width
        self.height = height
        self.grid = None
        self.grid_size = grid_size

        # Trackers
        if not brains:
            self.brains = [None for _ in range(nr_agents)]
        else:
            self.brains = brains
        self.agents = []
        self.agent_coordinates = []
        self.nr_agents = nr_agents
        self.current_step = 0
        self.max_step = 30
        self.mode = mode
        self.evolution = evolution
        self.actions = Actions
        self.entities = Entities
        self.fps = fps
        self.best_agent = None
        self.max_gen = nr_agents

        # For Gym
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-10,
                                            high=200,
                                            shape=(151,),
                                            dtype=np.float)

        # Render
        self.viz = Visualize(self.width, self.height, self.grid_size)

    def reset(self, is_render=False):
        """ Reset the environment to the beginning """
        self.current_step = 0
        self.grid = Grid(self.width, self.height)
        self.agent_coordinates = []

        # Add agents
        self.agents = []
        for i in range(self.nr_agents):
            self._add_agent(random_loc=True, brain=self.brains[i], gen=i+1)
        self.best_agent = self.agents[0]

        # Add Good Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(self.entities.food, p=1)

        # Add Bad Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(self.entities.poison, p=1)

        obs, _ = self._get_obs()
        return obs

    def step(self, actions):
        """ move a single step """
        self.current_step += 1
        self._act(actions)
        rewards, dones, infos = self._get_rewards()

        # Add food
        if np.count_nonzero(self.grid.grid == self.entities.food) <= 20:
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(self.entities.food, p=1)

        # Add poison
        if np.count_nonzero(self.grid.grid == self.entities.poison) <= 20:
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(self.entities.poison, p=1)

        obs, _ = self._get_obs()

        # Update best agent
        for agent in self.agents:
            if agent.age > self.best_agent.age:
                self.best_agent = agent

        return obs, rewards, dones, infos

    def render(self, fps=10):
        """ Render the game using pygame """
        return self.viz.render(self.agents, self.grid.grid, fps=10)

    def update_env(self):
        self._reproduce()
        self._remove_dead_agents()
        self._produce()

        obs, _ = self._get_obs()

        return obs

    def _get_rewards(self):
        """ Extract reward and whether the game has finished """
        rewards = [0 for _ in range(len(self.agents))]
        dones = [False for _ in self.agents]
        infos = ["" for _ in self.agents]

        for index, agent in enumerate(self.agents):
            reward = 0
            info = ""
            done = False

            if not agent.dead:
                if agent.health <= 0:
                    agent.dead = True
                    reward = -400
                    done = True
                    info = "Dead"
                elif self.evolution:
                    # reward = 5
                    reward = 5 + (5 * sum([1 for other_agent in self.agents if agent.gen == other_agent.gen]))
                elif self.current_step >= self.max_step:
                    done = True
                    reward = 400

            else:
                done = True

            agent.fitness += reward

            dones[index] = done
            infos[index] = info
            rewards[index] = reward

        return rewards, dones, infos

    def _check_death(self):
        for agent in self.agents:
            if not agent.dead and agent.health <= 0:
                self.grid.set(*agent.coordinates, self.entities.food)

    def _act(self, actions):
        """ Make the agents act and reduce its health with each step """
        for agent in self.agents:
            agent.health -= 10
            agent.age += 1

        self._prepare_movement(actions)
        self._execute_movement(actions)
        self._attack(actions)
        self._check_death()

    def _add_agent(self, coordinates=None, brain=None, gen=None, random_loc=False, p=1):
        """ Add agent, if random_loc then add at a random location with probability p """
        if random_loc:
            coordinates = self.grid.set_random(self.entities.agent, p=p)
        agent = Agent(coordinates, self.entities.agent, brain=brain, gen=gen)
        self.agents.append(agent)
        self.agent_coordinates.append(tuple(agent.coordinates))

    def _reproduce(self):
        for agent in self.agents:
            if not agent.dead and not agent.reproduced:
                if len(self.agents) <= 10 and random.random() > 0.95 and agent.age > 5:
                    new_brain = copy.copy(agent.brain)
                    self._add_agent(random_loc=True, brain=new_brain, gen=agent.gen)

    def _produce(self):
        if len(self.agents) <= 11 and random.random() > 0.95:
            self.max_gen += 1
            new_brain = copy.copy(self.best_agent.brain)
            self._add_agent(random_loc=True, brain=new_brain, gen=self.max_gen)

    def _remove_dead_agents(self):
        for agent in self.agents:
            if agent.dead:
                self.agents.remove(agent)
                self.agent_coordinates.remove(tuple(agent.coordinates))

    def _get_obs(self):
        observations = []
        for agent in self.agents:
            observation = self.grid.fov(agent.x, agent.y, 3)

            # Extract food
            fov_food = np.zeros([7, 7])
            loc = np.where(observation == 1)
            for i, j in zip(loc[0], loc[1]):
                fov_food[i, j] = 1

            loc = np.where(observation == 2)
            for i, j in zip(loc[0], loc[1]):
                fov_food[i, j] = -1
            fov_food = list(fov_food.flatten())

            # See family
            # The i, j coordinates are taken from within the fov-field, which
            # doesn't match the global i, j coordinates. Therefore, they need to be adjusted using the difference
            # between the local i, j coordinates and the global coordinates of the agents (as that one is at the center)
            loc_agents = np.where(observation == self.entities.agent)
            family = np.zeros([7, 7])
            for i_local, j_local in zip(loc_agents[0], loc_agents[1]):
                i_global, j_global = agent.x + i_local - 3, agent.y + j_local - 3
                for other_agent in self.agents:
                    if other_agent.coordinates == [i_global, j_global]:
                        if other_agent.gen == agent.gen:
                            family[i_local, j_local] = 1
                        else:
                            family[i_local, j_local] = -1

            family_flat = list(family.flatten())

            reproduced = 0
            if agent.reproduced:
                reproduced = 1

            fov = np.array(fov_food + family_flat + [agent.health/200] +
                           [agent.x/self.width] + [agent.y/self.height] + [reproduced])

            observations.append(fov)

        return observations, None

    def _prepare_movement(self, actions):
        """ Store the target coordinates agents want to go to """
        for agent, action in zip(self.agents, actions):

            if action <= 3 and not agent.dead:

                if action == self.actions.up:
                    if agent.x == 0:
                        agent.target_location(self.height - 1, agent.y)
                    else:
                        agent.target_location(agent.x - 1, agent.y)

                elif action == self.actions.right:
                    if agent.y == self.width - 1:
                        agent.target_location(agent.x, 0)
                    else:
                        agent.target_location(agent.x, agent.y + 1)

                elif action == self.actions.down:
                    if agent.x == self.height - 1:
                        agent.target_location(0, agent.y)
                    else:
                        agent.target_location(agent.x + 1, agent.y)

                elif action == self.actions.left:
                    if agent.y == 0:
                        agent.target_location(agent.x, self.width - 1)
                    else:
                        agent.target_location(agent.x, agent.y - 1)
            else:
                agent.target_location(agent.x, agent.y)

    def _execute_movement(self, actions):
        """ Move if no agents want to go to the same spot """

        self.grid.reset(self.entities.agent, 0)

        impossible_coordinates = True
        while impossible_coordinates:

            impossible_coordinates = self._get_impossible_coordinates()

            for index, (agent, action) in enumerate(zip(self.agents, actions)):

                if agent.target_coordinates in impossible_coordinates:
                    agent.target_location(agent.x, agent.y)

        for index, (agent, action) in enumerate(zip(self.agents, actions)):
            if not agent.dead:

                # Move if only agent that wants to move there
                if action <= 3:
                    self._eat(agent, self.grid.get(*agent.target_coordinates))
                    self._update_agent_position(agent, *agent.target_coordinates, self.entities.agent, index)

                # Stay if other agents wants to move there
                else:
                    self._update_agent_position(agent, *agent.coordinates, self.entities.agent, index)

    def _attack(self, actions):
        for agent, action in zip(self.agents, actions):
            if not agent.dead:

                if action == self.actions.attack_up:
                    if (agent.x - 1, agent.y) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x - 1, agent.y))
                        self.agents[index].health -= 50  # 1
                    elif agent.x == 0 and (self.height - 1, agent.y) in self.agent_coordinates:
                        index = self.agent_coordinates.index((self.height - 1, agent.y))
                        self.agents[index].health -= 50  # 2

                elif action == self.actions.attack_right:
                    if (agent.x, agent.y + 1) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x, agent.y + 1))
                        self.agents[index].health -= 50  # 3
                    elif agent.y == (self.width - 1) and (agent.x, 0) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x, 0))
                        self.agents[index].health -= 50  # 4

                elif action == self.actions.attack_down:
                    if (agent.x + 1, agent.y) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x + 1, agent.y))
                        self.agents[index].health -= 50  # 5
                    elif agent.x == (self.height - 1) and (0, agent.y) in self.agent_coordinates:
                        index = self.agent_coordinates.index((0, agent.y))
                        self.agents[index].health -= 50  # 6

                elif action == self.actions.attack_left:
                    if (agent.x, agent.y - 1) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x, agent.y - 1))
                        self.agents[index].health -= 50  # 7
                    elif agent.y == 0 and (agent.x, self.width - 1) in self.agent_coordinates:
                        index = self.agent_coordinates.index((agent.x, self.width - 1))
                        self.agents[index].health -= 50  # 8

    def _eat(self, agent, entity):
        """ Eat food """
        if entity == self.entities.food:
            agent.health = min(200, agent.health + 40)
        elif entity == self.entities.poison:
            agent.health -= 40

    def _update_agent_position(self, agent, x, y, v, i):
        """ Update position of an agent """
        self.grid.set(x, y, v)
        agent.move(x, y)
        self.agent_coordinates[i] = (x, y)

    def _get_impossible_coordinates(self):
        """ Returns coordinates of coordinates where multiple agents want to go """
        target_coordinates = [agent.target_coordinates for agent in self.agents if not agent.dead]

        if target_coordinates:
            unq, count = np.unique(target_coordinates, axis=0, return_counts=True)
            impossible_coordinates = [list(coordinate) for coordinate in unq[count > 1]]
            return impossible_coordinates
        else:
            return []
