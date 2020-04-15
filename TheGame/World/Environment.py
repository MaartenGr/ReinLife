# Native
import os
import random
import copy

# 3rd party
import numpy as np
import gym
from gym import spaces

# Custom packages
from TheGame.World.Entities import Entity, Agent, Empty
from TheGame.World.Grid import Grid
from TheGame.World.utils import Actions, EntityTypes
from TheGame.World.Render import Visualize


class Environment(gym.Env):
    def __init__(self, width=30, height=30, evolution=False, fps=20, brains=None,
                 grid_size=16, max_agents=10, pastel=False, extended_fov=False):

        # Classes
        self.actions = Actions
        self.entities = EntityTypes

        # Coordinate information
        self.width = width
        self.height = height
        self.grid = None
        self.previous_grid = None
        self.grid_size = grid_size

        # Trackers
        self.agents = []
        self.best_agents = []
        self.nr_agents = len(brains)
        self.max_agents = max_agents
        self.max_gen = self.nr_agents
        self.evolution = evolution
        self.fps = fps
        self.pastel = pastel
        self.extended_fov = extended_fov

        if not brains:
            self.brains = [None for _ in range(self.nr_agents)]
        else:
            self.brains = brains

        self.current_step = 0
        if not evolution:
            self.max_step = 30
        else:
            self.max_step = 1e18

        # For Gym
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(151,),
                                            dtype=np.float)

        # Render
        self.viz = Visualize(self.width, self.height, self.grid_size, self.pastel)

    def reset(self, is_render=False):
        """ Reset the environment to the beginning """
        self.current_step = 0
        self.grid = Grid(self.width, self.height)

        if self.extended_fov:
            self.previous_grid = self.grid.copy()

        # Add agents
        self.agents = []
        for i in range(self.nr_agents):
            self._add_agent(random_loc=True, brain=self.brains[i], gen=i)
        self.best_agents = [self.grid.get_entities(self.entities.agent)[0] for _ in range(5)]

        # Add Good Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(Entity, p=1, value=self.entities.food)

        # Add Bad Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.05:
                self.grid.set_random(Entity, p=1, value=self.entities.poison)

        # Add super berrry
        self.grid.set_random(Entity, p=1, value=self.entities.super_berry)

        obs, _ = self._get_obs()
        for agent in self.agents:
            agent.state = agent.state_prime
        return obs

    def step(self):
        """ move a single step """
        if self.extended_fov:
            self.previous_grid = self.grid.copy()
        self.current_step += 1
        self._act()
        rewards, dones, infos = self._get_rewards()

        # Add food
        if len(np.where(self.grid.get_numpy() == self.entities.food)[0]) <= ((self.width * self.height) / 10):
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(Entity, p=1, value=self.entities.food)

        # Add poison
        if len(np.where(self.grid.get_numpy() == self.entities.poison)[0]) <= ((self.width * self.height) / 20):
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(Entity, p=1, value=self.entities.poison)

        if len(np.where(self.grid.get_numpy() == self.entities.super_berry)[0]) == 0:
            self.grid.set_random(Entity, p=1, value=self.entities.super_berry)

        obs, _ = self._get_obs()
        # self._update_best_agents()

    def render(self, fps=10):
        """ Render the game using pygame """
        return self.viz.render(self.agents, self.grid, fps=fps)

    def update_env(self):
        """ Update the environment """
        self.agents = self.grid.get_entities(self.entities.agent)
        self._reproduce()
        self._remove_dead_agents()
        self._produce()
        obs, _ = self._get_obs()

        for agent in self.agents:
            agent.state = agent.state_prime

        return obs

    def save_best_brain(self, n_epi):
        """ Save the best brain for further use """
        models = os.listdir(f'Brains/{self.best_agents[0].brain.method}')
        if len(models) != 0:
            index = max([int(x.split(".")[0][-1]) for x in models]) + 1
        else:
            index = 0

        best_agent = random.choice(self.best_agents)

        if best_agent.brain.method == "A2C":
            best_agent.brain.actor.save_weights(f"Brains/{best_agent.brain.method}/"
                                                f"model_{n_epi}_{index}.h5")

        elif best_agent.brain.method == "DQN":
            import torch
            torch.save(best_agent.brain.agent.state_dict(), f"Brains/{best_agent.brain.method}/"
            f"model_{n_epi}_{index}.pt")
        elif best_agent.brain.method == "PERDQN":
            import torch
            torch.save(best_agent.brain.model.state_dict(), f"Brains/{best_agent.brain.method}/"
            f"model_{n_epi}_{index}.pt")

        elif best_agent.brain.method == "DRQN":
            import torch
            torch.save(best_agent.brain.eval_net.state_dict(), f"Brains/{best_agent.brain.method}/"
            f"model_{n_epi}_{index}.pt")

        elif best_agent.brain.method == "PPO":
            import torch
            torch.save(best_agent.brain.agent.state_dict(), f"Brains/{best_agent.brain.method}/"
            f"model_{n_epi}_{index}.pt")

    def _get_rewards(self):
        """ Extract reward and whether the game has finished """
        rewards = [0 for _ in range(len(self.agents))]
        dones = [False for _ in self.agents]
        infos = ["" for _ in self.agents]

        # Update death status for all agents
        for agent in self.agents:
            if agent.health <= 0 or agent.age == agent.max_age:
                agent.dead = True

        for agent in self.agents:
            reward = 0
            info = ""
            done = False

            nr_kin_alive = max(0, sum([1 for other_agent in self.agents if
                                       not other_agent.dead and
                                       agent.gen == other_agent.gen]) - 1)
            alive_agents = sum([1 for other_agent in self.agents if not other_agent.dead])

            if agent.dead:
                reward = (-1 * alive_agents) + nr_kin_alive
                done = True
                info = "Dead"

            elif self.evolution:
                if alive_agents == 1:
                    reward = 0
                else:
                    reward = nr_kin_alive / alive_agents

            elif self.current_step >= self.max_step:
                done = True
                reward = 400

            if agent.killed:
                reward += 0.2

            agent.fitness += reward
            agent.reward = reward
            agent.done = done
            agent.info = info

            dones[0] = done
            infos[0] = info
            rewards[0] = reward

        return rewards, dones, infos

    def _act(self):
        """ Make the agents act and reduce its health with each step """
        self.agents = self.grid.get_entities(self.entities.agent)
        for agent in self.agents:
            agent.health = min(200, agent.health - 10)
            agent.age = min(agent.max_age, agent.age + 1)
            agent.killed = 0

        self._attack()  # To do: first attack, then move!
        self._prepare_movement()
        self._execute_movement()

    def _add_agent(self, coordinates=None, brain=None, gen=None, random_loc=False, p=1):
        """ Add agent, if random_loc then add at a random location with probability p """
        if random_loc:
            self.grid.set_random(Agent, p=p, value=self.entities.agent, brain=brain, gen=gen)
        else:
            self.grid.set(coordinates[0], coordinates[1], Agent, value=self.entities.agent, brain=brain, gen=gen)

    def _reproduce(self):
        """ Reproduce if old enough """
        for agent in self.agents:
            if not agent.dead and not agent.reproduced:
                if len(self.agents) <= self.max_agents and random.random() > 0.95 and agent.age > 5:
                    new_brain = self.brains[agent.gen]
                    coordinates = self._get_empty_within_fov(agent)
                    if coordinates:
                        self._add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                        brain=new_brain, gen=agent.gen)
                    else:
                        self._add_agent(random_loc=True, brain=new_brain, gen=agent.gen)

    def _produce(self):
        """ Randomly produce new agent if too little agents are alive """
        if len(self.agents) <= self.max_agents + 1 and random.random() > 0.95:
            gen = random.choice([x for x in range(self.nr_agents)])
            brain = self.brains[gen]
            self._add_agent(random_loc=True, brain=brain, gen=gen)

    def _remove_dead_agents(self):
        """ Remove dead agent from grid """
        for agent in self.agents:
            if agent.dead:
                self.grid.grid[agent.i, agent.j] = Entity((agent.i, agent.j), value=self.entities.food)

    def _get_obs(self):
        """ Get the observation (fov) for each agent """
        self.agents = self.grid.get_entities(self.entities.agent)
        observations = []

        for agent in self.agents:

            # Additional observations
            if self.extended_fov:
                previous_observation = self.previous_grid.fov(agent.i, agent.j, 3)

                # Previous Family
                vectorize = np.vectorize(lambda obj: self._get_family(obj, agent))
                previous_family_obs = list(vectorize(previous_observation).flatten())

            # Main observations
            observation = self.grid.fov(agent.i, agent.j, 3)

            # Food
            vectorize = np.vectorize(lambda obj: self._get_food(obj))
            fov_food = list(vectorize(observation).flatten())

            # Family
            vectorize = np.vectorize(lambda obj: self._get_family(obj, agent))
            family_obs = list(vectorize(observation).flatten())

            # health
            vectorize = np.vectorize(lambda obj: obj.health / 200 if obj.value == self.entities.agent else -1)
            health_obs = list(vectorize(observation).flatten())

            nr_genes = sum([1 for other_agent in self.agents if agent.gen == other_agent.gen])

            reproduced = 0
            if agent.reproduced:
                reproduced = 1

            if self.extended_fov:
                fov = np.array(fov_food + family_obs + previous_family_obs + health_obs + [agent.health / 200] +
                               [agent.i / self.width] + [agent.j / self.height] + [reproduced] + [nr_genes])
            else:
                fov = np.array(fov_food + family_obs + health_obs + [agent.health / 200] + [reproduced] + [nr_genes]
                               + [len(self.agents)] + [agent.killed] + [agent.ate_berry])

            if agent.age == 0:
                agent.state = fov
            agent.state_prime = fov
            observations.append(fov)

        return observations, None

    def _get_food(self, obj):
        """ Return 1 for food -1 for poison and 0 for everything else, 1 is returned if agent dies """
        if obj.value == self.entities.food:
            return .5
        if obj.value == self.entities.super_berry:
            return 1.
        elif obj.value == self.entities.poison:
            return -1.
        elif obj.value == self.entities.agent:
            if obj.health < 0:
                return 1.
            else:
                return 0.
        else:
            return 0.

    def _get_family(self, obj, agent):
        """ Return 1 for family, -1 for non-family and 0 otherwise """
        if obj.value == self.entities.agent:
            if obj.dead:
                return 0
            elif obj.gen == agent.gen:
                return 1
            else:
                return -1
        else:
            return 0

    def _get_empty_within_fov(self, agent):
        """ Gets global coordinates of all empty space within an agents fov """
        observation = self.grid.fov(agent.i, agent.j, 3)
        loc = np.where(observation == 0)
        coordinates = []

        for i_local, j_local in zip(loc[0], loc[1]):
            diff_x = i_local - 3
            diff_y = j_local - 3

            # Get coordinates if within normal range
            global_x = agent.i + diff_x
            global_y = agent.j + diff_y

            # Get coordinates if through wall (left vs right)
            if global_y < 0:
                global_y = self.width + global_y
            elif global_y >= self.width:
                global_y = global_y - self.width

            # Get coordinates if through wall (up vs down)
            if global_x < 0:
                global_x = self.height + global_x
            elif global_x >= self.height:
                global_x = global_x - self.height

            coordinates.append([global_x, global_y])

        return coordinates

    def _prepare_movement(self):
        """ Store the target coordinates agents want to go to """
        for agent in self.agents:

            if agent.action <= 3 and not agent.dead:

                if agent.action == self.actions.up:
                    if agent.i == 0:
                        agent.target_location(self.height - 1, agent.j)
                    else:
                        agent.target_location(agent.i - 1, agent.j)

                elif agent.action == self.actions.right:
                    if agent.j == self.width - 1:
                        agent.target_location(agent.i, 0)
                    else:
                        agent.target_location(agent.i, agent.j + 1)

                elif agent.action == self.actions.down:
                    if agent.i == self.height - 1:
                        agent.target_location(0, agent.j)
                    else:
                        agent.target_location(agent.i + 1, agent.j)

                elif agent.action == self.actions.left:
                    if agent.j == 0:
                        agent.target_location(agent.i, self.width - 1)
                    else:
                        agent.target_location(agent.i, agent.j - 1)
            else:
                agent.target_location(agent.i, agent.j)

    def _execute_movement(self):
        """ Move if no agents want to go to the same spot """

        loop_count = 0  # Not sure why, but it seems it gets stuck in an infinite loop
        impossible_coordinates = True
        while impossible_coordinates:

            impossible_coordinates = self._get_impossible_coordinates()
            for agent in self.agents:

                if agent.target_coordinates in impossible_coordinates:
                    agent.target_location(agent.i, agent.j)

            loop_count += 1
            if loop_count >= 1_000:
                print("Loop error")
                # for index, (agent, action) in enumerate(zip(self.agents, actions)):
                #     agent.target_location(agent.i, agent.j)

                print([agent.target_coordinates for agent in self.agents])
                print([agent.coordinates for agent in self.agents])
                print()
                return

        # Execute movement
        for agent in self.agents:
            if agent.action <= 3:
                self._eat(agent)
                self._update_agent_position(agent)

    def _attack(self):
        """ Attack and decrease health if target is hit """
        for agent in self.agents:
            target_i = None
            target_j = None
            target_agent = None

            if not agent.dead:

                if agent.action == self.actions.attack_up:
                    if agent.i == 0:
                        if self.grid.grid[self.height - 1, agent.j].value == self.entities.agent:
                            target_agent = self.grid.grid[self.height - 1, agent.j]
                    else:
                        if self.grid.grid[agent.i - 1, agent.j].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i - 1, agent.j]

                elif agent.action == self.actions.attack_right:
                    if agent.j == (self.width - 1):
                        if self.grid.grid[agent.i, 0].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, 0]
                    else:
                        if self.grid.grid[agent.i, agent.j + 1].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, agent.j + 1]

                elif agent.action == self.actions.attack_right:
                    if agent.j == (self.width - 1):
                        if self.grid.grid[agent.i, 0].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, 0]
                    else:
                        if self.grid.grid[agent.i, agent.j + 1].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, agent.j + 1]

                elif agent.action == self.actions.attack_down:
                    if agent.i == (self.height - 1):
                        if self.grid.grid[0, agent.j].value == self.entities.agent:
                            target_agent = self.grid.grid[0, agent.j]
                    else:
                        if self.grid.grid[agent.i + 1, agent.j].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i + 1, agent.j]

                elif agent.action == self.actions.attack_left:
                    if agent.j == 0:
                        if self.grid.grid[agent.i, self.width - 1].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, self.width - 1]
                    else:
                        if self.grid.grid[agent.i, agent.j - 1].value == self.entities.agent:
                            target_agent = self.grid.grid[agent.i, agent.j - 1]

                # Execute attack
                if target_agent:
                    target_agent.health = 0
                    agent.health = min(200, agent.health + 100)
                    agent.killed = 1

    def _eat(self, agent):
        """ Eat food """

        if self.grid.grid[agent.i_target, agent.j_target].value == self.entities.food:
            agent.health = min(200, agent.health + 40)
        elif self.grid.grid[agent.i_target, agent.j_target].value == self.entities.poison:
            agent.health = min(200, agent.health - 40)
        elif self.grid.grid[agent.i_target, agent.j_target].value == self.entities.super_berry:
            agent.health = 200
            agent.max_age = 80
            agent.ate_berry = 1.

    def _update_agent_position(self, agent):
        """ Update position of an agent """
        self.grid.grid[agent.i, agent.j] = Empty((agent.i, agent.j), value=self.entities.empty)
        self.grid.grid[agent.i_target, agent.j_target] = agent
        agent.move()

    def _get_impossible_coordinates(self):
        """ Returns coordinates of coordinates where multiple agents want to go """
        target_coordinates = [agent.target_coordinates for agent in self.agents]

        if target_coordinates:
            unq, count = np.unique(target_coordinates, axis=0, return_counts=True)
            impossible_coordinates = [list(coordinate) for coordinate in unq[count > 1]]
            return impossible_coordinates
        else:
            return []

    def _update_best_agents(self):
        """ Update best agents, replace weakest if a better is found """
        min_fitness_idx = int(np.argmin([agent.fitness for agent in self.best_agents]))
        min_fitness = self.best_agents[min_fitness_idx].fitness
        for agent in self.agents:
            if agent.fitness > min_fitness and agent not in self.best_agents:
                self.best_agents[min_fitness_idx] = agent
