import random
import copy
from typing import List
import numpy as np

# Custom packages
from .entities import Agent, Empty, Food, Poison, SuperFood
from .grid import Grid
from .utils import Actions, EntityTypes
from TheGame.Helpers.Tracker import Tracker
from TheGame.Helpers.Saver import Saver
from TheGame.Helpers.Render import Visualize
from TheGame.Models.utils import BasicBrain


class Environment:
    """ The main environment in which agents act


    Parameters:
    -----------
    width : int, default 30
        The width of the environment

    height : int, default 30
        The height of the environment

    brains : List[BasicBrain], default None
        A list of brains used in the environment

    grid_size : int, default 16
        The size, in pixels, of each grid. Shape = (grid_size * grid_size)

    max_agents : int, default 50
        The maximum number of agents that can be alive in the environment at the same time

    print_interval : int, default 500
        The number of episodes at which interval to track results and update graphs

    static_families : bool, default True
        If True, a fixed number of families sharing the same gene can appear in the environment.
        This number is then defined by the number of brains that were selected. Each family
        would then share a brain, but brains differ between families.

    interactive_results : bool, default False
        Whether to show interactive matplotlib results of the simulation.

        NOTE: Due to the generation of matplotlib plots, this can significantly slow
        down the simulation. Set the print_interval high enough (i.e., 500 oir higher)
        to prevent this from happening.

    google_colab : bool, default False
        Set to true if you are running the simulation in google colab

    training : bool, default True
        Set to False if you want to test resulting brains. Keep True if you want to train new brains.

    save : bool, default False
        Whether you want to save the resulting brains

    pastel_colors : bool, default False
        Whether to use pastel colors for rendering the agents in the pygame simulation
    """

    def __init__(self,
                 width: int = 30,
                 height: int = 30,
                 brains: List[BasicBrain] = None,
                 grid_size: int = 16,
                 max_agents: int = 50,
                 print_interval: int = 500,
                 static_families: bool = True,
                 interactive_results: bool = False,
                 google_colab: bool = False,
                 training: bool = True,
                 save: bool = False,
                 pastel_colors: bool = False):

        # Coordinate information
        self.width = width
        self.height = height
        self.grid = None

        # Classes
        self.actions = Actions
        self.entities = EntityTypes

        # Trackers
        self.agents = []
        self.best_agents = []
        self.brains = brains

        # Statistics
        self.max_agents = max_agents
        self.max_genes = len(brains)

        # Options
        self.static_families = static_families
        self.google_colab = google_colab
        self.save = save
        self.training = training

        # Not used, but helps in understanding the environment
        self.action_space = 8
        self.observation_space = 153

        # Render
        self.viz = Visualize(self.width, self.height, grid_size, pastel_colors, self.static_families)

        # Results tracker
        self.tracker = Tracker(print_interval, interactive_results,
                               google_colab, len(self.brains), self.static_families,
                               brains)

    def reset(self):
        """ Reset the environment to the beginning """
        self.grid = Grid(self.width, self.height)
        self.agents = [self._add_agent(random_loc=True, brain=self.brains[i], gene=i) for i in range(self.max_genes)]

        self.best_agents = None
        if not self.static_families:
            self.best_agents = [copy.deepcopy(self.grid.get_entities(self.entities.agent)[0]) for _ in range(10)]

        # Add Good Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(Food, p=1)

        # Add Bad Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.05:
                self.grid.set_random(Poison, p=1)

        # Add super food
        self.grid.set_random(SuperFood, p=1)

        obs, _ = self._get_obs()

        # Update states of agents
        for agent in self.agents:
            agent.state = agent.state_prime

        return obs

    def step(self):
        """ move a single step """
        self._act()
        self._get_rewards()

        # Add food
        if len(np.where(self.grid.get_numpy() == self.entities.food)[0]) <= ((self.width * self.height) / 10):
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(Food, p=1)

        # Add poison
        if len(np.where(self.grid.get_numpy() == self.entities.poison)[0]) <= ((self.width * self.height) / 20):
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(Poison, p=1)

        if len(np.where(self.grid.get_numpy() == self.entities.super_food)[0]) == 0:
            self.grid.set_random(SuperFood, p=1)

        obs, _ = self._get_obs()

    def render(self, fps=10):
        """ Render the game using pygame """
        return self.viz.render(self.agents, self.grid, fps=fps)

    def update_env(self, n_epi=0):
        """ Update the environment """
        if self.training:
            self.tracker.update_results(self.agents, n_epi)

        if not self.static_families:
            self._update_best_agents()

        self.agents = self.grid.get_entities(self.entities.agent)
        self._reproduce()
        self._produce()
        self._remove_dead_agents()

        obs, _ = self._get_obs()

        for agent in self.agents:
            agent.state = agent.state_prime

        return obs

    def save_results(self):
        saver = Saver('Experiments', google_colab=self.google_colab)
        settings = {"print interval": self.tracker.print_interval,
                    "width": self.width,
                    "height": self.height,
                    "max agents": self.max_agents,
                    "families": self.static_families}

        fig = self.tracker.fig if not self.google_colab else None
        if self.static_families:
            saver.save([Agent(gene=gene, brain=brain) for gene, brain in enumerate(self.brains)], self.static_families,
                       self.tracker.results, settings, fig)
        else:
            saver.save(self.best_agents, self.static_families, self.tracker.results, settings, fig)

    def _get_rewards(self):
        """ Extract reward and whether the game has finished """

        # Update death status for all agents
        for agent in self.agents:
            if agent.health <= 0 or agent.age == agent.max_age:
                agent.dead = True

        for agent in self.agents:
            info = ""
            done = False

            nr_kin_alive = max(0, sum([1 for other_agent in self.agents if
                                       not other_agent.dead and
                                       agent.gene == other_agent.gene]) - 1)
            alive_agents = sum([1 for other_agent in self.agents if not other_agent.dead])

            if agent.dead:
                reward = (-1 * alive_agents) + nr_kin_alive
                done = True
                info = "Dead"
            elif alive_agents == 1:
                reward = 0
            else:
                reward = nr_kin_alive / alive_agents

            if agent.killed:
                reward += 0.2

            agent.update_rl_stats(reward, done, info)

    def _act(self):
        """ Make the agents act and reduce its health with each step """
        self.agents = self.grid.get_entities(self.entities.agent)
        for agent in self.agents:
            agent.health = min(200, agent.health - 10)
            agent.age = min(agent.max_age, agent.age + 1)
            agent.reset_killed()

        self._attack()  # To do: first attack, then move!
        self._prepare_movement()
        self._execute_movement()

    def _add_agent(self, coordinates=None, brain=None, gene=None, random_loc=False, p=1):
        """ Add agent, if random_loc then add at a random location with probability p """
        if random_loc:
            return self.grid.set_random(Agent, p=p, brain=brain, gene=gene)
        else:
            return self.grid.set(coordinates[0], coordinates[1], Agent, brain=brain, gene=gene)

    def _reproduce(self):
        """ Reproduce if old enough """
        for agent in self.agents:
            if not agent.dead and not agent.reproduced:
                if len(self.agents) <= self.max_agents and random.random() > 0.95 and agent.age > 5:

                    new_brain = agent.brain
                    if self.static_families:
                        new_brain = self.brains[agent.gene]

                    coordinates = self._get_empty_within_fov(agent)
                    if coordinates:
                        self._add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                        brain=new_brain, gene=agent.gene)
                    else:
                        self._add_agent(random_loc=True, brain=new_brain, gene=agent.gene)

    def _produce(self):
        """ Randomly produce new agent if too little agents are alive """
        if len(self.agents) <= self.max_agents + 1 and random.random() > 0.95:

            if self.static_families:
                gene = random.choice([x for x in range(len(self.brains))])
                brain = self.brains[gene]
                self._add_agent(random_loc=True, brain=brain, gene=gene)

            else:
                best_agent = random.choice(self.best_agents)
                brain = copy.deepcopy(best_agent.brain)
                self.max_genes += 1
                agent = self._add_agent(random_loc=True, brain=brain, gene=self.max_genes)
                if agent:
                    agent.scramble_brain()

    def _remove_dead_agents(self):
        """ Remove dead agent from grid """
        for agent in self.agents:
            if agent.dead:
                self.grid.grid[agent.i, agent.j] = Food((agent.i, agent.j))

    def _get_obs(self):
        """ Get the observation (fov) for each agent """
        self.agents = self.grid.get_entities(self.entities.agent)
        observations = []

        for agent in self.agents:

            # Main observations
            observation = self.grid.fov(agent.i, agent.j, 3)

            # Food
            vectorize = np.vectorize(lambda obj: self._get_food(obj))
            fov_food = list(vectorize(observation).flatten())

            # Family
            vectorize = np.vectorize(lambda obj: self._get_family(obj, agent))
            family_obs = list(vectorize(observation).flatten())

            # health
            vectorize = np.vectorize(lambda obj: obj.health / 200 if obj.entity_type == self.entities.agent else -1)
            health_obs = list(vectorize(observation).flatten())

            nr_genes = sum([1 for other_agent in self.agents if agent.gene == other_agent.gene])

            reproduced = 0
            if agent.reproduced:
                reproduced = 1

            fov = np.array(fov_food + family_obs + health_obs + [agent.health / 200] + [reproduced] + [nr_genes]
                           + [len(self.agents)] + [agent.killed] + [agent.ate_berry])

            if agent.age == 0:
                agent.state = fov
            agent.state_prime = fov
            observations.append(fov)

        return observations, None

    def _get_food(self, obj):
        """ Return 1 for food -1 for poison and 0 for everything else, 1 is returned if agent dies """
        if obj.entity_type == self.entities.food:
            return .5
        if obj.entity_type == self.entities.super_food:
            return 1.
        elif obj.entity_type == self.entities.poison:
            return -1.
        elif obj.entity_type == self.entities.agent:
            if obj.health < 0:
                return 1.
            else:
                return 0.
        else:
            return 0.

    def _get_family(self, obj, agent):
        """ Return 1 for family, -1 for non-family and 0 otherwise """
        if obj.entity_type == self.entities.agent:
            if obj.dead:
                return 0
            elif obj.gene == agent.gene:
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
                        agent.update_target_location(self.height - 1, agent.j)
                    else:
                        agent.update_target_location(agent.i - 1, agent.j)

                elif agent.action == self.actions.right:
                    if agent.j == self.width - 1:
                        agent.update_target_location(agent.i, 0)
                    else:
                        agent.update_target_location(agent.i, agent.j + 1)

                elif agent.action == self.actions.down:
                    if agent.i == self.height - 1:
                        agent.update_target_location(0, agent.j)
                    else:
                        agent.update_target_location(agent.i + 1, agent.j)

                elif agent.action == self.actions.left:
                    if agent.j == 0:
                        agent.update_target_location(agent.i, self.width - 1)
                    else:
                        agent.update_target_location(agent.i, agent.j - 1)
            else:
                agent.update_target_location(agent.i, agent.j)

    def _execute_movement(self):
        """ Move if no agents want to go to the same spot """

        impossible_coordinates = True
        while impossible_coordinates:

            impossible_coordinates = self._get_impossible_coordinates()
            for agent in self.agents:

                if agent.target_coordinates in impossible_coordinates:
                    agent.update_target_location(agent.i, agent.j)

        # Execute movement
        for agent in self.agents:
            if agent.action <= 3:
                self._eat(agent)
                self._update_agent_position(agent)

    def _attack(self):
        """ Attack and decrease health if target is hit """
        for agent in self.agents:
            target = Empty((-1, -1))

            if not agent.dead:

                # Get target
                if agent.action == self.actions.attack_up:
                    if agent.i == 0:
                        target = self.grid.get(self.height - 1, agent.j)
                    else:
                        target = self.grid.get(agent.i - 1, agent.j)

                elif agent.action == self.actions.attack_right:
                    if agent.j == (self.width - 1):
                        target = self.grid.get(agent.i, 0)
                    else:
                        target = self.grid.get(agent.i, agent.j + 1)

                elif agent.action == self.actions.attack_right:
                    if agent.j == (self.width - 1):
                        target = self.grid.get(agent.i, 0)
                    else:
                        target = self.grid.get(agent.i, agent.j + 1)

                elif agent.action == self.actions.attack_down:
                    if agent.i == (self.height - 1):
                        target = self.grid.get(0, agent.j)
                    else:
                        target = self.grid.get(agent.i + 1, agent.j)

                elif agent.action == self.actions.attack_left:
                    if agent.j == 0:
                        target = self.grid.get(agent.i, self.width - 1)
                    else:
                        target = self.grid.get(agent.i, agent.j - 1)

                # Execute attack
                if target.entity_type == self.entities.agent:
                    target.is_attacked()
                    agent.execute_attack()

                    if target.gene == agent.gene:
                        agent.inter_killed = 1
                    else:
                        agent.intra_killed = 1

    def _eat(self, agent):
        """ Eat food """

        if self.grid.grid[agent.i_target, agent.j_target].entity_type == self.entities.food:
            agent.health = min(200, agent.health + 40)
        elif self.grid.grid[agent.i_target, agent.j_target].entity_type == self.entities.poison:
            agent.health = min(200, agent.health - 40)
        elif self.grid.grid[agent.i_target, agent.j_target].entity_type == self.entities.super_food:
            agent.health = min(200, agent.health + 40)
            agent.max_age = int(agent.max_age * 1.2)
            agent.ate_berry = 1.

    def _update_agent_position(self, agent):
        """ Update position of an agent in the grid """
        self.grid.grid[agent.i, agent.j] = Empty((agent.i, agent.j))
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

        if self.agents:
            max_fitness_idx = int(np.argmax([agent.fitness for agent in self.agents]))
            max_fitness = self.agents[max_fitness_idx].fitness

            if self.agents[max_fitness_idx] not in self.best_agents and max_fitness > min_fitness:
                self.best_agents[min_fitness_idx] = self.agents[max_fitness_idx]
