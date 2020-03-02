import numpy as np
from enum import IntEnum, Enum
import gym
from gym import spaces
import pygame
from Field.utils import check_pygame_exit


def pad_with(vector, pad_width, iaxis, kwargs):
    """ For padding the grid """
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = np.zeros([self.height, self.width])

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        self.grid[i, j] = v

    def set_random(self, v, p):
        zero_indices = np.argwhere(self.grid == 0)
        if zero_indices.size != 0:
            random_index = np.random.randint(0, zero_indices.shape[0])
            i, j = zero_indices[random_index]
            if np.random.random() < p:
                self.grid[i, j] = v
        return i, j

    def get(self, i, j):
        assert i >= 0 and j < self.width
        assert i >= 0 and j < self.height
        return self.grid[i, j]

    def reset(self, v, t):
        self.grid = np.where(self.grid == v, t, self.grid)

    def update(self, i, j, v1, k, l, v2):
        self.grid[i, j] = v1
        self.grid[k, l] = v2

    def fov(self, i, j, dist):
        fov = np.pad(self.grid, dist, pad_with, padder=-1)
        fov = fov[i:i + (2 * dist) + 1, j:j + (2 * dist) + 1]
        return fov

    def fov_new(self, i, j, dist):
        top = self.grid[:dist, :]
        bottom = self.grid[self.height - dist:, :]
        right = self.grid[:, self.width - dist:]
        left = self.grid[:, :dist]

        lower_left = self.grid[self.height - dist:, :dist]
        lower_right = self.grid[self.height - dist:, self.width - dist:]
        upper_left = self.grid[:dist, :dist]
        upper_right = self.grid[:dist, self.width - dist:]

        full_top = np.concatenate((lower_right, bottom, lower_left), axis=1)
        middle = np.concatenate((right, self.grid, left), axis=1)
        full_bottom = np.concatenate((upper_right, top, upper_left), axis=1)

        fov = np.concatenate((full_top, middle, full_bottom), axis=0)
        fov = fov[i:i + (2 * dist) + 1, j:j + (2 * dist) + 1]

        return fov


class Actions(IntEnum):
    # Turn left, turn right, move forward
    up = 0
    right = 1
    down = 2
    left = 3

    attack_up = 4
    attack_right = 5
    attack_down = 6
    attack_left = 7


class Entities(IntEnum):
    food = 1
    poison = 2
    agent = 3


class Agent:
    def __init__(self, coordinates, value):
        self.x, self.y = coordinates
        self.health = 200
        self.value = value
        self.coordinates = list(coordinates)
        self.target_coordinates = list(coordinates)
        self.dead = False
        self.done = False
        self.age = 0

        self.x_target = None
        self.y_target = None

    def move(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = [x, y]

    def target_location(self, x, y):
        self.x_target = x
        self.y_target = y
        self.target_coordinates = [x, y]


class GridWorld(gym.Env):
    def __init__(self, width=30, height=30, mode='computer', nr_agents=10):

        # Coordinate information
        self.width = width
        self.height = height
        self.grid = None

        # Render info
        self.grid_size = 16
        self.tile_location = np.random.randint(0, 2, (self.height, self.width))

        # Trackers
        self.agents = []
        self.agent_coordinates = []
        self.nr_agents = nr_agents
        self.current_step = 0
        self.max_step = 30
        self.mode = mode

        self.actions = Actions
        self.entities = Entities

        # # For Gym
        # self.action_space = spaces.Discrete(4)
        # self.observation_space = spaces.Box(low=-1,
        #                                     high=1,
        #                                     shape=(7, 7),
        #                                     dtype=np.float)

    def reset(self, is_render=False):
        """ Reset the environment to the beginning """
        self.current_step = 0
        self.grid = Grid(self.width, self.height)
        self.agent_coordinates = []

        # Add agents
        self.agents = []
        for _ in range(self.nr_agents):
            coordinates = self.grid.set_random(self.entities.agent, p=1)
            agent = Agent(coordinates, self.entities.agent)
            self.agents.append(agent)
            self.agent_coordinates.append(tuple(agent.coordinates))

        # Add Good Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(self.entities.food, p=1)

        # Add Bad Food
        for i in range(self.width * self.height):
            if np.random.random() < 0.1:
                self.grid.set_random(self.entities.poison, p=1)

        #         for i in range(40):
        #             self.grid.set_random(self.entities.food, p=1)
        #             self.grid.set_random(self.entities.poison, p=1)

        # obs = np.array([self.grid.fov(agent.x, agent.y, 2) for agent in self.agents])
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        observations = [self.grid.fov_new(agent.x, agent.y, 3) for agent in self.agents]

        one_hot_observations = []

        for index, observation in enumerate(observations):
            fov = np.zeros([7, 7])

            loc = np.where(observation == 1)
            for i, j in zip(loc[0], loc[1]):
                fov[i, j] = 1

            loc = np.where(observation == 2)
            for i, j in zip(loc[0], loc[1]):
                fov[i, j] = -1
            fov = list(fov.flatten())

            agents = np.zeros([7, 7])
            loc = np.where(observation == self.entities.agent)
            for i, j in zip(loc[0], loc[1]):
                agents[i, j] = 1
            agents = list(agents.flatten())

            total_fov = np.array(fov + agents + [self.agents[index].health] +
                                 [self.agents[index].x] + [self.agents[index].y])

            one_hot_observations.append(total_fov)

        return one_hot_observations

    def step(self, actions):
        """ move a single step """
        self.current_step += 1
        self._act(actions)
        rewards, dones, infos = self._get_rewards()

        # Add food
        if np.count_nonzero(self.grid.grid == self.entities.food) <= 500:
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(self.entities.food, p=1)

        # Add poison
        if np.count_nonzero(self.grid.grid == self.entities.poison) <= 500:
            for i in range(3):
                if np.random.random() < 0.2:
                    self.grid.set_random(self.entities.poison, p=1)

        # obs = np.array([self.grid.fov(agent.x, agent.y, 2) for agent in self.agents])
        obs = self._get_obs()

        return obs, rewards, dones, infos

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
                elif self.current_step >= self.max_step:
                    done = True
                    reward = 400
            else:
                done = True

            dones[index] = done
            infos[index] = info
            rewards[index] = reward

        return rewards, dones, infos

    def render(self):
        """ Render the game using pygame """
        self._init_pygame_screen()
        self._step_human()
        self._draw()

        return check_pygame_exit()

    def _check_death(self):
        for agent in self.agents:
            if not agent.dead and agent.health <= 0:
                self.grid.set(*agent.coordinates, self.entities.food)
                # agent.dead = True

    def _act(self, actions):
        """ Make the agents act and reduce its health with each step """
        for agent in self.agents:
            # print(self.agents)
            agent.health -= 10
            agent.age += 1

        self._prepare_movement(actions)
        self._execute_movement(actions)
        self._attack(actions)
        self._check_death()

        # print([agent.dead for agent in self.agents], [agent.health for agent in self.agents])

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
            agent.health += 50
        elif entity == self.entities.poison:
            agent.health -= 50

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

    def _init_pygame_screen(self):
        """ Initialize the pygame screen for rendering """
        pygame.init()
        self.screen = pygame.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
        clock = pygame.time.Clock()
        clock.tick(3)
        self.screen.fill((255, 255, 255))
        self._draw_tiles()

    def _draw_tiles(self):
        """ Draw tiles on screen """
        tiles = self._get_tiles()
        for i in range(self.width):
            for j in range(self.height):
                self.screen.blit(tiles[self.tile_location[j, i]], (i * 16, j * 16))

    def _draw(self):
        """ Draw all sprites and tiles """
        # Draw agent
        agent_img = pygame.image.load(r'Sprites/agent.png')

        for agent in self.agents:
            if not agent.dead:
                self.screen.blit(agent_img, (agent.x * 16, agent.y * 16))

        # Draw food
        apple = pygame.image.load(r'Sprites/apple.png')
        poison = pygame.image.load(r'Sprites/poison.png')

        # print(self.grid.grid)

        food = np.where(self.grid.grid == self.entities.food)
        for i, j in zip(food[0], food[1]):
            self.screen.blit(apple, (i * 16, j * 16))

        food = np.where(self.grid.grid == self.entities.poison)
        for i, j in zip(food[0], food[1]):
            self.screen.blit(poison, (i * 16, j * 16))

        pygame.display.update()

    def _get_tiles(self):
        """ Load tile images """
        tile_1 = pygame.image.load(r'Sprites/tile_green_dark_grass.png')
        tile_2 = pygame.image.load(r'Sprites/tile_green_light_grass.png')
        return [tile_1, tile_2]

    def _step_human(self):
        """ Execute an action manually """
        if self.mode == 'human':
            events = pygame.event.get()
            action = 4
            actions = [10 for _ in range(1)]
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        actions[0] = 3
                    if event.key == pygame.K_RIGHT:
                        actions[0] = 2
                    if event.key == pygame.K_DOWN:
                        actions[0] = 1
                    if event.key == pygame.K_LEFT:
                        actions[0] = 0
                    self.step(actions)
                    obs = self._get_obs()
                    print(f"{obs[0].T}")
                    # fov_food, fov_agents = self._get_fov_matrix(self.entities["Agents"][0])
                    #
                    # print(f"Health : {self.entities['Agents'][0].health}")
                    # print(f"Dead: {str(self.entities['Agents'][0].dead)}")
                    # print(f"Enemies: {fov_agents}")
