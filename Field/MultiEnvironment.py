import numpy as np

# Gym
import gym
from gym import spaces

# Pygame
import pygame

# Custom
from Field.utils import check_pygame_exit
from Field import Food, Agent


class MultiEnvironment(gym.Env):
    def __init__(self, width=30, height=30, mode='computer'):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(7, 7),
                                            dtype=np.float)
        self.current_episode = 0
        self.scores = []
        self.mode = mode
        self.grid_size = 16
        self.tile_location = np.random.randint(0, 2, (self.height, self.width))
        self.object_types = {"Food": Food,
                             "Agent": Agent}
        
        # Init variables
        self._reset_variables()

    def reset(self):
        """ Reset the environment to the beginning """
        self._reset_variables()
        self._init_objects()  # Init agent and food pellets

        observations = []
        for agent_index in range(len(self.agents)):
            obs = self.update_fov(agent_index)
            observations.append(obs)

        return observations

    def step(self, actions):
        """ _move a single step """
        self.current_step += 1

        rewards = []
        dones = []
        for agent_index, action in enumerate(actions):
            self._act(action, agent_index)
            reward, done = self._get_reward(agent_index)
            rewards.append(reward)
            dones.append(done)

        # Add pellet randomly
        if np.random.random() < 0.1:
            self._add_entity(x=np.random.randint(self.width - 1),
                             y=np.random.randint(self.height - 1), value=1, entity_type="Food")

        observations = []
        for agent_index in range(len(self.agents)):
            obs = self.update_fov(agent_index)
            observations.append(obs)
        return observations, rewards, dones, {}

    def render(self):
        """ Render the game using pygame """
        self._init_pygame_screen()
        self._step_human()
        self._draw()

        return check_pygame_exit()

    def _act(self, action, agent_index):
        """ Make the agent act and reduce its health with each step """
        self._move(action, agent_index)
        self.agents[agent_index].health -= 10

    def _init_pygame_screen(self):
        """ Initialize the pygame screen for rendering """
        pygame.init()
        self.screen = pygame.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
        clock = pygame.time.Clock()
        clock.tick(5)
        self.screen.fill((255, 255, 255))
        self._draw_tiles()

    def update_fov(self, agent_index):
        """ Update the agent's field of view """
        # Initialize matrix
        fov = np.zeros([2, 7, 7])

        if self.agents[agent_index].x <= 3:
            for i in range(7):
                fov[0][i, 0] = 1
        if self.width - self.agents[agent_index].x <= 3:
            for i in range(7):
                fov[0][i, 4] = 1
        if self.agents[agent_index].y <= 3:
            fov[0][0] = 1
        if self.height - self.agents[agent_index].y <= 3:
            fov[0][-1] = 1

        for food in self.all_entities["Food"]:
            if abs(food.x - self.agents[agent_index].x) <= 3 and abs(food.y - self.agents[agent_index].y) <= 3:
                diff_x = food.x - self.agents[agent_index].x
                diff_y = self.agents[agent_index].y - food.y
                fov[1][3 - diff_y, 3 + diff_x] = food.value
        return fov

    def _init_objects(self):
        """ Initialize the objects """
        # Add agent
        self.agents = [self._add_entity(x=np.random.randint(self.width - 1),
                                        y=np.random.randint(self.height - 1),
                                        value=0, entity_type="Agent") for _ in range(20)]

        # Add Good Food
        for i in range(self.width*self.height):
            if np.random.random() < 0.05:
                self._add_entity(random_location=True, value=1, entity_type="Food")

        # Add Bad Food
        for i in range(self.width*self.height):
            if np.random.random() < 0.05:
                self._add_entity(random_location=True, value=-1, entity_type="Food")

    def _reset_variables(self):
        """ Reset variables back their starting values """
        self.total_reward = 0
        self.all_entities = {"Food": [],
                             "Agent": []}
        self.all_entity_coordinates = []
        self.current_step = 0
        self.max_step = 30

    def _add_entity(self, x=None, y=None, value=0, entity_type=None, random_location=False):
        """ Add an entity to a specified (x, y) coordinate. If random_location = True, then
            the entity will be added to a random unoccupied location. """
        if random_location:
            for i in range(20):
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)
                entity = self.object_types[entity_type](x, y, value)

                if not self._coordinate_is_occupied(entity):
                    self.all_entities[entity_type].append(entity)
                    self.all_entity_coordinates.append(entity.coordinates)
                    return entity
        else:
            entity = self.object_types[entity_type](x, y, value)
            if not self._coordinate_is_occupied(entity):
                self.all_entities[entity_type].append(entity)
                self.all_entity_coordinates.append(entity.coordinates)
            return entity

    def _remove_entity(self, entity):
        """ Remove an entity """
        self.all_entities[entity.entity_type].remove(entity)
        self.all_entity_coordinates.remove(entity.coordinates)

    def _coordinate_is_occupied(self, entity):
        """ Check if coordinate is occupied """
        if any((entity.coordinates == coordinates).all() for coordinates in self.all_entity_coordinates):
            return True
        return False

    def _get_closest_food_pellet(self, agent_index):
        """  Get the closest food pellet to the agent """
        distances = [abs(food.x-self.agents[agent_index].x) + abs(food.y-self.agents[agent_index].y) for food in self.all_entities["Food"]]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Food(-1, -1, 0)
        return self.all_entities["Food"][idx_closest_distance]

    def _move(self, action, agent_index):
        """ Move the agent to one space adjacent (up, right, down, left) """
        if action == 0 and self.agents[agent_index].y != (self.height - 1):  # up
            self.agents[agent_index].update(self.agents[agent_index].x, self.agents[agent_index].y + 1)
        elif action == 1 and self.agents[agent_index].x != (self.width - 1):  # right
            self.agents[agent_index].update(self.agents[agent_index].x + 1, self.agents[agent_index].y)
        elif action == 2 and self.agents[agent_index].y != 0:  # down
            self.agents[agent_index].update(self.agents[agent_index].x, self.agents[agent_index].y - 1)
        elif action == 3 and self.agents[agent_index].x != 0:  # left
            self.agents[agent_index].update(self.agents[agent_index].x - 1, self.agents[agent_index].y)

    def _get_reward(self, agent_index):
        """ Extract reward and whether the game has finished """
        closest_food = self._get_closest_food_pellet(agent_index)

        reward = 0
        done = False

        if np.array_equal(self.agents[agent_index].coordinates, closest_food.coordinates):
            if closest_food.value == 1:
                self.agents[agent_index].health += 50
                self.all_entities["Food"].remove(closest_food)
                reward = 300
                self.total_reward += 300
            elif closest_food.value == -1:
                self.agents[agent_index].health -= 20
                self.all_entities["Food"].remove(closest_food)
                reward -= 300
                self.total_reward -= 300
        elif self.agents[agent_index].health <= 0:
            reward -= 400
            self.total_reward -= 400
            done = True
        elif self.current_step == self.max_step:
            done = True

        return reward, done

    def _get_tiles(self):
        """ Load tile images """
        tile_1 = pygame.image.load(r'Sprites/tile_green_dark_grass.png')
        tile_2 = pygame.image.load(r'Sprites/tile_green_light_grass.png')
        return [tile_1, tile_2]

    def _draw_tiles(self):
        """ Draw tiles on screen """
        tiles = self._get_tiles()
        for i in range(self.width):
            for j in range(self.height):
                self.screen.blit(tiles[self.tile_location[j, i]], (i*16, j*16))

    def _draw(self):
        """ Draw all sprites and tiles """
        # Draw agent
        agent = pygame.image.load(r'Sprites/agent.png')
        for agent_index in range(len(self.agents)):
            self.screen.blit(agent, (self.agents[agent_index].x * 16, self.agents[agent_index].y * 16))

        # Draw food
        apple = pygame.image.load(r'Sprites/apple.png')
        poison = pygame.image.load(r'Sprites/poison.png')
        for food in self.all_entities["Food"]:
            if food.value == 1:
                self.screen.blit(apple, (food.x * 16, food.y * 16))
            elif food.value == -1:
                self.screen.blit(poison, (food.x * 16, food.y * 16))

        pygame.display.update()

    def _step_human(self):
        """ Execute an action manually """
        if self.mode == 'human':
            events = pygame.event.get()
            action = 4
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 2
                    if event.key == pygame.K_RIGHT:
                        action = 1
                    if event.key == pygame.K_DOWN:
                        action = 0
                    if event.key == pygame.K_LEFT:
                        action = 3
                    self.step([action, action])
                    for agent_index in range(len(self.agents)):
                        print(self.update_fov(agent_index))



