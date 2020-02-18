import numpy as np

# Gym
import gym
from gym import spaces
import pygame

# Custom
from Field.utils import check_pygame_exit
from Field import Food, Agent


class MultiEnvironment(gym.Env):
    def __init__(self, width=30, height=30, mode='computer'):

        # Coordinate information
        self.width = width
        self.height = height
        self.grid_size = 16
        self.tile_location = np.random.randint(0, 2, (self.height, self.width))
        self.entity_coordinates = []

        # Trackers
        self.entities = {"Food": [],
                         "Agent": []}
        self.object_types = {"Food": Food,
                             "Agent": Agent}

        self.current_step = 0
        self.max_step = 30
        self.scores = []
        self.mode = mode

        # For Gym
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(7, 7),
                                            dtype=np.float)

    def reset(self):
        """ Reset the environment to the beginning """
        self._reset_variables()
        self._init_objects()
        obs = [self.get_fov(agent) for agent in self.entities["Agents"]]
        return obs

    def step(self, actions):
        """ move a single step """
        self.current_step += 1

        rewards = []
        dones = []
        for agent, action in zip(self.entities["Agents"], actions):
            self._act(action, agent)
            reward, done = self._get_reward(agent)
            rewards.append(reward)
            dones.append(done)

        # Add pellets randomly
        for i in range(3):
            if np.random.random() < 0.4:
                self._add_entity(random_location=True, value=1, entity_type="Food")

        obs = [self.get_fov(agent) for agent in self.entities["Agents"]]

        return obs, rewards, dones, {}

    def render(self):
        """ Render the game using pygame """
        self._init_pygame_screen()
        self._step_human()
        self._draw()

        return check_pygame_exit()

    def _act(self, action, agent):
        """ Make the agent act and reduce its health with each step """
        self._move(action, agent)
        agent.health -= 10

    def _init_pygame_screen(self):
        """ Initialize the pygame screen for rendering """
        pygame.init()
        self.screen = pygame.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
        clock = pygame.time.Clock()
        clock.tick(5)
        self.screen.fill((255, 255, 255))
        self._draw_tiles()

    def get_fov(self, agent):
        """ Update the agent's field of view """
        # Initialize matrix
        fov = np.zeros([3, 7, 7])

        if agent.x <= 3:
            for i in range(7):
                fov[0][i, 0] = 1
        if self.width - agent.x <= 3:
            for i in range(7):
                fov[0][i, 4] = 1
        if agent.y <= 3:
            fov[0][0] = 1
        if self.height - agent.y <= 3:
            fov[0][-1] = 1

        for food in self.entities["Food"]:
            if abs(food.x - agent.x) <= 3 and abs(food.y - agent.y) <= 3:
                diff_x = food.x - agent.x
                diff_y = agent.y - food.y
                fov[1][3 - diff_y, 3 + diff_x] = food.value

        for other_agent in self.entities["Agents"]:
            if agent != other_agent and agent.health > 0:
                if abs(other_agent.x - agent.x) <= 3 and abs(other_agent.y - agent.y) <= 3:
                    diff_x = other_agent.x - agent.x
                    diff_y = agent.y - other_agent.y
                    fov[2][3 - diff_y, 3 + diff_x] = 1
        return fov

    def _init_objects(self):
        """ Initialize the objects """
        # Add agent
        self.entities["Agents"] = [self._add_entity(random_location=True, value=0, entity_type="Agent") for _ in range(20)]

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
        self.entities = {"Food": [],
                         "Agent": []}
        self.entity_coordinates = []
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
                    self.entities[entity_type].append(entity)
                    self.entity_coordinates.append(entity.coordinates)
                    return entity
        else:
            entity = self.object_types[entity_type](x, y, value)
            if not self._coordinate_is_occupied(entity):
                self.entities[entity_type].append(entity)
                self.entity_coordinates.append(entity.coordinates)
            return entity

    def _remove_entity(self, entity):
        """ Remove an entity """
        print(entity.coordinates)
        print(self.entity_coordinates)
        self.entity_coordinates.remove(entity.coordinates)
        self.entities[entity.entity_type].remove(entity)

    # def _update_entity(self, old_entity, new_entity):
    #     self.entity_coordinates.remove
    #     self._remove_entity(entity)

    def _coordinate_is_occupied(self, entity):
        """ Check if coordinate is occupied """
        if any((entity.coordinates == coordinates).all() for coordinates in self.entity_coordinates):
            return True
        return False

    def _get_closest_food_pellet(self, agent):
        """  Get the closest food pellet to the agent """
        distances = [abs(food.x-agent.x) + abs(food.y-agent.y) for food in self.entities["Food"]]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Food(-1, -1, 0)
        return self.entities["Food"][idx_closest_distance]

    def _move(self, action, agent):
        """ Move the agent to one space adjacent (up, right, down, left) """
        if agent.health > 0:
            if action == 0 and agent.y != (self.height - 1):  # up
                agent.update(agent.x, agent.y + 1)
            elif action == 1 and agent.x != (self.width - 1):  # right
                agent.update(agent.x + 1, agent.y)
            elif action == 2 and agent.y != 0:  # down
                agent.update(agent.x, agent.y - 1)
            elif action == 3 and agent.x != 0:  # left
                agent.update(agent.x - 1, agent.y)

    def _get_reward(self, agent):
        """ Extract reward and whether the game has finished """
        closest_food = self._get_closest_food_pellet(agent)
        reward = 0
        done = False

        if not agent.dead:
            if np.array_equal(agent.coordinates, closest_food.coordinates):
                if closest_food.value == 1:
                    agent.health += 50
                    self.entities["Food"].remove(closest_food)
                    reward = 300
                elif closest_food.value == -1:
                    agent.health -= 20
                    self.entities["Food"].remove(closest_food)
                    reward -= 300
            elif agent.health <= 0:
                agent.dead = True
                reward -= 400
                done = True
            elif self.current_step == self.max_step:
                done = True
        else:
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
        agent_img = pygame.image.load(r'Sprites/agent.png')

        for agent in self.entities["Agents"]:
            if not agent.dead:
                self.screen.blit(agent_img, (agent.x * 16, agent.y * 16))

        # Draw food
        apple = pygame.image.load(r'Sprites/apple.png')
        poison = pygame.image.load(r'Sprites/poison.png')
        for food in self.entities["Food"]:
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
                    self.step([action])

                    fov = self.get_fov(self.entities["Agents"][0])

                    print(f"Walls : \n{fov[0]}")
                    print(f"Food  : \n{fov[1]}")
                    print(f"Players : \n{fov[2]}")




