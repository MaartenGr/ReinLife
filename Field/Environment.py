import numpy as np

# Gym
import gym
from gym import spaces

# Pygame
import pygame

# Custom
from Field.utils import check_pygame_exit
from Field import Point


class Environment(gym.Env):
    def __init__(self, width=30, height=30, mode='computer'):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=(2, 5, 5),
                                            dtype=np.float)
        self.current_episode = 0
        self.scores = []
        self.mode = mode
        self.grid_size = 16
        self.tile_location = np.random.randint(0, 4, (self.height, self.width))
        
        # Init variables
        self._reset_variables()

    def reset(self):
        """ Reset the environment to the beginning """
        self._reset_variables()
        self._init_objects()  # Init player and food pellets
        obs = self.update_fov()

        return obs

    def step(self, action):
        """ _move a single step """
        self.current_step += 1
        self._act(action)

        reward, done = self._get_reward()

        if done:
            self.current_episode += 1
            self.scores.append(self.total_reward)

        # Add pellet randomly
        if np.random.random() < 0.1:
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=1)
            self.food.append(food_pellet)

        obs = self.update_fov()

        return obs, reward, done, {}

    def render(self):
        self._init_pygame_screen()
        self._step_human()
        self._draw()

        return check_pygame_exit()

    def _act(self, action):
        self._move(action)

        self.health -= 10

    def _init_pygame_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
        clock = pygame.time.Clock()
        clock.tick(5)
        self.screen.fill((255, 255, 255))
        self._draw_tiles()

    def update_fov(self):
        # Initialize matrix
        fov = np.zeros([2, 5, 5])

        if self.player.x <= 2:
            for i in range(5):
                fov[0][i, 0] = 1
        if self.width - self.player.x <= 2:
            for i in range(5):
                fov[0][i, 4] = 1
        if self.player.y <= 2:
            fov[0][0] = 1
        if self.height - self.player.y <= 2:
            fov[0][-1] = 1

        for food in self.food:
            if abs(food.x - self.player.x) <= 2 and abs(food.y - self.player.y) <= 2:
                diff_x = food.x - self.player.x
                diff_y = self.player.y - food.y
                fov[1][2 - diff_y, 2 + diff_x] = food.value
        return fov

    def _init_objects(self):
        # Initialize food pellets
        for i in range(3):
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=1)
            self.food.append(food_pellet)

        # Initialize poison pellets
        for i in range(3):
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=-1)
            self.food.append(food_pellet)

        self.player = self._add_point(x=np.random.randint(self.width - 1),
                                      y=np.random.randint(self.height - 1),
                                      value=0)

    def _reset_variables(self):
        self.health = 200
        self.total_reward = 0
        self.coordinates = []
        self.food = []
        self.current_step = 0
        self.max_step = 30

    def _add_point(self, x, y, value):
        """ Create and add a point to all coordinates """
        point = Point(x, y, value)
        self.coordinates.append(point)
        return point

    def _get_closest_food_pellet(self):
        distances = [abs(point.x - self.player.x) + abs(point.y - self.player.y) for point in self.food]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Point(-1, -1, 0)
        return self.food[idx_closest_distance]

    def _move(self, action):
        """ Move the player to one space adjacent (up, right, down, left) """
        if action == 0 and self.player.y != (self.height - 1):  # up
            self.player.update(self.player.x, self.player.y + 1)
        elif action == 1 and self.player.x != (self.width - 1):  # right
            self.player.update(self.player.x + 1, self.player.y)
        elif action == 2 and self.player.y != 0:  # down
            self.player.update(self.player.x, self.player.y - 1)
        elif action == 3 and self.player.x != 0:  # left
            self.player.update(self.player.x - 1, self.player.y)

    def _get_reward(self):
        """ Extract reward and whether the game has finished """
        closest_food = self._get_closest_food_pellet()

        reward = 0
        done = False

        if np.array_equal(self.player.coordinates, closest_food.coordinates):
            if closest_food.value == 1:
                self.health += 50
                self.food.remove(closest_food)
                reward = 300
                self.total_reward += 300
            elif closest_food.value == -1:
                self.health -= 20
                self.food.remove(closest_food)
                reward -= 300
                self.total_reward -= 300
        elif self.health <= 0:
            reward -= 400
            self.total_reward -= 400
            done = True
        elif self.current_step == self.max_step:
            done = True

        return reward, done

    def _get_tiles(self):
        tile_1 = pygame.image.load(r'Sprites/tile_green_dark.png')
        tile_2 = pygame.image.load(r'Sprites/tile_green_light.png')
        tile_3 = pygame.image.load(r'Sprites/tile_green_dark_grass.png')
        tile_4 = pygame.image.load(r'Sprites/tile_green_light_grass.png')
        return [tile_3, tile_4, tile_3, tile_4]

    def _draw_tiles(self):
        tiles = self._get_tiles()
        for i in range(self.width):
            for j in range(self.height):
                self.screen.blit(tiles[self.tile_location[j, i]], (i*16, j*16))

    def _draw(self):
        # Draw player
        player = pygame.image.load(r'Sprites/player.png')
        self.screen.blit(player, (self.player.x * 16, self.player.y * 16))

        # Draw food
        apple = pygame.image.load(r'Sprites/apple.png')
        poison = pygame.image.load(r'Sprites/poison.png')
        for food in self.food:
            if food.value == 1:
                self.screen.blit(apple, (food.x * 16, food.y * 16))
            elif food.value == -1:
                self.screen.blit(poison, (food.x * 16, food.y * 16))

        pygame.display.update()

    def _step_human(self):
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
                    self.step(action)
                    print(self.update_fov(), self.health)



