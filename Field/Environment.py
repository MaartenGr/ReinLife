import numpy as np

# Gym
import gym
from gym import spaces

# Pygame
import pygame
from pygame import gfxdraw

# Custom
from Field import Point


class MazeEnv(gym.Env):
    def __init__(self, width=10, height=12):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-max([height, width]),
                                            high=max([height, width]),
                                            shape=(1, 2),
                                            dtype=np.int16)
        self.current_episode = 0
        self.scores = []

    def add_point(self, x, y, value):
        """ Create and add a point to all coordinates """
        point = Point(x, y, value)
        self.coordinates.append(point)
        return point

    def create_numpy_map(self):
        """ Convert all coordinates to a numpy representation """
        world = np.zeros((self.height, self.width))

        for point in self.coordinates:
            world[(world.shape[0] - point.y - 1, point.x)] = point.value

        return world

    def get_closest_food_pellet(self):
        distances = [abs(point.x - self.player.x) + abs(point.y - self.player.y) for point in self.food]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Point(-1, -1, 0)
        return self.food[idx_closest_distance]

    def reset(self):
        """ Reset the environment to the beginning """

        # Init variables
        self.health = 100
        self.total_reward = 0
        self.coordinates = []
        self.food = []
        self.current_step = 0
        self.max_step = 20

        # Initialize food pellets and player
        for i in range(3):
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=2)
            self.food.append(food_pellet)

        self.player = self.add_point(x=np.random.randint(self.width - 1),
                                     y=np.random.randint(self.height - 1),
                                     value=1)

        closest_food = self.get_closest_food_pellet()
        obs = np.array([closest_food.x - self.player.x, closest_food.y - self.player.y])

        return obs

    def move(self, action):
        """ Move the player to one space adjacent (up, right, down, left) """
        if action == 0 and self.player.y != (self.height - 1):  # up
            self.player.update(self.player.x, self.player.y + 1)
        elif action == 1 and self.player.x != (self.width - 1):  # right
            self.player.update(self.player.x + 1, self.player.y)
        elif action == 2 and self.player.y != 0:  # down
            self.player.update(self.player.x, self.player.y - 1)
        elif action == 3 and self.player.x != 0:  # left
            self.player.update(self.player.x - 1, self.player.y)

    def get_reward(self):
        """ Extract reward and whether the game has finished """
        closest_food = self.get_closest_food_pellet()

        reward = 0
        done = False

        if np.array_equal(self.player.coordinates, closest_food.coordinates):
            self.health += 50
            self.food.remove(closest_food)
            reward = 100
            self.total_reward += 100
        elif self.health <= 0:
            reward -= 400
            self.total_reward -= 400
            done = True
        elif self.current_step == self.max_step:
            done = True

        return reward, done

    def step(self, action):
        """ Move a single step """
        self.current_step += 1

        self.move(action)
        self.health -= 10
        reward, done = self.get_reward()

        if done:
            self.current_episode += 1
            self.scores.append(self.total_reward)

        # Add pellet
        if np.random.random() < 0.1:
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=2)
            self.food.append(food_pellet)

        closest_food = self.get_closest_food_pellet()
        obs = np.array([closest_food.x - self.player.x, closest_food.y - self.player.y])

        return obs, reward, done, {}

    def render(self):
        pygame.init()
        multiplier = 20
        self.screen = pygame.display.set_mode((round(self.width) * multiplier, round(self.height) * multiplier))
        clock = pygame.time.Clock()
        clock.tick(5)
        self.screen.fill((255, 255, 255))

        pygame.gfxdraw.filled_circle(self.screen, round(self.player.x * multiplier), round(self.player.y * multiplier),
                                     int(multiplier / 2), (255, 0, 0))

        for food in self.food:
            pygame.gfxdraw.filled_circle(self.screen, round(food.x * multiplier), round(food.y * multiplier),
                                         int(multiplier / 2), (0, 255, 0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True
