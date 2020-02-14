import numpy as np

# Gym
import gym
from gym import spaces

# Pygame
import pygame
from pygame import gfxdraw

# Custom
from Field.utils import check_pygame_exit, Grid
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
        
        # Init variables
        self._reset_variables()

    def reset(self):
        """ Reset the environment to the beginning """
        self._reset_variables()
        self._init_objects()  # Init player and food pellets
        obs = self._get_obs()
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

        obs = self._get_obs()
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

    def update_fov(self):
        # Initialize matrix
        fov = np.zeros([2, 5, 5])
        # fov[2, 2] = -1

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

        # # Add food at the correct location in the matrix
        # for food in self.food:
        #     if abs(food.x - self.player.x) + abs(food.y - self.player.y) <= 3:
        #         diff_x = food.x - self.player.x
        #         diff_y = self.player.y - food.y
        #         fov[3 - diff_y, 3 + diff_x] = food.value
        #
        # # Flatten and remove coordinates with distance > 2 and remove player coordinate
        # center = (3, 3)
        # selected_coordinates = [(i, j) for i in range(7) for j in range(7) if
        #                         abs(i - center[0]) + abs(j - center[1]) <= 3]
        # selected_coordinates.remove(center)
        #
        # # Flatten matrix and keep selected values
        # fov = [fov[coordinate] for coordinate in selected_coordinates]
        # fov.append(self.player.x / self.width)
        # fov.append(self.player.y / self.height)
        # # print(fov)
        #
        # # fov = np.array([])
        return fov

    def _get_obs(self):
        closest_food = self._get_closest_food_pellet()
        obs = np.array([closest_food.x - self.player.x, closest_food.y - self.player.y])
        obs = np.array([(closest_food.x - self.player.x) / self.width,
                        (closest_food.y - self.player.y) / self.height,
                        closest_food.value / 2,
                        self.player.x / self.width,
                        self.player.y / self.height])

        return obs

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

    def _draw(self):
        # Draw Grid
        Grid(surface=self.screen, cellSize=16).create_grid()

        # Draw player
        pygame.gfxdraw.filled_circle(self.screen,
                                     round(self.player.x * self.grid_size)+round(self.grid_size/2),
                                     round(self.player.y * self.grid_size)+round(self.grid_size/2),
                                     int(self.grid_size / 2), (255, 0, 0))

        # Draw food
        for food in self.food:
            if food.value == 1:
                pygame.gfxdraw.filled_circle(self.screen,
                                             round(food.x * self.grid_size)+round(self.grid_size/2),
                                             round(food.y * self.grid_size)+round(self.grid_size/2),
                                             int(self.grid_size / 2), (0, 255, 0))
            elif food.value == -1:
                pygame.gfxdraw.filled_circle(self.screen,
                                             round(food.x * self.grid_size)+round(self.grid_size/2),
                                             round(food.y * self.grid_size)+round(self.grid_size/2),
                                             int(self.grid_size / 2), (0, 0, 255))

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



