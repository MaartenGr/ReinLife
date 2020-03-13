import random
import pygame as pg
import numpy as np
from TheGame.Environments.utils import Entities


class Visualize:
    def __init__(self, width, height, grid_size):

        # Track size
        self.width = width
        self.height = height
        self.grid_size = grid_size

        # other
        self.entities = Entities
        self.colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                       (255, 0, 255),
                       (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0), (0, 128, 0), (128, 0, 128),
                       (0, 128, 128), (0, 0, 128)]

        # Pygame related vars
        self.background = None
        self.clock = None
        self.screen = None
        self.rendered = False

    def render(self, agents, grid, fps):
        """ Draw all sprites and tiles """
        if not self.rendered:
            # Init pg
            pg.init()
            self.screen = pg.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self.clock = pg.time.Clock()
            self.clock.tick(50)

            # Background
            self.background = pg.Surface((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self._get_tile_colors()
            self._draw_tiles()

            self.rendered = True

        self.screen.blit(self.background, (0, 0))
        self._draw_agents(agents)
        self._draw_food(grid)
        pg.display.update()
        self.clock.tick(fps)

        return check_pygame_exit()

    def _draw_agents(self, agents):
        for agent in agents:
            if not agent.dead:
                if agent.gen < len(self.colors):
                    color = self.colors[agent.gen]
                else:
                    color = self.colors[agent.gen % len(self.colors)]

                # Body
                pg.draw.rect(self.screen, color,
                             ((agent.x * self.grid_size) + max(1, int(self.grid_size / 8)),
                              (agent.y * self.grid_size) + max(1, int(self.grid_size / 8)),
                              self.grid_size - max(1, int(self.grid_size / 8) * 2),
                              self.grid_size - max(1, int(self.grid_size / 8)) * 2), 0)

                pg.draw.rect(self.screen, (0, 0, 0),
                             ((agent.x * self.grid_size) + max(1, int(self.grid_size / 8)),
                              (agent.y * self.grid_size) + max(1, int(self.grid_size / 8)),
                              self.grid_size - max(1, int(self.grid_size / 8) * 2),
                              self.grid_size - max(1, int(self.grid_size / 8)) * 2), 2)

                # Eyes
                pg.draw.rect(self.screen, (0, 0, 0),
                             ((agent.x * self.grid_size) + max(1, int(self.grid_size / 3)),
                              (agent.y * self.grid_size) + max(1, int(self.grid_size / 3)),
                              self.grid_size - max(1, int(self.grid_size * .9)),
                              self.grid_size - max(1, int(self.grid_size * .9))), 0)

                pg.draw.rect(self.screen, (0, 0, 0),
                             ((agent.x * self.grid_size) + max(1, int(self.grid_size / 1.8)),
                              (agent.y * self.grid_size) + max(1, int(self.grid_size / 3)),
                              self.grid_size - max(1, int(self.grid_size * .9)),
                              self.grid_size - max(1, int(self.grid_size * .9))), 0)

    def _draw_food(self, grid):
        food = np.where(grid == self.entities.food)
        for i, j in zip(food[0], food[1]):
            pg.draw.rect(self.screen, (255, 255, 255), ((i * self.grid_size) + int(self.grid_size/2.5),
                                                        (j * self.grid_size) + int(self.grid_size/2.5),
                                                        self.grid_size - int(self.grid_size/2.5)*2,
                                                        self.grid_size - int(self.grid_size/2.5)*2), 0)

        food = np.where(grid == self.entities.poison)
        for i, j in zip(food[0], food[1]):
            pg.draw.rect(self.screen, (0, 0, 0), ((i * self.grid_size) + int(self.grid_size/2.5),
                                                  (j * self.grid_size) + int(self.grid_size/2.5),
                                                  self.grid_size - int(self.grid_size/2.5)*2,
                                                  self.grid_size - int(self.grid_size/2.5)*2), 0)

    def _get_tile_colors(self):
        self.tile_colors = {i: {j: None} for i in range(self.width) for j in range(self.height)}
        for i in range(self.width):
            for j in range(self.height):
                offset_r = 0
                offset_g = 0
                offset_b = 0
                if random.random() > (1-.9):
                    offset_r = random.randint(-30, 30)

                self.tile_colors[i][j] = (50 + offset_r, 205 + offset_g, 50 + offset_b)

    def _draw_tiles(self):
        """ Draw tiles on background """
        for i in range(self.width):
            for j in range(self.height):
                pg.draw.rect(self.background, self.tile_colors[i][j],
                             (i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size), 0)

    def _step_human(self):
        """ Execute an action manually """
        if self.mode == 'human':
            events = pg.event.get()
            action = 4
            actions = [10 for _ in range(1)]
            for event in events:
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_UP:
                        actions[0] = 3
                    if event.key == pg.K_RIGHT:
                        actions[0] = 2
                    if event.key == pg.K_DOWN:
                        actions[0] = 1
                    if event.key == pg.K_LEFT:
                        actions[0] = 0
                    self.step(actions)
                    obs, family = self._get_obs()
                    print(f"{obs[0].T}, {len(obs[0])}")
                    print(family)
                    print([agent.coordinates for agent in self.agents])

                    # fov_food, fov_agents = self._get_fov_matrix(self.entities["Agents"][0])
                    #
                    # print(f"Health : {self.entities['Agents'][0].health}")
                    # print(f"Dead: {str(self.entities['Agents'][0].dead)}")
                    # print(f"Enemies: {fov_agents}")


def check_pygame_exit():
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            return False
    return True
