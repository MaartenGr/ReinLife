import random
import numpy as np
import pygame as pg
from TheGame.World.utils import EntityTypes


class Visualize:
    def __init__(self, width, height, grid_size, pastel=False, families=True):

        # Track size
        self.width = width
        self.height = height
        self.grid_size = grid_size

        # other
        self.entities = EntityTypes
        self.families = families

        if pastel:
            self.colors = [tuple(generate_random_pastel()) for _ in range(100)]
        elif self.families:
            self.colors = [(0, 0, 255), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 255, 255)]
        else:
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
            self.clock.tick(fps)

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
                if self.families:
                    color = self.colors[agent.gen]
                else:
                    if agent.gen < len(self.colors):
                        color = self.colors[agent.gen]
                    else:
                        color = self.colors[agent.gen % len(self.colors)]

                # Body
                pg.draw.rect(self.screen, color,
                             ((agent.j * self.grid_size) + max(1, int(self.grid_size / 8)),
                              (agent.i * self.grid_size) + max(1, int(self.grid_size / 8)),
                              self.grid_size - max(1, int(self.grid_size / 8) * 2),
                              self.grid_size - max(1, int(self.grid_size / 8)) * 2), 0)
                if agent.killed == 1:
                    border_color = (255, 0, 0)
                else:
                    health_multiplier = agent.health / 205
                    border_color = lerp(np.array(color), np.array([0, 0, 0]), health_multiplier)


                pg.draw.rect(self.screen, border_color,
                             ((agent.j * self.grid_size) + max(1, int(self.grid_size / 8)),
                              (agent.i * self.grid_size) + max(1, int(self.grid_size / 8)),
                              self.grid_size - max(1, int(self.grid_size / 8) * 2),
                              self.grid_size - max(1, int(self.grid_size / 8)) * 2), 2)

                # Eyes
                pg.draw.rect(self.screen, (0, 0, 0),
                             ((agent.j * self.grid_size) + max(1, int(self.grid_size / 3)),
                              (agent.i * self.grid_size) + max(1, int(self.grid_size / 3)),
                              self.grid_size - max(1, int(self.grid_size * .9)),
                              self.grid_size - max(1, int(self.grid_size * .9))), 0)

                pg.draw.rect(self.screen, (0, 0, 0),
                             ((agent.j * self.grid_size) + max(1, int(self.grid_size / 1.8)),
                              (agent.i * self.grid_size) + max(1, int(self.grid_size / 3)),
                              self.grid_size - max(1, int(self.grid_size * .9)),
                              self.grid_size - max(1, int(self.grid_size * .9))), 0)

    def _draw_food(self, grid):
        food = grid.get_entities(self.entities.food)
        for item in food:
            pg.draw.rect(self.screen, (255, 255, 255), ((item.j * self.grid_size) + int(self.grid_size/2.5),
                                                        (item.i * self.grid_size) + int(self.grid_size/2.5),
                                                        self.grid_size - int(self.grid_size/2.5)*2,
                                                        self.grid_size - int(self.grid_size/2.5)*2), 0)

        poison = grid.get_entities(self.entities.poison)
        for item in poison:
            pg.draw.rect(self.screen, (0, 0, 0), ((item.j * self.grid_size) + int(self.grid_size/2.5),
                                                  (item.i * self.grid_size) + int(self.grid_size/2.5),
                                                  self.grid_size - int(self.grid_size/2.5)*2,
                                                  self.grid_size - int(self.grid_size/2.5)*2), 0)

        berry = grid.get_entities(self.entities.super_food)
        for item in berry:
            pg.draw.rect(self.screen, (255, 0, 0), ((item.j * self.grid_size) + int(self.grid_size/2.5),
                                                  (item.i * self.grid_size) + int(self.grid_size/2.5),
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


def generate_random_pastel():
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)

    red = (red + 255) / 2
    blue = (blue + 255) / 2
    green = (green + 255) / 2

    return red, green, blue


def lerp(a, b, t):
    return a*(1 - t) + b*t
