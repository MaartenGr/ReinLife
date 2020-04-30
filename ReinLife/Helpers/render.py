import random
import numpy as np
import pygame as pg
from typing import List
from ReinLife.World.utils import EntityTypes
from ReinLife.World.entities import Entity, Agent


class Visualize:
    """ Used for rendering the game using pygame

    Parameters:
    -----------

    width, height : int
        The width and height of the environment

    grid_size : int
        The size of each grid in pixels: (grid_size * grid_size)

    pastel : bool, default False
        Whether to use pastel colors for the agents
    """
    def __init__(self, width: int, height: int, grid_size: int, pastel: bool = False):

        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.entities = EntityTypes

        if pastel:
            self.colors = [tuple(generate_random_pastel()) for _ in range(100)]
        else:
            self.colors = [
                (24, 255, 255),  # Light Blue
                (255, 94, 89),  # Red
                (255, 238, 88),  # Yellow
                (255, 255, 255),  # White
                (126, 87, 194),  # Purple
                (66, 165, 245),  # Light Blue
                (121, 85, 72),  # Brown
                (0, 200, 83),  # Green
            ]

        # Pygame related vars
        self.background = None
        self.clock = None
        self.screen = None
        self.rendered = False

    def render(self, agents: List[Agent], grid: np.array, fps: int) -> bool:
        """ Draw and render all sprites and tiles

        Parameters:
        -----------
        agents : List[Agent]
            List of Agent classes

        grid : np.array
            A grid with all entities in the environment

        fps : int
            Frames per second

        Returns:
        --------
        bool
            To indicate whether the game has exited or not
        """
        # Initialize pygame
        if not self.rendered:
            pg.init()
            self.screen = pg.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self.clock = pg.time.Clock()
            self.clock.tick(fps)

            # Background
            self.background = pg.Surface((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self._get_tile_colors()
            self._draw_tiles()
            self.rendered = True

        # Draw and update all entities
        self.screen.blit(self.background, (0, 0))
        self._draw_agents(agents)
        self._draw_food(grid)
        pg.display.update()
        self.clock.tick(fps)

        return check_pygame_exit()

    def _draw_agents(self, agents: List[Agent]):
        """ Draw all agents

        Each agent has a body color based on its gene value.
        If the agent is at full health, then its border is black. The lower it becomes, the more
        the border color becomes its body color.

        Parameters:
        ----------
        agents : List[Agent]
            List of Agent classes
        """
        for agent in agents:
            if not agent.dead:
                if agent.gene < len(self.colors):
                    body_color = self.colors[agent.gene]
                else:
                    body_color = self.colors[agent.gene % len(self.colors)]

                self._draw_agent_body(agent, body_color)
                self._draw_agent_border(agent, body_color)
                self._draw_agent_eyes(agent)

    def _draw_agent_body(self, agent: Agent, color: tuple):
        """ Draw the square body of the agent

        Parameters:
        -----------
        agent : Agent
            An agent for which to draw the body

        color : tuple
            RGB tuple to color the body of the agent
        """
        j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 8))
        i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 8))
        size = self.grid_size - max(1, int(self.grid_size / 8) * 2)
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, color, surface, 0)

    def _draw_agent_border(self, agent: Agent, color: tuple):
        """ Draw the border agent's body

        The color of the agent's body starts out black and slowly becomes more
        like the color of its body the lower its health becomes.

        If the agent succesfully kills an entity, the agent's border becomes red

        Parameters:
        -----------
        agent : Agent
            An agent for which to draw the body

        color : tuple
            RGB tuple to color the body of the agent
        """
        if agent.killed == 1:
            border_color = (255, 0, 0)
        else:
            health_multiplier = agent.health / 205
            border_color = lerp(np.array(color), np.array([0, 0, 0]), health_multiplier)

        j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 8))
        i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 8))
        size = self.grid_size - max(1, int(self.grid_size / 8) * 2)
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, border_color, surface, 2)

    def _draw_agent_eyes(self, agent: Agent):
        """ Draw the eyes of an agent

        Parameters:
        -----------
        agent : Agent
            An agent for which to draw the body
        """
        size = self.grid_size - max(1, int(self.grid_size * .9))

        # Left eyes
        j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 3))
        i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 3))
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

        # Right eyes
        j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 1.8))
        i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 3))
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

    def _draw_food(self, grid: np.array):
        """ Draw all types of food

        Parameters:
        -----------
        grid : np.array
            A grid with all entities in the environment
        """
        food = grid.get_entities(self.entities.food)
        for item in food:
            self._draw_food_rect(item, color=(255, 255, 255))

        poison = grid.get_entities(self.entities.poison)
        for item in poison:
            self._draw_food_rect(item, color=(0, 0, 0))

        berry = grid.get_entities(self.entities.super_food)
        for item in berry:
            self._draw_food_rect(item, color=(255, 0, 0))

    def _draw_food_rect(self, item: Entity, color: tuple):
        """ Draw a rectangle for a food item

        Parameters:
        -----------
        item : Entity
            An Entity class for which to draw the rectangle

        color : tuple
            RGB tuple to color the body of the rectangle
        """
        j = (item.j * self.grid_size) + int(self.grid_size / 2.5)
        i = (item.i * self.grid_size) + int(self.grid_size / 2.5)
        size = self.grid_size - int(self.grid_size / 2.5) * 2
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, color, surface, 0)

    def _get_tile_colors(self):
        """ Create green background tiles that are each slightly differently shaded """
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
        """ Draw green tiles on background """
        for i in range(self.width):
            for j in range(self.height):
                pg.draw.rect(self.background, self.tile_colors[i][j],
                             (i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size), 0)


def check_pygame_exit() -> bool:
    """ Check whether someone has quit the simulation """
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            return False
    return True


def generate_random_pastel() -> tuple:
    """ Generate a random pastel color """
    red = (random.randint(0, 255) + 255) / 2
    green = (random.randint(0, 255) + 255) / 2
    blue = (random.randint(0, 255) + 255) / 2
    return red, green, blue


def lerp(a: np.array, b: np.array, t: float) -> np.array:
    """ Linearly interpolate between colors a and b with multiplier t """
    return a*(1 - t) + b*t
