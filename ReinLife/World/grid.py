import numpy as np
from copy import deepcopy
from typing import List, Type

from .entities import Entity
from .utils import EntityTypes


class Grid:
    """ Represents a basic grid and operations within it

    Parameters:
    -----------
    width : int
        The width of the grid

    heigth : int
        The height of the grid

    """

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.entity_type = EntityTypes

        self.grid = np.zeros([self.height, self.width], dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i, j] = Entity((i, j), entity_type=self.entity_type.empty)

    def copy(self):
        """ Make a copy of the grid """
        return deepcopy(self)

    def get(self, i: int, j: int) -> Entity:
        """ Get an entity at location i, j """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        return self.grid[i, j]

    def set(self, i: int, j: int, entity: Type[Entity], **kwargs) -> np.array:
        """ Insert an entity at location i, j """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        self.grid[i, j] = entity((i, j), **kwargs)
        return self.grid[i, j]

    def get_numpy(self, entity_type: int = None) -> np.array:
        """ Get a numpy representation of the grid """
        if entity_type:
            vectorized = np.vectorize(lambda obj: obj.entity_type == entity_type)
        else:
            vectorized = np.vectorize(lambda obj: obj.entity_type)
        return vectorized(self.grid)

    def get_entities(self, entity_type: int = None) -> List[Entity]:
        """ Get all entities of a specific type """
        grid = self.get_numpy()
        coordinates = np.where(grid == entity_type)
        coordinates = [(i, j) for i, j in zip(coordinates[0], coordinates[1])]
        entities = [self.grid[coordinate] for coordinate in coordinates]

        return entities

    def set_random(self, entity: Type[Entity], p: float, **kwargs) -> np.array:
        """ Set an entity at a random location iff there is space """
        grid = self.get_numpy()
        indices = np.where(grid == self.entity_type.empty)

        try:
            random_index = np.random.randint(0, len(indices[0]))
            i, j = indices[0][random_index], indices[1][random_index]
            if np.random.random() < p:
                self.grid[i, j] = entity((i, j), **kwargs)
                return self.grid[i, j]
            else:
                return None
        except ValueError:
            return None

    def update(self, i: int, j: int, entity_1: Type[Entity], k: int, l: int, entity_2: Type[Entity]):
        """ Update location i, j with entity_1 and k, l with entity_2 """
        self.grid[i, j] = entity_1
        self.grid[k, l] = entity_2

    def fov(self, i: int, j: int, dist: int, grid: np.ndarray = None) -> np.ndarray:
        """ Get the fov (also through walls) for location i, j with distance dist
        If grid is given, use that grid to extract the fov from i, j, and dist.
        """

        if grid is None:
            grid = self.grid

        # Get sections
        top = grid[:dist, :]
        bottom = grid[self.height - dist:, :]
        right = grid[:, self.width - dist:]
        left = grid[:, :dist]
        lower_left = grid[self.height - dist:, :dist]
        lower_right = grid[self.height - dist:, self.width - dist:]
        upper_left = grid[:dist, :dist]
        upper_right = grid[:dist, self.width - dist:]

        # Create top, middle and bottom sections
        full_top = np.concatenate((lower_right, bottom, lower_left), axis=1)
        full_middle = np.concatenate((right, grid, left), axis=1)
        full_bottom = np.concatenate((upper_right, top, upper_left), axis=1)

        # Apply fov based on dist
        fov = np.concatenate((full_top, full_middle, full_bottom), axis=0)
        fov = fov[i:i + (2 * dist) + 1, j:j + (2 * dist) + 1]

        return fov
