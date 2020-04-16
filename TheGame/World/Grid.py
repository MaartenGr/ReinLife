import numpy as np
from TheGame.World.Entities import Empty
from TheGame.World.utils import EntityTypes
from copy import deepcopy


class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.entity_type = EntityTypes

        self.grid = np.zeros([self.height, self.width], dtype=object)

        for i in range(self.height):
            for j in range(self.width):
                self.grid[i, j] = Empty((i, j), value=self.entity_type.empty)

    def copy(self):
        return deepcopy(self)

    def set(self, i, j, entity, **kwargs):
        """ Insert an entity at location i, j """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        self.grid[i, j] = entity((i, j), **kwargs)
        return self.grid[i, j]

    def get_numpy(self, value=None):
        """ Get a numpy representation of the grid """
        if value:
            vectorized = np.vectorize(lambda obj: obj.value == value)
        else:
            vectorized = np.vectorize(lambda obj: obj.value)
        return vectorized(self.grid)

    def get_entities(self, entity_type):
        """ Get all entities of a specific type """
        grid = self.get_numpy()
        coordinates = np.where(grid == entity_type)
        coordinates = [(i, j) for i, j in zip(coordinates[0], coordinates[1])]
        entities = [self.grid[coordinate] for coordinate in coordinates]

        return entities

    def set_random(self, entity, p, **kwargs):
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
                return None, None
        except ValueError:
            return None, None

    def get(self, i, j):
        """ Get an entity at location i, j """
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        return self.grid[i, j]

    def update(self, i, j, entity_1, k, l, entity_2):
        """ Update location i, j with entity_1 and k, l with entity_2 """
        self.grid[i, j] = entity_1
        self.grid[k, l] = entity_2

    def fov(self, i, j, dist):
        """ Get the fov (also through walls) for location i, j with distance dist """
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
