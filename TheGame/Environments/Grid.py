import numpy as np


class Grid:
    """
    Represent a grid and operations on it
    """

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = np.zeros([self.height, self.width])

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.height
        assert j >= 0 and j < self.width
        self.grid[i, j] = v

    def set_random(self, v, p):
        i, j = np.random.randint(0, self.width), np.random.randint(0, self.height)
        zero_indices = np.argwhere(self.grid == 0)
        if zero_indices.size != 0:
            random_index = np.random.randint(0, zero_indices.shape[0])
            i, j = zero_indices[random_index]
            if np.random.random() < p:
                self.grid[i, j] = v
        return i, j

    def get(self, i, j):
        assert i >= 0 and j < self.width
        assert i >= 0 and j < self.height
        return self.grid[i, j]

    def reset(self, v, t):
        self.grid = np.where(self.grid == v, t, self.grid)

    def update(self, i, j, v1, k, l, v2):
        self.grid[i, j] = v1
        self.grid[k, l] = v2

    def fov(self, i, j, dist):
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
