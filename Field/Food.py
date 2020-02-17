import numpy as np


class Food:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
        self.coordinates = np.array([x, y], dtype=int)
        self.entity_type = "Food"

    def update(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = np.array([x, y], dtype=int)