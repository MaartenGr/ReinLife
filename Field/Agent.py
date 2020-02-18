import numpy as np


class Agent:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.health = 200
        self.value = value
        self.coordinates = np.array([x, y], dtype=int)
        self.name = "Agent"
        self.dead = False

    def update(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = np.array([x, y], dtype=int)