import numpy as np


class Food:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
        self.coordinates = (x, y)
        self.name = "Food"

    def update(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = (x, y)