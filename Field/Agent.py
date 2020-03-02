import numpy as np


class Agent:
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.health = 200
        self.value = value
        self.coordinates = (x, y)
        self.name = "Agent"
        self.dead = False
        self.action = None
        self.age = 0

    def update(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = (x, y)

    def update_action(self, action):
        self.action = action

    def get_action(self):
        action = self.action
        self.action = None
        return action
