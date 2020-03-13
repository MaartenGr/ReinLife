class Agent:
    def __init__(self, coordinates, value, brain=None, gen=None):
        self.x, self.y = coordinates
        self.health = 200
        self.value = value
        self.coordinates = list(coordinates)
        self.target_coordinates = list(coordinates)
        self.dead = False
        self.done = False
        self.age = 0
        self.brain = brain
        self.x_target = None
        self.y_target = None
        self.reproduced = False
        self.gen = gen
        self.fitness = 0

    def move(self, x, y):
        self.x = x
        self.y = y
        self.coordinates = [x, y]

    def target_location(self, x, y):
        self.x_target = x
        self.y_target = y
        self.target_coordinates = [x, y]
