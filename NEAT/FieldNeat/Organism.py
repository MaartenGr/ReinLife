import neat
import numpy as np
from Field import Point


class Organism(Point):
    def __init__(self, x, y, genome, config):
        super().__init__(x, y, 0)
        self.health = 100
        self.age = 0
        self.genome = genome
        self.nr_food = 0
        self.config = config
        self.genome.fitness = 0

    def think(self, inputs):
        self.genome.fitness = 0
        brain = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        return np.argmax(brain.activate(inputs))

    def update_fitness(self):
        self.genome.fitness += self.age
