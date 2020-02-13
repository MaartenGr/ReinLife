import numpy as np

# Pygame
import pygame
from pygame import gfxdraw

# Custom
from Field import Point
from Field.Organism import Organism


class Environment:
    def __init__(self, organisms, config, width=10, height=12):
        self.config = config
        self.width = width
        self.height = height
        self.organisms = organisms
        self.food = []
        self.init_food(n=5)
        self.best_organism = organisms[0]

    def init_food(self, n):
        for i in range(n):
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=2)
            self.food.append(food_pellet)

    def get_closest_food_pellet(self, organism):
        distances = [abs(point.x - organism.x) + abs(point.y - organism.y) for point in self.food]
        if distances:
            idx_closest_distance = int(np.argmin(distances))
        else:
            return Point(-1, -1, 0)
        return self.food[idx_closest_distance]

    def move(self, action, organism):
        # print(action)
        """ Move the agent to one space adjacent (up, right, down, left) """
        if action == 0 and organism.y != (self.height - 1):  # up
            organism.update(organism.x, organism.y + 1)
        elif action == 1 and organism.x != (self.width - 1):  # right
            organism.update(organism.x + 1, organism.y)
        elif action == 2 and organism.y != 0:  # down
            organism.update(organism.x, organism.y - 1)
        elif action == 3 and organism.x != 0:  # left
            organism.update(organism.x - 1, organism.y)

    def eat_food(self, organism, closest_food):
        if np.array_equal(organism.coordinates, closest_food.coordinates):
            organism.health += 50
            self.food.remove(closest_food)
            organism.nr_food += 5

    def update(self):
        for organism in self.organisms:
            closest_food = self.get_closest_food_pellet(organism)
            obs = np.array([closest_food.x - organism.x, closest_food.y - organism.y, organism.age, organism.nr_food])
            action = organism.think(obs)
            self.move(action, organism)
            organism.health -= 10
            organism.age += 1
            self.eat_food(organism, closest_food)

            if organism.health <= 0:
                organism.update_fitness()
                fitnesses = [organism.genome.fitness for organism in self.organisms]
                parent1 = self.organisms[np.argmax(fitnesses)]
                parent2 = self.organisms[np.argmax(fitnesses)]

                if self.best_organism.genome.fitness < np.argmax(fitnesses):
                    parent1 = self.best_organism
                    parent2 = self.best_organism
                    self.best_organism = self.organisms[np.argmax(fitnesses)]

                child = self.config.genome_type(1)
                child.configure_crossover(parent1.genome, parent2.genome, self.config.genome_config)
                child.mutate(self.config.genome_config)
                child = Organism(np.random.randint(self.width - 1),
                                 np.random.randint(self.height - 1), child, self.config)
                self.organisms.append(child)
                self.organisms.remove(organism)

            # elif organism.age % 7 == 0:
            #     organism.update_fitness()
            #     fitnesses = [organism.genome.fitness for organism in self.organisms]
            #
            #     if organism.genome.fitness == min(fitnesses):
            #         parent1 = self.organisms[np.argmax(fitnesses)]
            #         parent2 = self.organisms[np.argmax(fitnesses)]
            #         if self.best_organism.genome.fitness < np.argmax(fitnesses):
            #             parent1 = self.best_organism
            #             parent2 = self.best_organism
            #             self.best_organism = self.organisms[np.argmax(fitnesses)]
            #
            #         child = self.config.genome_type(1)
            #         child.configure_crossover(parent1.genome, parent2.genome, self.config.genome_config)
            #         child.mutate(self.config.genome_config)
            #         child = Organism(np.random.randint(self.width - 1),
            #                          np.random.randint(self.height - 1), child, self.config)
            #         self.organisms.append(child)
            #         self.organisms.remove(organism)


        if np.random.random() < 0.05:
            food_pellet = Point(x=np.random.randint(self.width - 1),
                                y=np.random.randint(self.height - 1), value=2)
            self.food.append(food_pellet)


    def render(self):
        pygame.init()
        multiplier = 20
        self.screen = pygame.display.set_mode((round(self.width) * multiplier, round(self.height) * multiplier))
        clock = pygame.time.Clock()
        clock.tick(5)
        self.screen.fill((255, 255, 255))

        for food in self.food:
            pygame.gfxdraw.filled_circle(self.screen, round(food.x * multiplier), round(food.y * multiplier),
                                         int(multiplier / 2), (0, 255, 0))

        for organism in self.organisms:
            pygame.gfxdraw.filled_circle(self.screen, round(organism.x * multiplier), round(organism.y * multiplier),
                                         int(multiplier / 2), (255, 0, 0))

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True
