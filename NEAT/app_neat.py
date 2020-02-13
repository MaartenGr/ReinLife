import neat
from FieldNeat.Organism import Organism
from FieldNeat.EnvNeat import Environment

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "FieldNeat/config-neat")
p = neat.Population(config)
organisms = [Organism(1, 1, p.population[key], config) for key in p.population.keys()]
env = Environment(organisms, config)
#
# while True:
#     env.update()
#     env.render()

for i in range(5_000):
    if i % 5_000 == 0:
        print(i)

    env.update()
    if i > 4_000:
        env.render()
