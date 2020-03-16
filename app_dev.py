from TheGame import Environment
import random

env = Environment(width=30, height=30, nr_agents=40, grid_size=24)
env.reset()

while True:
    actions = [random.randint(0, 8) for _ in range(40)]
    env.step(actions)
    env.update_env()
    env.render(fps=5)
