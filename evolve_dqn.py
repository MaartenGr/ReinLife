from TheGame.Models.dqn import DQNAgent
from TheGame import Environment

# Hyperparameters
learning_rate = 0.0005
nr_agents = 1
best_score = -10_000
print_interval = 10

# Initialize Env
brains = [DQNAgent(152, learning_rate) for _ in range(nr_agents)]
env = Environment(width=20, height=20, nr_agents=nr_agents, evolution=True, fps=20, brains=brains, grid_size=24)
env.max_step = 30_000
s = env.reset()

for n_epi in range(3_000):
    if n_epi % 30 == 0:
        epsilon = max(0.01, 0.08 - 0.08 * (n_epi / 500))  # Linear annealing from 8% to 1%

    actions = [agent.brain.get_action(s[i], epsilon) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    # Memorize
    for i, agent in enumerate(env.agents):
        if agent.fitness > best_score:
            best_score = agent.fitness

        agent.brain.memorize(s[i], actions[i], r[i] / 200.0, s_prime[i], dones[i])

    # Learn
    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            agent.brain.train()

    s = env.update_env()

    if n_epi % 100 == 0:
        print(f"Best score: {best_score}. Nr episodes: {n_epi}. Nr_agents: {len(env.agents)}")

    # env.render()

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i], epsilon)
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()

    if not env.render():
        break
