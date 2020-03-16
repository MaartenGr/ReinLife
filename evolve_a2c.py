from TheGame.Models.a2c import A2CAgent
from TheGame import Environment

# Hyperparameters
nr_agents = 1
learning_rate = 0.0005
best_score = -10_000
print_interval = 10

# Init env
brains = [A2CAgent(152, 8) for _ in range(nr_agents)]
env = Environment(width=20, height=20, nr_agents=nr_agents, evolution=True, brains=brains, grid_size=24)
env.max_step = 30_000
s = env.reset()

for n_epi in range(3_000):
    actions = [agent.brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        if agent.fitness > best_score:
            best_score = agent.fitness

        agent.brain.train_model(s[i], actions[i], r[i], s_prime[i], dones[i])

    s = env.update_env()

    if n_epi % 100 == 0:
        print(f"Best score: {best_score}. Nr episodes: {n_epi}. Nr_agents: {len(env.agents)}")

    # env.render(fps=50)

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=5)
