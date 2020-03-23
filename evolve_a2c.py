from TheGame.Models.a2c import A2CAgent
from TheGame import Environment
from TheGame.utils import Results

# Hyperparameters
max_epi = 10_000
save_best = True
track_results = Results(print_interval=100, interactive=True)

# Init env
env = Environment(width=20, height=20, nr_agents=1, evolution=True, brains=[A2CAgent(152, 8)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    actions = [agent.brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    for i, agent in enumerate(env.agents):
        agent.brain.train_model(s[i], actions[i], r[i], s_prime[i], dones[i])

    track_results.update_results(env.agents, n_epi, actions)
    s = env.update_env()

    # env.render(fps=50)

if save_best:
    env.save_best_brain(max_epi)

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=5)
