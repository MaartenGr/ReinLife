from TheGame.Models.perdqn import DQNAgent
from TheGame import Environment
from TheGame.utils import Results

# Hyperparameters
save_best = True
max_epi = 6_000
track_results = Results(print_interval=100, interactive=True)

# Init env
env = Environment(width=60, height=30, nr_agents=1, max_agents=100,
                  evolution=True, fps=20, brains=[DQNAgent(152, 8)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    actions = [agent.brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    # Memorize & Learn
    for i, agent in enumerate(env.agents):
        agent.brain.append_sample(s[i], actions[i], r[i], s_prime[i], dones[i])
        if agent.brain.memory.tree.n_entries >= agent.brain.train_start:
            agent.brain.train_model()

    # Update model
    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            agent.brain.update_target_model()

    track_results.update_results(env.agents, n_epi, actions, r)
    s = env.update_env()

    # env.render()

if save_best:
    env.save_best_brain(max_epi)

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()

    if not env.render():
        break
