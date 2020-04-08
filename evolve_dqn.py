from TheGame.Models.dqn import DQNAgent
from TheGame import Environment
from TheGame.utils import Results

# Hyperparameters
save_best = True
max_epi = 10_000
learning_rate = 0.0005
track_results = Results(print_interval=500, interactive=True)

# Initialize Env
env = Environment(width=30, height=30, nr_agents=2, evolution=True, fps=20, max_agents=100,
                  brains=[DQNAgent(151, learning_rate) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    if n_epi % 30 == 0:
        epsilon = max(0.01, 0.08 - 0.08 * (n_epi / max_epi))  # Linear annealing from 8% to 1%

    for agent in env.agents:
        agent.action = agent.brain.get_action(agent.state, epsilon)
    s_prime, r, dones, infos = env.step()

    # Memorize
    for agent in env.agents:
        agent.brain.memorize(agent.state, agent.action, agent.reward / 200.0, agent.state_prime, agent.done)

    # Learn
    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            agent.brain.train()

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()

    # env.render()

if save_best:
    env.save_best_brain(max_epi)

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i], epsilon)
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()

    if not env.render():
        break
