from TheGame.Models.drqn import DRQNAgent
from TheGame import Environment
from TheGame.utils import Results

# Hyperparameters
gamma = 0.99
learning_rate = 1e-3
soft_update_freq = 100
capacity = 10000
exploration = 300
epsilon_init = 0.9
epsilon_min = 0.05
decay = 0.99
episode = 1000000
render = False
save_best = True
max_epi = 40_000
track_results = Results(print_interval=100, interactive=True)

# Initialize Env
env = Environment(width=20, height=20, nr_agents=1, evolution=True, fps=20,
                  brains=[DRQNAgent(learning_rate, capacity, epsilon_init, gamma, soft_update_freq)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):

    for agent in env.agents:
        agent.brain.count += 1
        if agent.brain.epsilon > epsilon_min:
            agent.brain.epsilon = agent.brain.epsilon * decay

    if n_epi % 30 == 0:
        epsilon = max(0.01, 0.08 - 0.08 * (n_epi / max_epi))  # Linear annealing from 8% to 1%

    actions = [agent.brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    # Memorize
    for i, agent in enumerate(env.agents):
        agent.brain.memorize(s[i], actions[i], r[i] / 200.0, s_prime[i], dones[i])

    # Learn
    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            agent.brain.train()

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
