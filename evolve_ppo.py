import numpy as np
from TheGame import Environment
from TheGame.Models.ppo import PPO, PPOAgent
from TheGame.utils import Results

# Hyperparameters
max_epi = 70_000
save_best = True
track_results = Results(print_interval=500, interactive=True, google_colab=False)
env = Environment(width=30, height=30, max_agents=50, nr_agents=5, evolution=True, fps=20,
                  brains=[PPOAgent(151, 8, learning_rate=0.0001) for _ in range(5)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    a = []
    probs = []
    for i, agent in enumerate(env.agents):
        action, prob = agent.brain.get_action(s[i])
        agent.action = action
        agent.prob = prob

    s_prime, r, dones, infos = env.step()

    # Learn only if still alive (not done)
    for agent in env.agents:
        agent.brain.put_data((agent.state, agent.action, agent.reward / 100.0,
                              agent.state_prime, agent.prob[agent.action].item(), agent.done))

        if agent.age % 20 == 0 or agent.dead:
            if agent.brain.data():
                agent.brain.learn()

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()

    for agent in env.agents:
        agent.state = agent.state_prime


if save_best:
    env.save_best_brain(max_epi)


# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action, _ = agent.brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    s = [np.array(i) for i in s]

    if not env.render():
        break
