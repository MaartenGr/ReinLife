import numpy as np
from TheGame import Environment
from TheGame.Models.ppo import PPO, PPOAgent
from TheGame.utils import Results

# Hyperparameters
max_epi = 70_000
save_best = True
track_results = Results(print_interval=500, interactive=True, google_colab=False)
env = Environment(width=60, height=30, max_agents=50, nr_agents=5, evolution=True, fps=20,
                  brains=[PPOAgent(150, 8, learning_rate=0.0001) for _ in range(5)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    a = []
    probs = []
    for i, agent in enumerate(env.agents):
        action, prob = agent.brain.get_action(s[i])
        a.append(action)
        probs.append(prob)

    s_prime, r, dones, infos = env.step(a)

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        agent.brain.put_data((s[i], a[i], r[i] / 100.0, s_prime[i], probs[i][a[i]].item(), dones[i]))

        if agent.age % 5 == 0 or agent.dead:
            if agent.brain.data():
                agent.brain.learn()

    track_results.update_results(env.agents, n_epi, a, r)
    s = env.update_env()


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
