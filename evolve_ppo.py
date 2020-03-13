# Actor2Critic
# https://github.com/Ricocotam/DeepRL/blob/master/src/actor_critic.py


from TheGame import Environment
from TheGame.Models.ppo_no_lstm import PPO
import numpy as np
import torch
from torch.distributions import Categorical


#Hyperparameters
learning_rate = 0.0005
nr_agents = 1
best_agent = 0
best_score = -10_000
print_interval = 10
scores = [0.0 for _ in range(nr_agents)]

brains = [PPO() for _ in range(nr_agents)]
env = Environment(width=20, height=20, nr_agents=nr_agents, evolution=True, fps=20, brains=brains, grid_size=24)
env.max_step = 30_000

s = env.reset()
s = [np.array(i) for i in s]

for n_epi in range(4_000):
    a = []
    probs = []
    for i, agent in enumerate(env.agents):
        prob = agent.brain.pi(torch.from_numpy(s[i]).float())
        m = Categorical(prob)
        action = m.sample().item()
        a.append(action)
        probs.append(prob)

    s_prime, r, dones, infos = env.step(a)
    s_prime = [np.array(i) for i in s_prime]

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        if agent.fitness > best_score:
            best_score = agent.fitness

        agent.brain.put_data((s[i], a[i], r[i] / 100.0, s_prime[i], probs[i][a[i]].item(), dones[i]))

    s = s_prime

    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            if agent.brain.data:
                agent.brain.train()

    s = env.update_env()

    if n_epi % 100 == 0:
        print(f"Best score: {best_score}. Nr episodes: {n_epi}. Nr_agents: {len(env.agents)}")

    # Update new nr of agents here
    # It could be that some agents die and no new agents were created
    # Also include the new s, r, dones, infos info
    # env.render()

# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        prob = agent.brain.pi(torch.from_numpy(s[i]).float())
        m = Categorical(prob)
        action = m.sample().item()
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    s = [np.array(i) for i in s]

    if not env.render():
        break
