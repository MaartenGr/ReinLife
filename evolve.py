import numpy as np
from TheGame import Environment
from TheGame.Models.ppo import PPOAgent
from TheGame.Models.dqn import DQNAgent
from TheGame.Models.a2c import A2CAgent

from TheGame.utils import Results

# Hyperparameters
max_epi = 30_000
save_best = True
track_results = Results(print_interval=500, interactive=True, google_colab=False)
brains = [PPOAgent(151, 8, learning_rate=0.0001),
          DQNAgent(151, learning_rate=0.0005),
          A2CAgent(151, 8)]
env = Environment(width=30, height=30, max_agents=100, nr_agents=len(brains), evolution=True, fps=20,
                  brains=brains, grid_size=24)
s = env.reset()

for n_epi in range(max_epi):
    if n_epi % 30 == 0:
        epsilon = max(0.01, 0.08 - 0.08 * (n_epi / max_epi))  # Linear annealing from 8% to 1%

    for i, agent in enumerate(env.agents):

        if agent.brain.method == "PPO":
            action, prob = agent.brain.get_action(s[i])
            agent.action = action
            agent.prob = prob

        elif agent.brain.method == "DQN":
            agent.action = agent.brain.get_action(s[i], epsilon)

        elif agent.brain.method == "A2C":
            agent.action = agent.brain.get_action(s[i])

        else:
            print(agent.brain.method)

    s_prime, r, dones, infos = env.step()

    # Learn only if still alive (not done)
    for agent in env.agents:

        if agent.brain.method == "PPO":
            agent.brain.put_data((agent.state, agent.action, agent.reward / 200.0,
                                  agent.state_prime, agent.prob[agent.action].item(), agent.done))

            if agent.age % 5 == 0 or agent.dead:
                if agent.brain.data():
                    agent.brain.learn()

        elif agent.brain.method == "DQN":
            agent.brain.memorize(agent.state, agent.action, agent.reward / 200.0,
                                 agent.state_prime, agent.done)

            if agent.age % 20 == 0 or agent.dead:
                agent.brain.train()

        elif agent.brain.method == "A2C":
            if agent.dead or agent.age % 20 == 0:
                agent.brain.train_model(agent.state, agent.action, agent.reward / 200.0,
                                        agent.state_prime, agent.done)

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()

    for agent in env.agents:
        agent.state = agent.state_prime

# if save_best:
#     env.save_best_brain(max_epi)

# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        if agent.brain.method == "PPO":
            action, prob = agent.brain.get_action(s[i])
            agent.action = action
            agent.prob = prob

        elif agent.brain.method == "DQN":
            agent.action = agent.brain.get_action(s[i], epsilon)

        elif agent.brain.method == "A2C":
            agent.action = agent.brain.get_action(s[i])
    _, rewards, dones, info = env.step()
    s = env.update_env()

    if not env.render():
        break
