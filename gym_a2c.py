# To do:
#   * Make sure to change the dones such that it does not update the memory or execute an action after its done
#   * Permanently remove an agent if dead (or prevent any actions) --> preferably removed from env
#   * Add 'deep neuroevolution'
#       * https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
#   * Only update coordinates after everyone has attacked
#   * Separate self.entities["Food"] by poison and apple
#   * Try to add:
#       * https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
# Technically, a dead agent is still alive seeing as it has an x, y coordinate and has the possibility to eat stuff...


############################## IMPORTANT #############################
# ADD: https://github.com/seungeunrho/minimalRL/blob/master/ppo.py ###
#####################################################################


import torch.optim as optim
import torch

from Field.a2c import A2CAgent
from Field import GridWorld
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

env = GridWorld(width=10, height=10, nr_agents=1, fps=5)
env.max_step = 30

agents = [A2CAgent(151, 8) for _ in range(1)]

print_interval = 10
scores = [0.0 for _ in range(len(agents))]

for n_epi in range(1_000):

    # Initialize Variables
    s = env.reset()
    s = [np.reshape(state, [1, 151]) for state in s]

    dones = [False for _ in range(len(agents))]

    while not all(dones):
        # print("yes")
        # Get Action
        actions = []
        for i in range(len(agents)):
            a = agents[i].get_action(s[i])
            actions.append(a)

        # Take Action --> Can only be taken all at the same time
        s_prime, r, dones, infos = env.step(actions)
        s_prime = [np.reshape(state, [1, 151]) for state in s_prime]

        # Learn only if still alive (not done)
        for i in range(len(agents)):
            if infos[i] == "Dead":
                scores[i] += r[i]
                done_mask = 0.0 if dones[i] else 1.0
                agents[i].train_model(s[i], actions[i], r[i], s_prime[i], dones[i])
            elif not dones[i]:
                scores[i] += r[i]
                done_mask = 0.0 if dones[i] else 1.0
                agents[i].train_model(s[i], actions[i], r[i], s_prime[i], dones[i])

        s = s_prime

        if all(dones):
            # print(dones)
            break

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(len(agents)):
            print("n_episode :{}, score : {:.1f}".format(
                n_epi, scores[i] / print_interval))
        scores = [0.0 for _ in range(len(agents))]


s = env.reset()
s = [np.reshape(state, [1, 151]) for state in s]

while True:
    actions = []
    for i in range(len(agents)):
        a = agents[i].get_action(s[i])
        actions.append(a)
    s, rewards, dones, info = env.step(actions)
    s = [np.reshape(state, [1, 151]) for state in s]

    if not env.render():
        break
