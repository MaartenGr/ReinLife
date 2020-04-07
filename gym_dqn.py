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

from TheGame.Models.dqn import ReplayBuffer, Qnet, train
from TheGame import Environment

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

env = Environment(width=10, height=10, nr_agents=1)
env.max_step = 30

agents = [Qnet(152) for _ in range(1)]
agents_targets = [Qnet(152) for _ in range(len(agents))]

for index, target in enumerate(agents_targets):
    target.load_state_dict(agents[index].state_dict())

memories = [ReplayBuffer() for _ in range(len(agents))]
optimizers = [optim.Adam(agents[index].parameters(), lr=learning_rate) for index in range(len(agents))]

print_interval = 10
scores = [0.0 for _ in range(len(agents))]

for n_epi in range(500):

    # Initialize Variables
    epsilon = max(0.01, 0.08 - 0.08 * (n_epi / 500))  # Linear annealing from 8% to 1%
    s = env.reset()
    dones = [False for _ in range(len(agents))]

    while not all(dones):
        # print("yes")
        # Get Action
        actions = []
        for i in range(len(agents)):
            a = agents[i].sample_action(s[i], epsilon)
            actions.append(a)

        # Take Action --> Can only be taken all at the same time
        s_prime, r, dones, infos = env.step(actions)

        # Learn only if still alive (not done)
        for i in range(len(agents)):
            if infos[i] == "Dead":
                scores[i] += r[i]
                done_mask = 0.0 if dones[i] else 1.0
                memories[i].put((s[i], actions[i], r[i] / 200.0, s_prime[i], done_mask))
            elif not dones[i]:
                scores[i] += r[i]
                done_mask = 0.0 if dones[i] else 1.0
                memories[i].put((s[i], actions[i], r[i] / 200.0, s_prime[i], done_mask))
        s = s_prime

        if all(dones):
            # print(dones)
            break

    for i in range(len(agents)):
        if memories[i].size() > 1000:
            train(agents[i], agents_targets[i], memories[i], optimizers[i])

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(len(agents)):
            agents_targets[i].load_state_dict(agents[i].state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, scores[i] / print_interval, memories[i].size(), epsilon * 100))
        scores = [0.0 for _ in range(len(agents))]


s = env.reset()
while True:
    actions = []
    for i in range(len(agents)):
        action = agents[i].sample_action(s[i], epsilon)
        actions.append(action)
    s, rewards, dones, info = env.step(actions)

    if not env.render():
        break