# To do:
#   * Make sure to change the dones such that it does not update the memory or execute an action after its done
#   * Permanently remove an agent if dead (or prevent any actions) --> preferably removed from env
#   * Perhaps move through walls? --> Fixes the stupid bug and creates a "round" world
#
#
#
# Technically, a dead agent is still alive seeing as it has an x, y coordinate and has the possibility to eat stuff...


import torch.optim as optim
import torch

from Field.dqn import ReplayBuffer, Qnet, train
from Field import MultiEnvironment as Environment

#Hyperparameters
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

env = Environment()

agents = [Qnet() for _ in range(20)]
agents_targets = [Qnet() for _ in range(20)]

for index, target in enumerate(agents_targets):
    target.load_state_dict(agents[index].state_dict())

memories = [ReplayBuffer() for _ in range(20)]
optimizers = [optim.Adam(agents[index].parameters(), lr=learning_rate) for index in range(20)]

print_interval = 10
scores = [0.0 for _ in range(len(agents))]


for n_epi in range(150):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    s = env.reset()
    dones = [False for _ in range(20)]

    while not all(dones):

        actions = []
        for i in range(len(agents)):
            a = agents[i].sample_action(torch.from_numpy(s[i].flatten()).float(), epsilon)
            actions.append(a)

        # print(actions)
        s_prime, r, dones, info = env.step(actions)

        for i in range(len(agents)):
            if not dones[i]:
                scores[i] += r[i]
                done_mask = 0.0 if dones[i] else 1.0
                memories[i].put((s[i].flatten(), actions[i], r[i] / 200.0, s_prime[i].flatten(), done_mask))
        s = s_prime

        if all(dones):
            # print(dones)
            break

    for i in range(len(agents)):
        if memories[i].size() > 2000:
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
        action = agents[i].sample_action(torch.from_numpy(s[i].flatten()).float(), epsilon)
        actions.append(action)
    s, rewards, dones, info = env.step(actions)

    if not env.render():
        break
