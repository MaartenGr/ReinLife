# https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

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

q = Qnet()
q_target = Qnet()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

print_interval = 20
score = 0.0
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

for n_epi in range(1000):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    s = env.reset()
    s = s[0].flatten()
    done = False

    while not done:
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        s_prime, r, done, info = env.step([a])
        s_prime = s_prime[0].flatten()
        r = r[0]
        done = done[0]
        done_mask = 0.0 if done else 1.0
        # print(s, a, r, s_prime, done_mask)
        # print(f"s: {s}")
        # print(f"a: {a}")
        # print(f"r: {r}")
        # print(f"s'': {s_prime}")
        # print(f"d: {done_mask}")

        break
        memory.put((s, a, r / 100.0, s_prime, done_mask))
        s = s_prime

        score += r
        if done:
            break

    if memory.size() > 2000:
        train(q, q_target, memory, optimizer)

    if n_epi % print_interval == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            n_epi, score / print_interval, memory.size(), epsilon * 100))
        score = 0.0


s = env.reset()
s = s[0].flatten()
while True:
    action = q.sample_action(torch.from_numpy(s).float(), epsilon)
    s, rewards, dones, info = env.step([action])
    s = s[0].flatten()
    if not env.render():
        break
