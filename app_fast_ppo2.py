# To do:
#   * Make sure to change the dones such that it does not update the memory or execute an action after its done
#   * Permanently remove an agent if dead (or prevent any actions) --> preferably removed from env
#   * Add 'deep neuroevolution'
#       * https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
#   * Only update coordinates after everyone has attacked
#   * Separate self.entities["Food"] by poison and apple --> OR implement gridworld which is faster
#   * Look into https://github.com/xuehy/pytorch-maddpg
#   * Multi-agent lib --> https://bair.berkeley.edu/blog/2018/12/12/rllib/
# Technically, a dead agent is still alive seeing as it has an x, y coordinate and has the possibility to eat stuff...


import numpy as np

from Field import PPO
from Field import GridWorld as Environment
from torch.distributions import Categorical
import torch


T_horizon = 5

agents = [PPO(101) for _ in range(5)]


env = Environment(width=20, height=20, nr_agents=5)
print_interval = 30
scores = [0.0 for _ in range(len(agents))]

# logging variables
running_reward = 0
avg_length = 0


for n_epi in range(10_000):
    s = env.reset()
    dones = [False for _ in range(len(agents))]
    h_outs = [(torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float)) for _ in
              range(len(agents))]
    h_ins = [False for _ in range(len(agents))]

    while not all(dones):
        for t in range(T_horizon):
            infos = [False for _ in range(len(agents))]
            # Get Action
            actions = [False for _ in range(len(agents))]
            probs = [False for _ in range(len(agents))]
            for i, agent in enumerate(agents):
                if infos[i] == "Dead" or not dones[i]:
                    h_ins[i] = h_outs[i]
                    prob, h_out = agent.pi(torch.from_numpy(s[i].flatten()).float(), h_ins[i])
                    h_outs[i] = h_out
                    prob = prob.view(-1)
                    probs[i] = prob
                    m = Categorical(prob)
                    a = m.sample().item()
                    actions[i] = a

            # Take Action --> Can only be taken all at the same time
            s_prime, r, dones, infos = env.step(actions)

            # Learn only if still alive (not done)
            for i in range(len(agents)):
                if infos[i] == "Dead" or not dones[i]:
                    scores[i] += r[i]
                    agents[i].put_data((s[i].flatten(), actions[i], r[i]/100.0, s_prime[i].flatten(),
                                        probs[i][actions[i]].item(),
                                        h_ins[i], h_outs[i], dones[i]))


            s = s_prime

            if all(dones):
                break
        for i, agent in enumerate(agents):
            if not dones[i]:

                agent.train_net()

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(len(agents)):
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, scores[i] / print_interval))
        scores = [0.0 for _ in range(len(agents))]

s = env.reset(is_render=True)
h_outs = [(torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float)) for _ in
          range(len(agents))]
h_ins = [False for _ in range(len(agents))]
while True:
    actions = []
    for i in range(len(agents)):
        h_ins[i] = h_outs[i]
        prob, h_out = agents[i].pi(torch.from_numpy(s[i].flatten()).float(), h_ins[i])
        h_outs[i] = h_out
        prob = prob.view(-1)
        m = Categorical(prob)
        a = m.sample().item()
        actions.append(a)
    s, rewards, dones, info = env.step(actions)

    if not env.render():
        break
