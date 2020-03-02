# To do:
#   * Make sure to change the dones such that it does not update the memory or execute an action after its done
#   * Permanently remove an agent if dead (or prevent any actions) --> preferably removed from env
#   * Add 'deep neuroevolution'
#       * https://github.com/paraschopra/deepneuroevolution/blob/master/openai-gym-cartpole-neuroevolution.ipynb
#   * Only update coordinates after everyone has attacked
#   * Separate self.entities["Food"] by poison and apple
#
# Technically, a dead agent is still alive seeing as it has an x, y coordinate and has the possibility to eat stuff...

import numpy as np
from Field import A2CAgent
from Field.MultiEnvironment import MultiEnvironment as Environment

env = Environment()
agents = [A2CAgent((2*7*7)+3, 8) for _ in range(20)]
print_interval = 1
scores = [0.0 for _ in range(len(agents))]

for n_epi in range(100):
    print(n_epi)
    s = env.reset()
    dones = [False for _ in range(20)]

    while not all(dones):
        # Get Action
        actions = []
        for i in range(len(agents)):
            a = agents[i].get_action(np.array(s[i]).reshape(1, -1))
            actions.append(a)

        # Take Action --> Can only be taken all at the same time
        s_prime, r, dones, info = env.step(actions)

        # Learn only if still alive (not done)
        for i in range(len(agents)):
            if not dones[i]:
                scores[i] += r[i]
                agents[i].train_model(np.array(s[i]).reshape(1, -1), actions[i], r[i],
                                      np.array(s_prime[i]).reshape(1, -1), dones[i])

        s = s_prime

        if all(dones):
            # print(dones)
            break

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(len(agents)):
            print("n_episode :{}, score : {:.1f}".format(n_epi, scores[i]))
        scores = [0.0 for _ in range(len(agents))]

s = env.reset(is_render=True)
while True:
    actions = []
    for i in range(len(agents)):
        action = agents[i].get_action(np.array(s[i]).reshape(1, -1))
        actions.append(action)
    s, rewards, dones, info = env.step(actions)

    if not env.render():
        break
