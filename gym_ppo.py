from TheGame.Models.ppo_no_lstm import PPO
from TheGame import Environment
import numpy as np

env = Environment(width=10, height=10, nr_agents=1, fps=5)
env.max_step = 30

agents = [PPO() for _ in range(1)]


print_interval = 10
scores = [0.0 for _ in range(len(agents))]
t = 0
for n_epi in range(2_000):

    # Initialize Variables
    s = env.reset()
    s = [np.array(state, dtype=np.float32	) for state in s]

    dones = [False for _ in range(len(agents))]

    while not all(dones):

        # print("yes")
        # Get Action
        actions = []
        probs = []
        for i in range(len(agents)):
            prob = agents[i].brain.pi(torch.from_numpy(s[i]).float())
            m = Categorical(prob)
            action = m.sample().item()
            a.append(action)
            probs.append(prob)

        # Take Action --> Can only be taken all at the same time
        s_prime, r, dones, infos = env.step(actions)
        s_prime = [np.array(state, dtype=np.float32	) for state in s_prime]

        # Learn only if still alive (not done)
        for i in range(len(agents)):
            scores[i] += r[i]
            agents[i].memory.rewards.append(r[i])
            agents[i].memory.is_terminals.append(dones[i])
        s = s_prime

        # update if its time
        for i in range(len(agents)):
            if t % update_timestep == 0 and t != 0:
                agents[i].update(agents[i].memory)
                agents[i].memory.clear_memory()

                t = 0

        if all(dones):
            # print(dones)
            break

        t += 1

    if n_epi % print_interval == 0 and n_epi != 0:
        for i in range(len(agents)):
            print("n_episode :{}, score : {:.1f}".format(
                n_epi, scores[i] / print_interval))
        scores = [0.0 for _ in range(len(agents))]


s = env.reset()
s = [np.array(state, dtype=np.float32) for state in s]

while True:
    actions = []
    for i in range(len(agents)):
        a = agents[i].policy_old.act(s[i], agents[i].memory)
        actions.append(a)
    s, rewards, dones, info = env.step(actions)
    s = [np.array(state, dtype=np.float32	) for state in s]

    if not env.render():
        break
