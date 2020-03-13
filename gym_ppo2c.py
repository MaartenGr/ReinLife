from TheGame.Models.ppo2c import PPO
from TheGame import Environment
import numpy as np

############## Hyperparameters ##############
state_dim = 102
action_dim = 8
render = False
solved_reward = 230  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # max training episodes
max_timesteps = 300  # max timesteps in one episode
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 2000  # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = None
#############################################

env = Environment(width=10, height=10, nr_agents=2, fps=5)
env.max_step = 30

agents = [PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip) for _ in range(2)]

print_interval = 10
scores = [0.0 for _ in range(len(agents))]
t = 0
for n_epi in range(2_000):

    # Initialize Variables
    s = env.reset()
    s = [np.array(state, dtype=np.float32	) for state in s]

    dones = [False for _ in range(len(agents))]

    while not all(dones):

        actions = []
        for i, agent in enumerate(env.agents):
            (action, agents[i].states,
             agents[i].actions, agents[i].logprobs) = agents[i].policy_old.act(s[i],
                                                                           agents[i].states,
                                                                           agents[i].actions, agents[i].logprobs)
            actions.append(action)

        # Take Action --> Can only be taken all at the same time
        s_prime, r, dones, infos = env.step(actions)
        s_prime = [np.array(state, dtype=np.float32	) for state in s_prime]

        # Learn only if still alive (not done)
        for i in range(len(agents)):
            scores[i] += r[i]

            agents[i].rewards.append(r[i])
            agents[i].is_terminals.append(dones[i])
        s = s_prime

        # update if its time
        for i in range(len(agents)):
            if t % update_timestep == 0 and t != 0:
                agents[i].update()
                agents[i].clear_memory()

        if t % update_timestep == 0 and t != 0:
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
