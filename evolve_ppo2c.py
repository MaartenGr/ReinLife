from TheGame.Models.ppo2c import PPO
from TheGame import Environment
import numpy as np

# Hyperparameters
state_dim = 152
action_dim = 8
render = False
solved_reward = 230  # stop training if avg_reward > solved_reward
log_interval = 20  # print avg reward in the interval
max_episodes = 50000  # max training episodes
n_latent_var = 64  # number of variables in hidden layer
update_timestep = 1000  # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = None
nr_agents = 1
best_agent = 0
best_score = -10_000
print_interval = 1000
scores = [0.0 for _ in range(nr_agents)]

# Init env
brains = [PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip) for _ in range(nr_agents)]
env = Environment(width=20, height=20, nr_agents=nr_agents, evolution=True, fps=20, brains=brains, grid_size=24)
env.max_step = 30_000
s = env.reset()

good = 0
error = 0
t = 0
for n_epi in range(30_000):
    # actions = [agent.brain.policy_old.act(s[i], agent.brain.memory) for i, agent in enumerate(env.agents)]

    actions = []
    for i, agent in enumerate(env.agents):
        (action, agent.brain.states,
         agent.brain.actions, agent.brain.logprobs) = agent.brain.policy_old.act(s[i],
                                                                                 agent.brain.states,
                                                                                 agent.brain.actions,
                                                                                 agent.brain.logprobs)
        actions.append(action)

    s_prime, r, dones, infos = env.step(actions)

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        if agent.fitness > best_score:
            best_score = agent.fitness

        agent.brain.rewards.append(r[i])
        agent.brain.is_terminals.append(dones[i])

    ################################
    #  ONLY UPDATES ONE AGENT....  #
    ################################
    for i, agent in enumerate(env.agents):
        if t % update_timestep == 0 and t != 0:
        # if agent.dead and agent.age > 5:
            # print(i, len(agent.brain.memory.states))
            agent.brain.update()
            agent.brain.clear_memory()
            good += 1

    if t % update_timestep == 0 and t != 0:
        t = 0

    s = env.update_env()

    if n_epi % print_interval == 0:
        print(f"Best score: {best_score}. Nr episodes: {n_epi}. Nr_agents: {len(env.agents)}. Good: {good}. Errors"
              f": {error}" )
    t += 1
    # env.render()

# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        (action, agent.brain.states,
         agent.brain.actions, agent.brain.logprobs) = agent.brain.policy_old.act(s[i],
                                                                                 agent.brain.states,
                                                                                 agent.brain.actions,
                                                                                 agent.brain.logprobs)
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()


    if not env.render():
        break
