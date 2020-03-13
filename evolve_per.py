from TheGame.Models.perdqn import DQNAgent
from TheGame import Environment

# Hyperparameters
learning_rate = 0.0005
nr_agents = 1
best_score = -10_000
print_interval = 10
scores = [0.0 for _ in range(nr_agents)]

# Init env
brains = [DQNAgent(102, 8) for _ in range(nr_agents)]
env = Environment(width=20, height=20, nr_agents=nr_agents, evolution=True, fps=20, brains=brains, grid_size=24)
env.max_step = 30_000
s = env.reset()

for n_epi in range(2_000):
    actions = [agent.brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    s_prime, r, dones, infos = env.step(actions)

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        if agent.fitness > best_score:
            best_score = agent.fitness

        # Memorize & Learn
        agent.brain.append_sample(s[i], actions[i], r[i], s_prime[i], dones[i])
        if agent.brain.memory.tree.n_entries >= agent.brain.train_start:
            agent.brain.train_model()

    # Update model
    for agent in env.agents:
        if agent.age % 20 == 0 or agent.dead:
            agent.brain.update_target_model()

    s = env.update_env()

    if n_epi % 100 == 0:
        print(f"Best score: {best_score}. Nr episodes: {n_epi}. Nr_agents: {len(env.agents)}")

    # env.render()

# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = agent.brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()

    if not env.render():
        break
