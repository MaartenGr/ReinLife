from TheGame import Environment
from TheGame.Models.ppo import PPOAgent
from TheGame.utils import Results

# Hyperparameters
max_epi = 65_000
save_best = True
track_results = Results(print_interval=500, interactive=True, google_colab=False)
env = Environment(width=30, height=30, max_agents=50, nr_agents=2, evolution=True, fps=20,
                  brains=[PPOAgent(152, 8, learning_rate=0.0001) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):

    for i, agent in enumerate(env.agents):
        agent.action, agent.prob = agent.brain.get_action(agent.state)

    env.step()

    # Learn only if still alive (not done)
    for i, agent in enumerate(env.agents):
        env.brains[agent.gen].put_data((agent.state, agent.action, agent.reward / 100.0,
                                        agent.state_prime, agent.prob[agent.action].item(), agent.done))

        if agent.age % 20 == 0 or agent.dead:
            if env.brains[agent.gen].data():
                env.brains[agent.gen].learn()

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    env.update_env()

if save_best:
    env.save_best_brain(max_epi)

# s = env.reset()
while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action, _ = agent.brain.get_action(s[i])
        actions.append(action)
    env.step()
    s = env.update_env()

    if not env.render():
        break
