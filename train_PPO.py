from TheGame import Environment
from TheGame.Models.PPO import PPOAgent
from TheGame.utils import Results

max_epi = 65_000
track_results = Results(print_interval=500, interactive=True, google_colab=False, nr_gens=2)
env = Environment(width=30, height=30, max_agents=100, nr_agents=2, evolution=True, fps=20,
                  brains=[PPOAgent(152, 8, learning_rate=0.0001) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in range(max_epi):

    for i, agent in enumerate(env.agents):
        agent.action, agent.prob = agent.brain.get_action(agent.state)

    env.step()

    for i, agent in enumerate(env.agents):
        agent.learn(n_epi=n_epi)

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    env.update_env()
