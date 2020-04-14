from TheGame.Models.PERDQN import PERDQNAgent
from TheGame import Environment
from TheGame.utils import Results
from tqdm import tqdm

save_best = True
max_epi = 10_000
track_results = Results(print_interval=500, interactive=True, google_colab=False, nr_gens=2)

# Init env
env = Environment(width=30, height=30, nr_agents=2, evolution=True, fps=20, max_agents=100,
                  brains=[PERDQNAgent(152, 8) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in tqdm(range(max_epi)):
    for agent in env.agents:
        agent.action = agent.brain.get_action(agent.state)

    env.step()

    for i, agent in enumerate(env.agents):
        agent.learn(n_epi=n_epi)

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()
    # env.render(fps=120)
