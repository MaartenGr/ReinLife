from TheGame.Models.rainbow import DQNAgent
from TheGame import Environment
from TheGame.utils import Results
from tqdm import tqdm

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100
max_epi = 30_000
track_results = Results(print_interval=500, interactive=True, google_colab=False, nr_gens=2)

# Initialize Env
env = Environment(width=30, height=30, nr_agents=2, evolution=True, fps=20, max_agents=100,
                  brains=[DQNAgent(152, 8, memory_size, batch_size, target_update, n_step=0) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in tqdm(range(max_epi)):

    for agent in env.agents:
        agent.action = env.brains[agent.gen].get_action(agent.state)

    env.step()

    # Memorize
    for i, agent in enumerate(env.agents):
        env.brains[agent.gen].memorize(agent.reward, agent.state_prime, agent.done)

        if agent.age % 10 == 0 or agent.dead:
            env.brains[agent.gen].train(n_epi, max_epi)

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()
