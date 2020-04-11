from TheGame.Models.ddqn import DDQNAgent
from TheGame import Environment
from TheGame.utils import Results
from tqdm import tqdm
# Hyperparameters
save_best = True
max_epi = 10_000
learning_rate = 0.0005
track_results = Results(print_interval=500, interactive=True, google_colab=False, nr_gens=2)

# Initialize Env
env = Environment(width=30, height=30, nr_agents=2, evolution=True, fps=20, max_agents=100,
                  brains=[DDQNAgent(152, 8) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in tqdm(range(max_epi)):
    for agent in env.agents:
        agent.action = env.brains[agent.gen].get_action(agent.state)

    env.step()

    # Memorize
    for i, agent in enumerate(env.agents):
        env.brains[agent.gen].memorize(agent.state, agent.action, agent.reward, agent.state_prime, agent.done)

    # Learn
    for agent in env.agents:

        if agent.age % 20 == 0 or agent.dead:
            if env.brains[agent.gen].buffer.size() >= 32:
                env.brains[agent.gen].replay()
            env.brains[agent.gen].target_update()

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()

    # env.render()

if save_best:
    env.save_best_brain(max_epi)

while True:
    for agent in env.agents:
        agent.action = env.brains[agent.gen].get_action(agent.state, 0)
    env.step()
    s = env.update_env()
    env.render(fps=10)
