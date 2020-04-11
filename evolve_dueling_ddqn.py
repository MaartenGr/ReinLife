from TheGame.Models.dueling_ddqn import DDQNAgent
from TheGame import Environment
from TheGame.utils import Results
from tqdm import tqdm
# Hyperparameters
save_best = True
max_epi = 10_000
learning_rate = 0.0005
epsilon = 0.9
epsilon_min = 0.05
decay = 0.99
soft_update_freq = 100
exploration = 500

track_results = Results(print_interval=500, interactive=True, google_colab=False, nr_gens=2)

# Initialize Env
env = Environment(width=30, height=30, nr_agents=2, evolution=True, fps=20, max_agents=100,
                  brains=[DDQNAgent(152, 8) for _ in range(2)], grid_size=24)
s = env.reset()

for n_epi in tqdm(range(max_epi)):
    if epsilon > epsilon_min:
        epsilon = epsilon * decay

    for agent in env.agents:
        agent.action = env.brains[agent.gen].get_action(agent.state, epsilon)

    env.step()

    # Memorize
    for i, agent in enumerate(env.agents):
        env.brains[agent.gen].memorize(agent.state, agent.action, agent.reward, agent.state_prime, agent.done)

    # Learn
    for agent in env.agents:
        if n_epi > exploration:
            if agent.age % 20 == 0 or agent.dead:
                env.brains[agent.gen].train()

        if n_epi % soft_update_freq == 0:
            env.brains[agent.gen].target_net.load_state_dict(env.brains[agent.gen].eval_net.state_dict())

    track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                 [agent.reward for agent in env.agents])
    s = env.update_env()
    env.render(fps=120)
