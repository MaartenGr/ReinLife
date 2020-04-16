from TheGame import Environment
from tqdm import tqdm


def trainer(brains, max_epi, print_interval, interactive_results, google_colab, width, height, max_agents=100, render=False,
            families=True, training=True):
    env = Environment(width=width, height=height, evolution=True, max_agents=max_agents, brains=brains, grid_size=24,
                      families=families, print_interval=print_interval, interactive_results=interactive_results,
                      google_colab=google_colab, training=training)
    env.reset()

    for n_epi in tqdm(range(max_epi)):
        for agent in env.agents:
            if agent.brain.method == "PPO":
                agent.action, agent.prob = agent.brain.get_action(agent.state, n_epi)
            else:
                agent.action = agent.brain.get_action(agent.state, n_epi)

        env.step()

        for agent in env.agents:
            agent.learn(n_epi=n_epi)

        env.update_env(n_epi)

        if render:
            env.render(fps=120)

    return env

