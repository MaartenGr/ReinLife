from TheGame import Environment
from TheGame.utils import Results
from tqdm import tqdm


def trainer(brains, max_epi, print_interval, interactive, google_colab, width, height, max_agents=100, render=False):
    track_results = Results(print_interval=print_interval, interactive=interactive,
                            google_colab=google_colab, nr_gens=len(brains))
    env = Environment(width=width, height=height, evolution=True, max_agents=max_agents, brains=brains, grid_size=24)
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

        track_results.update_results(env.agents, n_epi, [agent.action for agent in env.agents],
                                     [agent.reward for agent in env.agents])
        env.update_env()

        if render:
            env.render(fps=120)

    return env

