from TheGame.World.environment import Environment
from tqdm import tqdm


def trainer(brains, n_episodes=10_000, width=30, height=30, visualize_results=False, google_colab=False,
            update_interval=500, print_results=True, max_agents=100, render=False, static_families=True, training=True, save=True,
            limit_reproduction=False, incentivize_killing=True):
    """ Automatically trains the models for n_episodes

    Parameters:
    -----------
    brains : list
        Contains a list of brains defined as Agents by the TheGame.Models folder.

    n_episodes : int, default 10_000
        The number of epsiodes to run the training sequence.

    width : int, default 30
        The width of the environment.

    height : int, default 30
        The height of the environment.

    visualize_results : boolean, default False
        Whether to visualize the results interactively in matplotlib.

    google_colab : boolean, default False
        If you want to visualize your results interactively in google_colab, also set this parameter
        to True as well as the one above.

    update_interval : int, default 500
        The interval at which average the results

    print_results : bool, default True
        Whether to print the results to the console

    max_agents : int, default 100
        The maximum number of agents can occupy the environment.

    render : boolean, default False
        Whether to render the environment in pygame whilst training.

    families: boolean, default False
        Whether you want a set number of families to be used. Each family has its own brain defined
        by the models in the variable brains. The number of families cannot exceed the number of models
        in the brains variable.

    training : boolean, default True
        Whether you want to train using the settings above or simply show the result.

    limit_reproduction : bool, default False
        If False, agents can reproduce indefinitely. If True, all agents can only reproduce once.

    incentivize_killing : bool, default True
        Whether to incentivize killing by adding 0.2 everytime an agent kills another

    Returns:
    --------
    env : gym.env
        The resulting environment in which the status of the last episode is saved. Using this, you can save
        the brains manually by accessing env.brains or env.best_agents.

    """

    env = Environment(width=width, height=height, max_agents=max_agents, brains=brains, grid_size=24,
                      static_families=static_families, update_interval=update_interval, print_results=print_results,
                      interactive_results=visualize_results, google_colab=google_colab, training=training,
                      limit_reproduction=limit_reproduction, incentivize_killing=incentivize_killing)
    env.reset()

    for n_epi in tqdm(range(n_episodes+1)):

        # Get actions for each brain
        for agent in env.agents:
            agent.get_action(n_epi)

        # Execute all actions
        env.step()

        # Learn based on the result
        for agent in env.agents:
            agent.learn(n_epi=n_epi)

        # Clean up environment, remove dead agents, reproduce, etc.
        env.update_env(n_epi)

        if render:
            env.render(fps=120)

    if save:
        env.save_results()

    return env

