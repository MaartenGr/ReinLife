from typing import List
from ReinLife.World.environment import Environment
from ReinLife.Models.utils import BasicBrain


def tester(brains: List[BasicBrain],
           width: int = 30,
           height: int = 30,
           max_agents: int = 100,
           pastel_colors: bool = False,
           static_families: bool = True,
           limit_reproduction: bool = False,
           fps: int = 10):
    """ Used to run your brains and see how they act and/or interact

    Parameters:
    -----------
    brains : list
        Contains a list of brains defined as Agents by the TheGame.Models folder.

    width : int, default 30
        The width of the environment.

    height : int, default 30
        The height of the environment.

    max_agents : int, default 100
        The maximum number of agents can occupy the environment.

    pastel_colors : bool, default False
        Whether to automatically generate random pastel colors

    static_families: boolean, default True
        Whether you want a set number of families to be used. Each family has its own brain defined
        by the models in the variable brains. The number of families cannot exceed the number of models
        in the brains variable.

    limit_reproduction : bool, default False
        If False, agents can reproduce indefinitely. If True, all agents can only reproduce once.

    fps : int, default 10
        Frames per second of the render

    Returns:
    --------
    env : gym.env
        The resulting environment in which the status of the last episode is saved. Using this, you can save
        the brains manually by accessing env.brains or env.best_agents.

    """
    env = Environment(width=width, height=height, grid_size=24, max_agents=max_agents,
                      pastel_colors=pastel_colors, brains=brains, training=False, static_families=static_families,
                      limit_reproduction=limit_reproduction)
    env.reset()
    env.render(fps=fps)

    while True:
        for agent in env.agents:
            if env.static_families:
                if env.brains[agent.gene].method in ["DQN", "D3QN", "PERD3QN"]:
                    agent.action = agent.brain.get_action(agent.state, 0)
                elif env.brains[agent.gene].method in ["PPO", "PERDQN"]:
                    agent.action = agent.brain.get_action(agent.state)
            else:
                if agent.brain.method in ["DQN", "D3QN", "PERD3QN"]:
                    agent.action = agent.brain.get_action(agent.state, 0)
                elif agent.brain.method in ["PPO", "PERDQN"]:
                    agent.action = agent.brain.get_action(agent.state)

        env.step()
        env.update_env()
        env.render(fps=fps)
