from TheGame import Environment
from TheGame.Models import DQN, D3QN, PERD3QN, DRQN, PPO, PERDQN

# main_brains = [PERDQN(152, 8, load_model="Brains/PERDQN/model_40000_881.pt"),  # <-- CURRENTLY BEST BRAIN!!!
#                DQN(152, load_model="Brains/DQN/model_40000_881.pt"),           # <-- CURRENTLY BEST BRAIN!!!
#                D3QN(152, 8, load_model="Brains/D3QN/model_60000_100.pt")]      # <-- CURRENTLY BEST BRAIN!!!

main_brains = [PERD3QN(153, 8, load_model="Experiments/2020-04-28_V6/PERD3QN/brain_gene_0.pt", training=False),
               PERD3QN(153, 8, load_model="Experiments/2020-04-28_V6/PERD3QN/brain_gene_0.pt", training=False)]

env = Environment(width=30, height=30, grid_size=24, max_agents=150,
                  pastel_colors=False, brains=main_brains, training=False, static_families=True)
s = env.reset()

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
    s = env.update_env()
    env.render(fps=10)