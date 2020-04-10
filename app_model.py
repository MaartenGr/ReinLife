from TheGame import Environment
from TheGame.Models.ppo import PPOAgent
from TheGame.Models.dqn import DQNAgent
from TheGame.Models.perdqn import DQNAgent as PERDQNAgent

main_brains = [PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_301.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_302.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_303.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt")]

# main_brains = [DQNAgent(151, load_model="Brains/DQN/model_30000_100.pt"),
#                DQNAgent(151, load_model="Brains/DQN/model_30000_101.pt")]

main_brains = [DQNAgent(152, load_model="Brains/DQN/model_40000_881.pt"),
               DQNAgent(152, load_model="Brains/DQN/model_40000_881.pt")]

env = Environment(width=30, height=30, nr_agents=len(main_brains), grid_size=24, evolution=True, max_agents=150,
                  pastel=False, extended_fov=False, brains=main_brains)
s = env.reset()

while True:
    for agent in env.agents:
        if env.brains[agent.gen].method == "DQN":
            agent.action = env.brains[agent.gen].get_action(agent.state, 0)
        elif env.brains[agent.gen].method in ["PPO", "PERDQN"]:
            agent.action = env.brains[agent.gen].get_action(agent.state)
    env.step()
    s = env.update_env()
    env.render(fps=10)