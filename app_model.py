from TheGame import Environment
from TheGame.Models.ppo import PPOAgent
from TheGame.Models.dqn import DQNAgent

main_brains = [PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_301.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_302.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_303.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt")]

main_brains = [DQNAgent(150, load_model="Brains/DQN/model_50000_990.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_50000_991.pt")]
main_brains = [DQNAgent(150, load_model="Brains/DQN/model_50000_990.pt"),
               DQNAgent(150, load_model="Brains/DQN/model_50000_990.pt")]
env = Environment(width=60, height=30, nr_agents=len(main_brains), grid_size=24, evolution=True, max_agents=30,
                  pastel=False, extended_fov=False, brains=main_brains)
s = env.reset()

while True:
    for agent in env.agents:
        if env.brains[agent.gen].method == "DQN":
            agent.action = env.brains[agent.gen].get_action(agent.state, 0)
        elif env.brains[agent.gen].method == "PPO":
            agent.action = env.brains[agent.gen].get_action(agent.state)
    env.step()
    s = env.update_env()
    env.render(fps=10)