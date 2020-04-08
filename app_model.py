from TheGame import Environment
from TheGame.Models.ppo import PPOAgent
from TheGame.Models.dqn import DQNAgent

main_brains = [PPOAgent(151, 8, load_model="Brains/PPO/model_90000_300.pt"),
               PPOAgent(151, 8, load_model="Brains/PPO/model_90000_301.pt"),
               PPOAgent(151, 8, load_model="Brains/PPO/model_90000_302.pt"),
               PPOAgent(151, 8, load_model="Brains/PPO/model_90000_303.pt"),
               PPOAgent(151, 8, load_model="Brains/PPO/model_90000_304.pt")]

main_brains = [DQNAgent(151, load_model="Brains/DQN/model_30000_400.pt"),
               DQNAgent(151, load_model="Brains/DQN/model_30000_401.pt")]

env = Environment(width=30, height=30, nr_agents=len(main_brains), grid_size=24, evolution=True, max_agents=30,
                  pastel=False, extended_fov=False, brains=main_brains)
s = env.reset()

while True:
    actions = [main_brains[agent.gen].get_action(s[i]) for i, agent in enumerate(env.agents)]
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=10)