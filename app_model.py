from TheGame import Environment
from TheGame.Models.ppo import PPOAgent

main_brains = [PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_301.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_302.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_303.pt"),
               PPOAgent(150, 8, load_model="Brains/PPO/model_120000_300.pt")]

env = Environment(width=60, height=30, nr_agents=len(main_brains), grid_size=24, evolution=True, max_agents=30,
                  pastel=False, extended_fov=False, brains=main_brains)
s = env.reset()

while True:
    actions = [main_brains[agent.gen].get_action(s[i]) for i, agent in enumerate(env.agents)]
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=10)