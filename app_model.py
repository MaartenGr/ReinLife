from TheGame import Environment
from TheGame.Models.a2c import A2CAgent
from TheGame.Models.dqn import DQNAgent
from TheGame.Models.perdqn import DQNAgent as PERDQNAgent
env = Environment(width=60, height=30, nr_agents=10, grid_size=24, evolution=True, max_agents=30)
s = env.reset()

# # This works
main_brain = A2CAgent(152, 8, load_model="Brains/A2C/model_5000_2.h5")

# main_brain = A2CAgent(152, 8, load_model="Brains/A2C/model_3000_5.h5")
# main_brain = DQNAgent(152, load_model="Brains/DQN/model_20000_4.pt")
# main_brain = PERDQNAgent(152, 8, load_model="Brains/PERDQN/model_6000_2.pt")

while True:
    actions = []
    for i, agent in enumerate(env.agents):
        action = main_brain.get_action(s[i])
        actions.append(action)
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=10)
