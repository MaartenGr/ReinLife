from TheGame import Environment
from TheGame.Models.a2c import A2CAgent
from TheGame.Models.dqn import DQNAgent
from TheGame.Models.perdqn import DQNAgent as PERDQNAgent
from TheGame.Models.drqn import DRQNAgent
from TheGame.Models.ppo import PPOAgent
env = Environment(width=60, height=30, nr_agents=1, grid_size=24, evolution=True, max_agents=50, pastel=False,
                  extended_fov=False)
s = env.reset()

model = "PPO"

if model == "DQN":
    main_brain = DQNAgent(152, load_model="Brains/DQN/model_20000_4.pt")
elif model == "DRQN":
    main_brain = DRQNAgent(152, load_model="Brains/DRQN/model_40000_2.pt")
elif model == "PERDQN":
    main_brain = PERDQNAgent(152, 8, load_model="Brains/PERDQN/model_6000_2.pt")
elif model == "A2C":
    main_brain = A2CAgent(152, 8, load_model="Brains/A2C/model_50000_10.h5")
elif model == "PPO":
    main_brain = PPOAgent(150, 8, load_model="Brains/PPO/model_75000_12.pt")
else:
    main_brain = DQNAgent(152, load_model="Brains/DQN/model_20000_4.pt")

while True:
    actions = [main_brain.get_action(s[i]) for i, agent in enumerate(env.agents)]
    _, rewards, dones, info = env.step(actions)
    s = env.update_env()
    env.render(fps=10)
