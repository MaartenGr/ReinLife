from TheGame.Models import A2C, D3QN, DRQN, DQN, PERDQN, PERD3QN, PPO
from TheGame.Trainer import trainer

max_epi = 10_000
brains = [PERD3QN(152, 8),
          D3QN(152, 8),
          DQN(152, max_epi=max_epi, learning_rate=0.0005),
          A2C(152, 8),
          PERDQN(152, 8),
          PPO(152, 8, learning_rate=0.0001),
          DRQN(learning_rate=1e-3, capacity=10000, epsilon_init=0.9, gamma=0.99, soft_update_freq=100)]

trainer(brains, max_epi=max_epi, print_interval=500, width=30, height=30, max_agents=100,
        interactive=True, google_colab=False, render=False)
