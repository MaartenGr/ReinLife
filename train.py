from TheGame.Models import A2C, D3QN, DRQN, DQN, PERDQN, PERD3QN, PPO
from TheGame.Trainer import trainer

n_episodes = 10_000
brains = [PERD3QN(152, 8),
          D3QN(152, 8),
          DQN(152, max_epi=n_episodes, learning_rate=0.0005),
          A2C(152, 8),
          PERDQN(152, 8),
          PPO(152, 8, learning_rate=0.0001),
          DRQN(learning_rate=1e-3, capacity=10000, epsilon_init=0.9, gamma=0.99, soft_update_freq=100)]

brains = [PERD3QN(153, 8),
          PERD3QN(153, 8),
          PERD3QN(153, 8)]

trainer(brains, n_episodes=n_episodes, print_interval=500, width=30, height=30, max_agents=100,
        visualize_results=True, google_colab=False, render=False, families=True, training=True)

# To do:
#       * Save the resulting brains in its own folder labelling the experiment
