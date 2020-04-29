from TheGame.Models import A2C, D3QN, DQN, DRQN, PERD3QN, PERDQN, PPO
from TheGame.Helpers import trainer

n_episodes = 15_000
# brains = [PERD3QN(153, 8),
#           D3QN(153, 8),
#           DQN(153, max_epi=n_episodes, learning_rate=0.0005),
#           A2C(153, 8),
#           PERDQN(153, 8),
#           PPO(153, 8, learning_rate=0.0001),
#           DRQN(learning_rate=1e-3, capacity=10000, epsilon_init=0.9, gamma=0.99, soft_update_freq=100)]

brains = [
          PERD3QN(153, 8, train_freq=5),
          PERD3QN(153, 8, train_freq=5)
          ]


trainer(brains, n_episodes=n_episodes, update_interval=300, width=30, height=30, max_agents=100,
        visualize_results=False, print_results=True, google_colab=False, render=True, static_families=True, training=True, save=True,
        limit_reproduction=False)

# To do:
#       * Requirements
#       * TO DO: Choose genes that are not alive to give them a fighting chance

