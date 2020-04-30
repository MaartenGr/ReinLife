from ReinLife.Models import D3QN, DQN, PERD3QN, PERDQN, PPO
from ReinLife.Helpers import trainer

n_episodes = 15_000

brains = [D3QN(153, 8, train_freq=10),
          D3QN(153, 8, train_freq=10)]

trainer(brains, n_episodes=n_episodes, update_interval=300, width=30, height=30, max_agents=100,
        visualize_results=True, print_results=False, google_colab=False, render=False, static_families=True,
        training=True, save=True, limit_reproduction=False, incentivize_killing=False)

# To do:
#       * TO DO: Choose genes that are not alive to give them a fighting chance

