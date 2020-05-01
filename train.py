from ReinLife.Models import D3QN, DQN, PERD3QN, PERDQN, PPO
from ReinLife.Helpers import trainer

n_episodes = 15_000
brains = [DQN(train_freq=20, max_epi=n_episodes),
          D3QN(train_freq=20),
          PERDQN(train_freq=20),
          PERD3QN(train_freq=20),
          PPO(train_freq=20)]

trainer(brains, n_episodes=n_episodes, update_interval=300, width=30, height=30, max_agents=100,
        visualize_results=True, print_results=False, google_colab=False, render=False, static_families=True,
        training=True, save=True, limit_reproduction=False, incentivize_killing=True)

# To do:
#       * Choose genes that are not alive to give them a fighting chance
#       * Update tester.py such that it does not include the button press anymore
