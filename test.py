from TheGame import tester
from TheGame.Models import DQN, D3QN, PERD3QN, DRQN, PPO, PERDQN

# main_brains = [PERDQN(152, 8, load_model="Brains/PERDQN/model_40000_881.pt"),  # <-- CURRENTLY BEST BRAIN!!!
#                DQN(152, load_model="Brains/DQN/model_40000_881.pt"),           # <-- CURRENTLY BEST BRAIN!!!
#                D3QN(152, 8, load_model="Brains/D3QN/model_60000_100.pt")]      # <-- CURRENTLY BEST BRAIN!!!

main_brains = [PERD3QN(153, 8, load_model="Experiments/Best Brain/PERD3QN/brain_gene_1.pt", training=False)
               for _ in range(8)]

tester(main_brains, width=30, height=30, max_agents=150, pastel_colors=False, static_families=True,
       limit_reproduction=False, fps=10)
