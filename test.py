from ReinLife import tester
from ReinLife.Models import DQN, D3QN, PERD3QN, PPO, PERDQN

# main_brains = [PERDQN(152, 8, load_model="Brains/PERDQN/model_40000_881.pt"),  # <-- CURRENTLY BEST BRAIN!!!
#                DQN(152, load_model="Brains/DQN/model_40000_881.pt"),           # <-- CURRENTLY BEST BRAIN!!!
#                D3QN(152, 8, load_model="Brains/D3QN/model_60000_100.pt")]      # <-- CURRENTLY BEST BRAIN!!!

# main_brains = [PERD3QN(153, 8, load_model="Experiments/Best Brain/PERD3QN/brain_gene_1.pt", training=False)
#                for _ in range(8)]
# main_brains = [PERD3QN(153, 8,
#                        load_model="Experiments/Pretrained Brains/PERD3QN/Static Families/PERD3QN/brain_gene_1.pt",
#                        training=False)
#                for _ in range(3)]
main_brains = [PPO(153, 8, load_model="pretrained/PPO/PPO/brain_gene_0.pt"),
               DQN(153, 8, load_model="pretrained/DQN/DQN/brain_gene_0.pt", training=False),
               D3QN(153, 8, load_model="pretrained/D3QN/D3QN/brain_gene_0.pt", training=False),
               PERD3QN(153, 8, load_model="pretrained/PERD3QN/Static Families/PERD3QN/brain_gene_1.pt", training=False),
               PERDQN(153, 8, load_model="pretrained/PERDQN/PERDQN/brain_gene_1.pt", training=False)]
tester(main_brains, width=30, height=20, max_agents=150, pastel_colors=False, static_families=True,
       limit_reproduction=False, fps=10)
