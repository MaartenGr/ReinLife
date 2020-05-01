from ReinLife import tester
from ReinLife.Models import DQN, D3QN, PERD3QN, PPO, PERDQN

main_brains = [PPO(load_model="pretrained/PPO/PPO/brain_gene_0.pt"),
               DQN(load_model="pretrained/DQN/DQN/brain_gene_0.pt", training=False),
               D3QN(load_model="pretrained/D3QN/D3QN/brain_gene_0.pt", training=False),
               PERD3QN(load_model="pretrained/PERD3QN/Static Families/PERD3QN/brain_gene_1.pt", training=False),
               PERDQN(load_model="pretrained/PERDQN/PERDQN/brain_gene_1.pt", training=False)]
tester(main_brains, width=30, height=20, max_agents=150, pastel_colors=False, static_families=True,
       limit_reproduction=False, fps=10)
