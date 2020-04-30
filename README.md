![](images/logo.png)

> Code and instructions for creating Artificial Life in a non-traditional 
way, namely with Reinforcement Learning instead of Evolutionary Algorithms. 

Although Evolutionary Algorithms have shown to result in interesting behavior,
they focus on learning across generations whereas behavior could also be learned
during ones lifetime. This is where Reinforcement Learning comes in, which 
learns through a reward/punishment system that allows it to learn new behavior
during its live time. Using Reinforcement Learning, entities learn to survive, 
reproduce, and make sure to maximize the fitness of their kin. 

![alt-text-1](images/animation_medium.gif "title-1")


<p float="left">
  <img src="https://github.com/MaartenGr/ReinLife/blob/family_new/images/animation_medium.gif?raw=true" height="366"/>
  <img src="https://github.com/MaartenGr/ReinLife/blob/family_new/images/instruction.png?raw=true" height="366"/>
</p>


## Table of Contents  
<!--ts-->
   * [About the Project](#about)
   * [Getting Started](#gettingstarted)
        * [Prerequisites](#prerequisites)
        * [Installation](#installation)
        * [Usage](#usage)
        * [Google Colaboratory](#colab)
   * [Environment](#overview)
   * [Agents](#agents)
        * [Actions](#agents-actions)
        * [Movement](#agents-actions)
        * [Brain](#agents-brain)
   * [Reward](#reward)
   * [Algorithms](#algorithms)
   * [Evolution](#evolution)
   * [Results](#results)
   * [Code](#code)
   100. [To Do](#todo)
<!--te-->

<a name="about"/></a>
## About the Project
The simulation above is a good summary of what this project is about. 
Entities move and learn independently, eat, attack other entities, and reproduce. 

This is all possible by applying Reinforcement Learning algorithms to each
entity, such as 
<a href="https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf">DQN</a> and 
<a href="https://arxiv.org/pdf/1707.06347.pdf">PPO</a>. 

The general principle is simple, each entity starts by randomly executing
some actions and will slowly learn, based on specific rewards, whether 
those actions helped or not. The entity is punished if the action is poor and
rewarded if it was helpful. 

It is then up to the entities to find a way to survive as long as possible while 
also making sure their kin is in good shape as possible. 

<a name="gettingstarted"/></a>
##  Getting Started

<a name="prerequisites"/></a>
###  Prerequisites

To install the requirements, simply run the following:  
```pip install -r requirements.txt```

<a name="usage"/></a>
###  Usage

Due to the many parameters within each model and the environment itself, 
it is advised to start with train.py and test.py and look around there. 

#### Training
To train one or models, simply run:
```python
from ReinLife.Models import PERD3QN
from ReinLife.Helpers import trainer

brains = [PERD3QN(153, 8, train_freq=10), 
          PERD3QN(153, 8, train_freq=10)]

trainer(brains, n_episodes=15_000, update_interval=300, width=30, height=30, max_agents=100,
        visualize_results=True, print_results=False, static_families=False, training=True, save=True)
``` 
This will start training the models for 15_000 episodes. The most important variable
here is `static_families`. If this is set to True, then there will be at most as
many genes as the number of brains chosen. Thus, you will only see two colors.
If you set this to False, then any number of genes will be created each with 
their own brain. 

#### Testing
To test one or models, simply run:
```python
from ReinLife import tester
from ReinLife.Models import DQN, D3QN, PERD3QN, PPO, PERDQN

main_brains = [PPO(153, 8, load_model="Experiments/Pretrained Brains/PPO/PPO/brain_gene_0.pt"),
               DQN(153, 8, load_model="Experiments/Pretrained Brains/DQN/DQN/brain_gene_0.pt", training=False),
               D3QN(153, 8, load_model="Experiments/Pretrained Brains/D3QN/D3QN/brain_gene_0.pt", training=False),
               PERD3QN(153, 8, load_model="Experiments/Pretrained Brains/PERD3QN/Static Families/PERD3QN/brain_gene_1.pt", training=False),
               PERDQN(153, 8, load_model="Experiments/Pretrained Brains/PERDQN/PERDQN/brain_gene_1.pt", training=False)]
tester(main_brains, width=30, height=30, max_agents=100, static_families=True, fps=10)
``` 
The models above are pre-trained (see results below).  
You can choose any number of brains that you have trained previously. Note,
make sure to set all models (except PPO) to training=False, otherwise it will
demonstrate more random behavior.
  
<a name="colab"/></a>
### Google Colaboratory
LINK FOR GOOGLE COLAB TRAINING HERE


<a name="entities"/></a>
##  Entities
In the animation above you can find the following objects:  
![](images/agent_transparent.png) **Agents** can move, attack other entities, eat, and reproduce asexually. Their 
color indicates kinship between others with the same color. 
They have a health of 200 which decreases 10 with each action they take.

![](images/food.png)  **White** blocks are food which increases an entity's health by 40.

![](images/poison.png) **Black** blocks are poison which decreases an entity's health by 40.

![test](images/superfood.png) **Red** blocks are especially helpful as they have the same properties as food but 
also multiplies an entity's maximum age by 1.2.  

The environment is build upon a numpy matrix of size `n` * `m` where
each grid has a pixel size of 24 by 24. Each location within the matrix 
represents a location which can be occupied by only a single entity.

<a name="agents"/></a>
## Agents ![](images/agents_transparent.png)

Agents are entities or organisms in the simulation that can move, attack, 
reproduce, and act independently. 

Each agent has the following characteristics:
* `Health`
    * Starts at 200 and decreases with 10 each step
    * Their health cannot exceed 200 
* `Age`
    * Starts at 0 and increases 1 with each step  
    * Their maximum age is 50, after which they die
* `Gene`
    * Each agents is given a gene, which simply represents an integer
    * All their offspring have the same gene value
    * Any new agent that is created not through reproduction gets a new value
    * This `gene` is represented by the color of the body    
    
An agent can perform one of the following **eight actions**:
* Move one space left, right, up, or down
* Attack in the left, right, up, or down direction 

The **order** of action execution is as follows:
* Attack ➡ Move ➡ Eat ➡ Reproduce

---

<a name="agents-movement"/></a>
### Movement

An agent can occupy any un-occupied space and, from that position, can move up, 
down, left or right. Entities cannot move diagonally. 

The environment has no walls, which means that if an entity moves left from the 
most left position in the numpy matrix, then it will move to the most right 
position. In other words, the environment is a fully-connected world.  

**Complex Movement**  
Although the movement in itself is not complex, it becomes more difficult as 
multiple entities want to move into the same spot. For that reason, each entity 
checks whether the target coordinate is unoccupied and if no other entity wants 
to move in that space. It does this iteratively as the target coordinate changes 
if an entity cannot move.    

---

<a name="agents-attacking"/></a>
### Attacking ![](images/agent_attacking.png)

An agent can attack in one of four directions:
* Up, Down, Left, or Right

They stand still if they attack. However, since it is the first thing they
do, the other agent cannot move away. When the agent successfully attacks 
another agent, the other agent dies and the attacker increases its health. 
Moreover, if the agent successfully attacks another agent, 
its **border** becomes **red**.

---

<a name="agents-observation"/></a>
### Observation

![test](images/observation.png) 

---

<a name="agents-brain"/></a>
#### Brain

The agent learns behavior through Reinforcement Learning. The agent learns continuously
during its lifetime. When the agents dies, the algorithm recognizes it as the end of an 
episode. 

When a new entity is reproduced, it inherits its brain (RL-algorithm) from its parent. 

When a new entity is produced, it inherits its brain (RL-algorithm) from the best
entity we have seen so far. 

---

<a name="reward"/></a>
##  Reward

The reward structure is tricky as you want to minimize the amount you steer
the entity towards certain behavior. For that reason, I've adopted a simple and 
straightforward fitness measure, namely: 

It gets a score of -400 if the entity dies and receives a combined score
each step its alive. That reward is calculated as follows: 

reward = sum([other_agent.health / max_age for other_agent in self.agents if agent.gen == other_agent.gen])

Thus, it simply sums the health of all agents (including itself) at each step. 
The result is that its behavior is steered towards surviving as long as possible
while keeping its kin alive.   

---

<a name="algorithms"/></a>
## Algorithms

Currently, the following algorithms are implemented that can be used as brains:
* DQN
* PER-DQN
* A2C

---

<!---
Font logo: https://www.1001fonts.com/fff-forward-font.html
-->

## Documentation

### Training

| Parameter                        | Description                              | Default value                  |
| :------------------------------- | :--------------------------------------- | :----------------------------- |
| brains | Contains a list of brains defined as Agents by the ReinLife.Models folder. ||
| n_episodes | The number of epsiodes to run the training sequence. | 10_000 |
| width, height | The width and height of the environment. | 30, 30 |
| visualize_results | Whether to visualize the results interactively in matplotlib.  | False |
| google_colab | If you want to visualize your results interactively in google_colab, also set this parameter to True as well as the one above.| False |
| update_interval | The interval at which average the results | 500 |
| print_results | Whether to print the results to the console | True |
| max_agents | The maximum number of agents can occupy the environment. | 100 |
| render | Whether to render the environment in pygame whilst training. | False |
| static_families | Whether you want a set number of families to be used. Each family has its own brain defined by the models in the variable brains. | False|
| training | Whether you want to train using the settings above or simply show the result. | True |
| limit_reproduction | If False, agents can reproduce indefinitely. If True, all agents can only reproduce once. | False |
| incentivize_killing | Whether to incentivize killing by adding 0.2 everytime an agent kills another | True |

### Testing

| Parameter                        | Description                              | Default value                  |
| :------------------------------- | :--------------------------------------- | :----------------------------- |
| brains | Contains a list of brains defined as Agents by the ReinLife.Models folder. ||
| width, height | The width and height of the environment. | 30, 30 |
| pastel_colors | Whether to automatically generate random pastel colors  | False |
| max_agents | The maximum number of agents can occupy the environment. | 100 |
| static_families | Whether you want a set number of families to be used. Each family has its own brain defined by the models in the variable brains. | False|
| limit_reproduction | If False, agents can reproduce indefinitely. If True, all agents can only reproduce once. | False |
| fps | Frames per second | 10 |


# To do:
* List of parameters for train.py and test.py