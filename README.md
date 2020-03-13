# The Game

The purpose of this repository is to explore different methods for creating Artificial Life.
Typically, this is done using some version of the genetic algorithm NEAT (e.g., RT-NEAT). However,
I wanted to look more into the Reinforcement Learning space and decided to forgo RT-NEAT for now. 

## Table of Contents  
<!--ts-->
   1. [Environment](#env)
   2. [Agents](#agents)
   3. [Reward](#reward)
   100. [To Do](#todo)
<!--te-->


Working:
* app_fast_dqn
* app_fast_dqn_evolve
* app_fast_ppo2a

* Changed reward structure such that is works with all of the above




<a name="env"/></a>
###  Environment

The basic environment is a grid of size *n* * *m* where each grid has a pixel size of *16* * *16*.

Within this environment we find the following things:
* Food (eating this food will increase your `health`)
    * Size = single grid 

To add:
* Food that slowly grows larger in size 
    * Hopefully, the agents will learn to wait before eating the food  

---

<a name="agents"/></a>
###  Agents

The agents are the size of a single grid. Only 1 agents is implemented.

An agent has the following characteristics:
* Health
    * Starts at 100 and decreases with 10 each step
* Age
    * Starts at 0 and increases 1 with each step  

An agent can perform the following actions:
* Move one space left, right, up, or down
    * They cannot move diagonally 

To add:
* Option to attack other agents   

---

<a name="reward"/></a>
###  Reward

The rewards are calculated as follows:
* 100 if eaten food
* -400 if died 

To add:
* More reward options   

---