import gym
from TheGame.Models.rainbow import DQNAgent

# parameters
num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100

env_id = "CartPole-v0"
env = gym.make(env_id)

# train
agent = DQNAgent(env, memory_size, batch_size, target_update)
agent.train(num_frames)
