from stable_baselines.common.vec_env import DummyVecEnv
from Field.Environment import Environment

env = DummyVecEnv([lambda: Environment(mode='human', width=10, height=12)])
obs = env.reset()
while True:
    is_running = env.render()
    if not is_running:
        break