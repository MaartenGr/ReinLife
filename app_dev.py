from stable_baselines.common.vec_env import DummyVecEnv
from Field import MazeEnv

env = DummyVecEnv([lambda: MazeEnv(mode='human')])
obs = env.reset()
while True:
    is_running = env.render()
    if not is_running:
        break