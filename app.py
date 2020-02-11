from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, DQN, SAC
from Field import MazeEnv

env = DummyVecEnv([lambda: MazeEnv()])
model = DQN(MlpPolicy, env, learning_rate=0.001, verbose=1)

model.learn(15_000, log_interval=100)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if not env.render():
        break