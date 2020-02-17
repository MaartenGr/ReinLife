from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy as MlpPolicyDQN
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, DQN, SAC
from Field.Environment import Environment
# import tensorflow as tf

# policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[32, 32])
env = make_vec_env(Environment, n_envs=8, env_kwargs=dict(width=30, height=30))

# model = DQN(MlpPolicyDQN, env, learning_rate=0.0001, verbose=1, tensorboard_log="logging")
model = PPO2(MlpPolicy, env, learning_rate=0.01, verbose=1, tensorboard_log="logging")
# model = PPO2('MlpPolicy',  env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="logging")

model.learn(1_000_000, log_interval=50)

env = DummyVecEnv([lambda: Environment(width=30, height=30)])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if not env.render():
        break