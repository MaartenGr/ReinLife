# from stable_baselines.common.vec_env import DummyVecEnv
# from Field.Environment import Environment
# from Field import MultiEnvironment as Environment
#
# env = DummyVecEnv([lambda: Environment(mode='human', width=30, height=30)])
# obs = env.reset()
# while True:
#     is_running = env.render()
#     if not is_running:
#         break


from Field.MultiEnvironmentTest import MultiEnvironment as Environment

# from Field import MultiEnvironment as Environment
env = Environment(mode='human', width=30, height=30)
obs = env.reset()
while True:
    is_running = env.render()
    if not is_running:
        break