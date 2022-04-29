from sim import Sim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = Sim(60)

env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./test_tensorboard2")
model.learn(total_timesteps=3000)
env.render_episode(model, "3000", 10)
model.learn(total_timesteps=3000)
env.render_episode(model, "6000", 10)

#model.save("saved")
#model = PPO.load("saved", env=env)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward: .2f}")

