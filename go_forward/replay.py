from sim_random import Sim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = Sim(30)

env.reset()

model = PPO.load("saved_steering_reward", env=env)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

env.render_episode(model, "test", 20, 190)
