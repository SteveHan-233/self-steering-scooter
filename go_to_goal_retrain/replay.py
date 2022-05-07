from sim import Sim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = Sim(30)

env.reset()

model = PPO.load("saved_retrain_easy_first", env=env)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

env.render_episode(model, "test", 40)

