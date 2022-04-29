from sim import Sim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = Sim(60)

env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./PPO_balancing_tensorboard")
for i in range(4):
    model.learn(total_timesteps=23000, reset_num_timesteps=False)
    env.render_episode(model, str(32000 * (i + 1)), 10)

model.save("saved")
#model = PPO.load("saved", env=env)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward: .2f}")

