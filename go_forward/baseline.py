from sim import Sim
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = Sim(30)

env.reset()

policy_kwargs = dict(net_arch=[dict(pi=[300, 400], vf=[200, 200])])

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./PPO_balancing_tensorboard", policy_kwargs=policy_kwargs)
#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./PPO_balancing_tensorboard")
env.render_episode(model, str(0), 10)
for i in range(20):
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    env.render_episode(model, str(100000 * (i + 1)), 10)
    model.save("saved")

#model = PPO.load("saved", env=env)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward: .2f}")

