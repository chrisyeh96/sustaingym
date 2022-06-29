import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from .ev_charging import EVChargingEnv


if __name__ == '__main__':
    env = EVChargingEnv()

    env_id = "CartPole-v1"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([EVChargingEnv() for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    avg_rewards = []
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        avg_rewards.append(rewards)
    avg_rewards = np.mean(avg_rewards)
    print("beginning rewards: ", avg_rewards)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=25_000)

    avg_rewards = []
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        avg_rewards.append(rewards)
    avg_rewards = np.mean(avg_rewards)
    print("ending rewards: ", avg_rewards)
