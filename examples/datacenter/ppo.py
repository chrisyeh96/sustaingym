from __future__ import annotations

import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
import sys
sys.path.append(".")
from sustaingym.envs.datacenter import DatacenterGym

if __name__ == "__main__":
    ray.init()
    # TODO:
    """
    `algo = Algorithm(env='<class 'sustaingym.envs.datacenter.gym.DatacenterGym'>', ...)` has been deprecated. Use `algo = AlgorithmConfig().environment('<class 'sustaingym.envs.datacenter.gym.DatacenterGym'>').build()` instead. This will raise an error in the future!
    """
    env = gym.wrappers.FlattenObservation(DatacenterGym())

    algo = ppo.PPO(env=env, config={ "env_config": {}, "framework": "torch", "disable_env_checking": True})

    episodes = []
    mean_reward = []
    while True:
        d = algo.train()
        episodes_total = d["episodes_total"]
        mean_reward.append(d["episode_reward_mean"])
        episodes.append(episodes_total)
        mean_reward.append(mean_reward)
        print(f"Episode #{episodes_total} reward: {d['episode_reward_mean']}")
        if episodes_total > 100:
            break

    # algo = PPOConfig().framework("torch").environment(env=DatacenterGym).build()
    # env = DatacenterGym()

    # episode_reward = 0
    # terminated = truncated = False

    # obs, info = env.reset()

    # while not terminated and not truncated:
    #     action = algo.compute_single_action(obs)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     episode_reward += reward
