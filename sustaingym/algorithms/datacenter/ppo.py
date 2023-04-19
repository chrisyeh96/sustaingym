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
    algo = ppo.PPO(env=DatacenterGym, config={ "env_config": {}, "framework": "torch", "disable_env_checking": True})

    while True:
        print(algo.train())

    # algo = PPOConfig().framework("torch").environment(env=DatacenterGym).build()
    # env = DatacenterGym()

    # episode_reward = 0
    # terminated = truncated = False

    # obs, info = env.reset()

    # while not terminated and not truncated:
    #     action = algo.compute_single_action(obs)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     episode_reward += reward
