from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
import os
from typing import Any
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation#, FlattenAction
import pandas as pd
from ray.rllib.algorithms import ppo, sac
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from tqdm import tqdm

from sustaingym.envs.cogen.cogen import CogenEnv
from sustaingym.algorithms.base import RLLibAlgorithm


ENV_NAME = 'cogen'

SPB = 4_000  # steps per batch
TOTAL_STEPS = 250_000
ROLLOUT_FRAGMENT_LENGTH = 1_000  # 'auto'
SAVE_BASE_DIR = f'logs/{ENV_NAME}/rllib'
TRAIN_RESULTS, TEST_RESULTS = 'train_results.csv', 'test_results.csv'


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""
    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)
        
    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


def parse_args() -> dict[str, Any]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f'train RLLib models on {ENV_NAME}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--algo', default='ppo',
        choices=['ppo', 'sac'],
        help="'ppo' or 'sac'")
    parser.add_argument('-m', '--multiagent', action='store_true')
    parser.add_argument(
        '-r', '--seed', type=int, default=123,
        help='Random seed')
    parser.add_argument(
        '-l', '--lr', type=float, default=5e-5,
        help='Learning rate')
    args = parser.parse_args()

    config = {
        'algo': args.algo,
        'multiagent': args.multiagent,
        'seed': args.seed,
        'lr': args.lr,
    }
    print('Config:', config)
    return config


def get_env(multiagent: bool = False, **kwargs) -> Callable:
    """Return environment.

    Args:
        multiagent: if True, return multi-agent environment

    Returns:
        Callable of environment
    """
    def _get_env() -> Any:
        if multiagent:
            raise NotImplementedError
        else:
            env = CogenEnv()
            env = FlattenObservation(env)
            # env = FlattenAction(env)
            return env
    return _get_env


def run_algo(config: dict, save_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_steps = 0

    # train environment
    config = config.copy()
    if config['algo'] == 'ppo':
        train_config = ppo.PPOConfig()
    elif config['algo'] == 'sac':
        train_config = sac.SACConfig()
    else:
        raise ValueError(f"{config['algo']} not in ['ppo', 'sac']")
    del config['algo']

    lr = config['lr']
    del config['lr']

    config['full'], config['real_trace'] = True, False
    train_config = (
        train_config
        .environment(ENV_NAME, env_config=config)
        .training(train_batch_size=SPB, lr=lr)
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
        .resources(num_gpus=1)
        .framework('torch')
    )
    if config['multiagent']:
        train_config = train_config.rollouts(rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH)
    algo = train_config.build(env=ENV_NAME)

    train_info = defaultdict(list)
    for i in tqdm(range(TOTAL_STEPS // SPB)):
        results = algo.train()
        algo.save(checkpoint_dir=save_dir)

        train_info['iter'].append(i)
        train_info['episode_reward_mean'].append(results['episode_reward_mean'])
        train_info['num_env_steps_trained'].append(results['num_env_steps_trained'])
        train_info['episodes_total'].append(results['episodes_total'])
        num_steps += SPB

    train_results_df = pd.DataFrame(train_info)
    train_results_df.to_csv(os.path.join(save_dir, TRAIN_RESULTS), index=False)

    # eval
    num_eval_episodes = 100  # YOU DECIDE
    eval_env = get_env(**config)()
    rllib_algo = RLLibAlgorithm(eval_env, algo, multiagent=config['multiagent'])
    test_results_df = rllib_algo.run(num_eval_episodes)
    test_results_df.to_csv(os.path.join(save_dir, TEST_RESULTS), index=False)

    return train_results_df, test_results_df


if __name__ == '__main__':
    register_env(ENV_NAME, lambda config: get_env(**config)())
    config = parse_args()

    algo_name = config['algo']
    if config['multiagent']:
        algo_name = f'ma-{algo_name}'

    # 'logs/ENV_NAME/rllib/{algo_name}_lr{lr}_seed{seed}'
    # TODO: customize for your environment (e.g., for dist shift)
    folder_name = f'{algo_name}_lr{config["lr"]}_seed{config["seed"]}'
    save_dir = os.path.join(SAVE_BASE_DIR, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_algo(config, save_dir)
