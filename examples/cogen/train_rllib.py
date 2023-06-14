from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
import os
from typing import Any

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pandas as pd
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from tqdm import tqdm

from sustaingym.envs.cogen import CogenEnv, MultiAgentCogenEnv
from sustaingym.algorithms.base import RandomAlgorithm, RLLibAlgorithm


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
        choices=['ppo', 'rand'],
        help='Only "ppo" supports the mixed discrete/continuous action space'
             'in CogenEnv')
    parser.add_argument('-m', '--multiagent', action='store_true')
    parser.add_argument(
        '-r', '--seed', type=int, default=123,
        help='Random seed')
    parser.add_argument(
        '-l', '--lr', type=float, default=5e-5,
        help='Learning rate')
    parser.add_argument(
        '-n', '--rm', type=float, default=0,
        help='Renewables magnitude')
    args = parser.parse_args()

    config = {
        'algo': args.algo,
        'multiagent': args.multiagent,
        'seed': args.seed,
        'lr': args.lr,
        'rm': args.rm,
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
    rm = kwargs['rm']
    def _get_env() -> Any:
        if multiagent:
            env = MultiAgentCogenEnv(renewables_magnitude=rm)
        else:
            env = CogenEnv(renewables_magnitude=rm)
            env = FlattenObservation(env)
            # env = FlattenAction(env)
        return env
    return _get_env


def run_algo(config: dict, save_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_steps = 0

    # train environment
    config = config.copy()
    assert config['algo'] == 'ppo'
    train_config = ppo.PPOConfig()
    del config['algo']

    lr = config['lr']
    del config['lr']

    train_config = (
        train_config
        .environment(ENV_NAME, env_config=config)
        .training(train_batch_size=SPB, lr=lr, vf_clip_param=1e6)
        .rollouts(num_rollout_workers=2, num_envs_per_worker=2)
        .resources(num_gpus=1)
        .framework('torch')
    )
    if config['multiagent']:
        ma_env = get_env(**config)()
        policies = {
            agent: PolicySpec(
                observation_space=ma_env.observation_spaces[agent],
                action_space=ma_env.action_spaces[agent],
            ) for agent in ['GT1', 'GT2', 'GT3', 'ST']
        }
        policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id  # simply returns agent ID
        train_config = (
            train_config
            .rollouts(rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH)
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        )
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
    num_eval_episodes = 250  # YOU DECIDE
    config['rm'] = 300
    eval_env = get_env(**config)()
    rllib_algo = RLLibAlgorithm(eval_env, algo, multiagent=config['multiagent'])
    test_results_df = rllib_algo.run(num_eval_episodes)
    test_results_df.to_csv(os.path.join(save_dir, TEST_RESULTS), index=False)

    return train_results_df, test_results_df


def deploy_algo(config: dict, save_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(save_dir):
        raise ValueError(f"Checkpoint directory {save_dir} does not exist")

    # load the model from checkpoint
    config = config.copy()
    algo = Algorithm.from_checkpoint(os.path.join(save_dir + '/checkpoint_000062'))

    # eval
    num_eval_episodes = 250  # YOU DECIDE
    eval_env = get_env(**config)()
    rllib_algo = RLLibAlgorithm(eval_env, algo, multiagent=config['multiagent'])
    test_results_df = rllib_algo.run(num_eval_episodes)
    test_results_df.to_csv(os.path.join(save_dir, TEST_RESULTS), index=False)

    return test_results_df


def deploy_random_policy(config: dict, save_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert not config['multiagent']

    # eval
    num_eval_episodes = 250  # YOU DECIDE
    eval_env = get_env(**config)()
    rand_algo = RandomAlgorithm(eval_env, multiagent=config['multiagent'])
    test_results_df = rand_algo.run(num_eval_episodes)
    test_results_df.to_csv(os.path.join(save_dir, TEST_RESULTS), index=False)

    return test_results_df


if __name__ == '__main__':
    register_env(ENV_NAME, lambda config: get_env(**config)())
    config = parse_args()

    algo_name = config['algo']
    if config['multiagent']:
        algo_name = f'ma-{algo_name}'

    # 'logs/ENV_NAME/rllib/{algo_name}_lr{lr}_seed{seed}_rm{rm}'
    # TODO: customize for your environment (e.g., for dist shift)
    if algo_name == 'rand':
        folder_name = f'rand_rm{config["rm"]}'
        save_dir = os.path.join('logs/cogen', folder_name)
    else:
        folder_name = f'{algo_name}_lr{config["lr"]}_seed{config["seed"]}_rm{config["rm"]}'
        save_dir = os.path.join(SAVE_BASE_DIR, folder_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # run_algo(config, save_dir)
    deploy_random_policy(config, save_dir)
    # deploy_algo(config, save_dir)