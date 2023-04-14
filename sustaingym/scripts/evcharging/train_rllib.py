import argparse
import os
import re
import string
from typing import Callable, Optional, Union

import gymnasium as gym
import pandas as pd
import ray
from ray import tune
from ray.air import session
from ray.rllib.algorithms import AlgorithmConfig, a2c, ppo, sac
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator, GMMsTraceGenerator, DiscreteActionWrapper, MultiAgentEVChargingEnv
from sustaingym.algorithms.evcharging.baselines import RLLibAlgorithm
from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator
from sustaingym.envs.evcharging.utils import \
    DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, SiteStr



###
SAMPLE_EVAL_PERIODS = {
    'Summer 2019':   ('2019-07-01', '2019-07-14'),
    'Fall 2019':     ('2019-11-04', '2019-11-17'),
    'Spring 2020':   ('2020-04-06', '2020-04-19'),
    'Summer 2021':   ('2021-07-05', '2021-07-18'),
}

ENV_NAME = "evcharging"

SPB = 1_000  # steps per batch
TOTAL_STEPS = 1_000  # 250_000
EVAL_EPISODES = 2  # 14
RLLIB_PATH = 'logs/RLLib'
TRAIN_RESULTS, TEST_RESULTS = 'train_results.csv', 'test_results.csv'


def parse_args() -> dict:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='train RLLib models on EVChargingEnv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-a', '--algo', type=str, default='ppo',
        help="'ppo', 'a2c', or 'sac'")
    parser.add_argument(
        '-t', '--train_date_period', type=str, default='Summer 2019',
        help="'Summer 2019' or 'Summer 2021'")
    parser.add_argument(
        '-s', '--site', type=str, default='caltech',
        help='site of garage. caltech or jpl')
    parser.add_argument('-d', '--discrete', action=argparse.BooleanOptionalAction)
    parser.add_argument('-m', '--multiagent', action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '-p', '--periods_delay', type=int, default=0,
        help='communication delay in multiagent setting. Ignored for single agent.')
    parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='Random seed')
    args = parser.parse_args()

    config = {
        "algo": args.algo,
        "dp": args.train_date_period,
        "site": args.site,
        "discrete": args.discrete,
        "multiagent": args.multiagent,
        "periods_delay": args.periods_delay,
        "seed": args.seed
    }
    print("Config: ", config)
    return config


def get_env(full: bool, real_trace: bool, dp: str, site: SiteStr, discrete: bool = False,
            multiagent: bool = True, periods_delay: int = 0, seed: int= None) -> Callable:
    """Return environment.

    Args:
        full: if True, use full season; otherwise, use sample 2 weeks
        real_trace: choice of generator
        dp: 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'
        site: 'caltech' or 'jpl'
        discrete: whether to wrap environment in discrete action wrapper
        seed: seed for GMMs generator
        multiagent: if True, return multi-agent environment. Note
            discrete = True and multiagent = True is currently not
            supported.
        periods_delay: number of timesteps for communication delay in
            the multiagent setting, ignored if multiagent = False
    
    Returns:
        Callable of environment
    """
    date_period = DEFAULT_PERIOD_TO_RANGE[dp] if full else SAMPLE_EVAL_PERIODS[dp]

    def _get_env():
        if real_trace:
            gen: AbstractTraceGenerator = RealTraceGenerator(site, date_period)
        else:
            gen = GMMsTraceGenerator(site, date_period, seed=seed)
        
        if discrete:
            if multiagent:
                raise ValueError("discrete = True and multiagent = True currently not supported")
            else:
                return DiscreteActionWrapper(EVChargingEnv(gen))
        else:
            if multiagent:
                return MultiAgentEVChargingEnv(gen, periods_delay=periods_delay)
            else:
                return EVChargingEnv(gen)
    return _get_env


def config_to_str(config: dict) -> str:
    s = str(config)
    savedir = s.translate(str.maketrans('', '', string.punctuation + ' '))
    return savedir


def run_algo(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_steps = 0

    # train environment
    config = config.copy()
    if config['algo'] == 'ppo':
        train_config = ppo.PPOConfig()
    elif config['algo'] == 'sac':
        train_config = sac.SACConfig()
    elif config['algo'] == 'a2c':
        train_config = a2c.A2CConfig()
    else:
        raise ValueError(f"{config['algo']} not in ['ppo', 'sac', 'a2c']")
    del config['algo']

    config['full'], config['real_trace'] = True, False
    train_config = (
        train_config
        .environment(ENV_NAME, env_config=config)
        .training(train_batch_size=SPB)
    )
    algo = train_config.build(env=ENV_NAME)

    # running eval environment
    config['full'], config['real_trace'], config['dp'], config['seed'] = False, True, 'Summer 2021', 0
    env = get_env(**config)

    train_results = []
    for i in range(TOTAL_STEPS // SPB):
        print(f"Iteration {i}")
        algo.train()
        algo.save()
        num_steps += SPB

        env = get_env(**config)
        rllib_algo = RLLibAlgorithm(env, algo, multiagent=config['multiagent'])
        reward_breakdown = rllib_algo.run(EVAL_EPISODES).to_dict('list')
        print(reward_breakdown)
        train_results.append(reward_breakdown)

    train_results_df = pd.DataFrame(train_results, index=range(1 * SPB, TOTAL_STEPS + SPB, SPB))
    train_results_df.to_csv(os.path.join(SAVE_PATH, TRAIN_RESULTS))

    # eval
    config['full'], config['real_trace'], config['dp'], config['seed'] = True, True, 'Summer 2021', 0
    eval_env = get_env(**config)
    rllib_algo = RLLibAlgorithm(eval_env, algo, multiagent=config['multiagent'])
    reward_breakdown = rllib_algo.run(EVAL_EPISODES).to_dict('list')
    test_results_df = pd.DataFrame(reward_breakdown)
    test_results_df.to_csv(os.path.join(SAVE_PATH, TEST_RESULTS))

    return train_results_df, test_results_df


def read_experiment(env_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = config_to_str(env_config)
    train_results_df = pd.read_csv(os.path.join(RLLIB_PATH, s, TRAIN_RESULTS), index_col=0)
    test_results_df = pd.read_csv(os.path.join(RLLIB_PATH, s, TEST_RESULTS), index_col=0)
    return train_results_df, test_results_df


if __name__ == '__main__':
    register_env(ENV_NAME, lambda config: get_env(**config)())

    config = parse_args()
    config_str = config_to_str(config)
    SAVE_PATH = os.path.join(RLLIB_PATH, config_str)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    run_algo(config)
    train_results_df, test_results_df = read_experiment(config)
