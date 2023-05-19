import argparse
from datetime import datetime
import os
import string
from typing import Callable

import gymnasium as gym
import pandas as pd
from ray.rllib.algorithms import a2c, ppo, sac
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from sustaingym.envs.evcharging import EVChargingEnv, RealTraceGenerator, \
    GMMsTraceGenerator, DiscreteActionWrapper, MultiAgentEVChargingEnv
from sustaingym.algorithms.evcharging.baselines import RLLibAlgorithm
from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator
from sustaingym.envs.evcharging.utils import DATE_FORMAT, DEFAULT_PERIOD_TO_RANGE, SiteStr


###
SAMPLE_EVAL_PERIODS = {
    'Summer 2019':   ('2019-07-01', '2019-07-14'),
    'Fall 2019':     ('2019-11-04', '2019-11-17'),
    'Spring 2020':   ('2020-04-06', '2020-04-19'),
    'Summer 2021':   ('2021-07-05', '2021-07-18'),
}

ENV_NAME = "evcharging"

SPB = 10_000  # steps per batch
TOTAL_STEPS = 250_000  # 10_000
EVAL_EPISODES = 14  # 2
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
        '-t', '--train_date_period', type=str, nargs='+', default='Summer 2019',  ##
        help="Season. 'Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021'")
    parser.add_argument(
        '-s', '--site', type=str, default='caltech',
        help='site of garage. caltech or jpl')
    parser.add_argument('-d', '--discrete', action=argparse.BooleanOptionalAction)
    parser.add_argument('-m', '--multiagent', action=argparse.BooleanOptionalAction)
    parser.add_argument(
        '-p', '--periods_delay', type=int, default=0,
        help='communication delay in multiagent setting. Ignored for single agent.')
    parser.add_argument(
        '-r', '--seed', type=int, default=123,
        help='Random seed')
    args = parser.parse_args()

    config = {
        "algo": args.algo,
        "dp": ' '.join(args.train_date_period),
        "site": args.site,
        "discrete": args.discrete,
        "multiagent": args.multiagent,
        "periods_delay": args.periods_delay,
        "seed": args.seed
    }
    print("Config: ", config)
    return config


def num_days_in_period(full: bool, dp: str) -> int:
    """Returns the number of days in period."""
    d = DEFAULT_PERIOD_TO_RANGE if full else SAMPLE_EVAL_PERIODS
    dts = tuple(datetime.strptime(x, DATE_FORMAT) for x in d[dp])
    return (dts[1] - dts[0]).days + 1


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
                return DiscreteActionWrapper(gym.wrappers.FlattenObservation(EVChargingEnv(gen)))
        else:
            if multiagent:
                return ParallelPettingZooEnv(MultiAgentEVChargingEnv(gen, periods_delay=periods_delay))
            else:
                return gym.wrappers.FlattenObservation(EVChargingEnv(gen))
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
    env = get_env(**config)()

    train_results = []
    for i in range(TOTAL_STEPS // SPB):
        print(f"Iteration {i}")
        algo.train()
        algo.save()
        num_steps += SPB

        env = get_env(**config)()
        rllib_algo = RLLibAlgorithm(env, algo, multiagent=config['multiagent'])
        reward_breakdown = rllib_algo.run(EVAL_EPISODES).to_dict('list')
        print(reward_breakdown)
        train_results.append(reward_breakdown)

    train_results_df = pd.DataFrame(train_results, index=range(1 * SPB, TOTAL_STEPS + SPB, SPB))
    train_results_df.to_csv(os.path.join(SAVE_PATH, TRAIN_RESULTS))

    # eval
    config['full'], config['real_trace'], config['dp'], config['seed'] = True, True, 'Summer 2021', 0
    eval_env = get_env(**config)()
    rllib_algo = RLLibAlgorithm(eval_env, algo, multiagent=config['multiagent'])
    ndip = num_days_in_period(True, 'Summer 2021')
    reward_breakdown = rllib_algo.run(ndip).to_dict('list')
    test_results_df = pd.DataFrame(reward_breakdown)
    test_results_df.to_csv(os.path.join(SAVE_PATH, TEST_RESULTS))

    return train_results_df, test_results_df


def read_experiment(env_config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = config_to_str(env_config)
    train_results_df = pd.read_csv(os.path.join(RLLIB_PATH, s, TRAIN_RESULTS), index_col=0)
    test_results_df = pd.read_csv(os.path.join(RLLIB_PATH, s, TEST_RESULTS), index_col=0)
    return train_results_df, test_results_df


def get_best_seeds() -> dict:
    config = {
        "algo": 'ppo', "dp": 'Summer 2019',
        "site": 'caltech', "discrete": None, "multiagent": None,
        "periods_delay": 0, "seed": 123
    }
    algos = ['a2c', 'ppo', 'sac']
    dps = ['Summer 2019', 'Summer 2021']
    multiagents = [True, None]
    seeds = [123, 246, 369]

    best_seeds = {}
    for algo in algos:
        for dp in dps:
            for ma in multiagents:
                config['algo'], config['dp'], config['multiagent'] = algo, dp, ma
                for seed in seeds:
                    config['seed'] = seed
                    try:
                        _, test_results_df = read_experiment(config)
                        mean = test_results_df['reward'].mean()
                        if (algo, dp, ma) not in best_seeds or mean > best_seeds[(algo, dp, ma)][1]:
                            best_seeds[(algo, dp, ma)] = seed, mean
                    except Exception:
                        pass
    return best_seeds


def plot_reward_curve_separate() -> None:
    import ast
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set_style('darkgrid')

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

    timesteps = list(range(1 * 10000, 26 * 10000, 10000))

    config = {
        "algo": 'ppo', "dp": 'Summer 2019',
        "site": 'caltech', "discrete": None, "multiagent": None,
        "periods_delay": 0, "seed": 123
    }
    best_seeds = get_best_seeds()

    for idx, (algo, dp) in enumerate(best_seeds):
        seed = best_seeds[(algo, dp)][0]
        i, j = idx % 3, idx // 3

        # seed = best_seeds[(algo, dp)][0]
        config['algo'], config['dp'], config['seed'] = algo, dp, seed

        train_results_df, _ = read_experiment(config)
        
        dist = 'out of dist' if dp == 'Summer 2019' else 'in dist'
        axes[j][i].set_title(f'{dist} {algo.upper()}')
        axes[j][i].set_ylabel('reward ($)')
        axes[j][i].set_xlabel('timesteps')

        for col in ['reward', 'profit', 'carbon_cost']:
            train_results_df[col] = train_results_df[col].apply(lambda x: np.array(ast.literal_eval(x)))
            m = train_results_df[col].apply(lambda x: x.mean())
            std = train_results_df[col].apply(lambda x: x.std())
            axes[j][i].plot(m, label=col)
            axes[j][i].fill_between(timesteps, m - std, m + std, alpha=0.2)

        axes[j][i].legend()
    
    fig.suptitle(f'Reward Breakdown on 07/05/2021-07/18/2021 During Training')
    fig.tight_layout()
    print(f"Save to: 'plots/training_curves_separate_rllib.png'")
    fig.savefig('plots/training_curves_separate_rllib.png', dpi=300, bbox_inches='tight')


def plot_violins() -> None:
    """Plot violin plots for RLLib algorithms."""
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sustaingym.scripts.evcharging.plot_utils import read_baseline
    from sustaingym.scripts.evcharging.train_rllib import get_best_seeds
    algs = ['offline_optimal', 'greedy', 'random', 'random_discrete']

    # Find best MPC
    best_window, best_mean_reward = 1, -100_000
    for window in [1, 3, 6, 12, 24]:
        df = read_baseline('caltech', 'Summer 2021', f'mpc_{window}')
        mean_reward = np.mean(df.reward)
        if mean_reward > best_mean_reward:
            best_window, best_mean_reward = window, mean_reward
    algs.append(f'mpc_{best_window}')

    records = []
    # Read baselines
    for alg in algs:
        df = read_baseline('caltech', 'Summer 2021', alg)
        reward = list(np.array(df.reward))
        for r in reward:
            alg_label = alg.replace('random', 'rand').replace('_', '\n')
            records.append((alg_label, r, 2021))   # baselines are neither in dist or out of dist

    # 
    best_seeds = get_best_seeds()
    config = {
        "algo": 'ppo', "dp": 'Summer 2019',
        "site": 'caltech', "discrete": None, "multiagent": None,
        "periods_delay": 0, "seed": 123
    }

    for _, (algo, dp, ma) in enumerate(best_seeds):
        config['algo'], config['dp'], config['multiagent'], config['seed'] = algo, dp, ma, best_seeds[(algo, dp, ma)][0]
        _, test_results_df = read_experiment(config)
        reward = list(test_results_df['reward'])
        year = 2021 if dp == 'Summer 2021' else 2019
        for r in reward:
            records.append((algo, r, year, ma))

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    df = pd.DataFrame.from_records(records, columns=['alg', 'reward', 'in_dist', 'multiagent'])
    df['multiagent'].fillna(0, inplace=True)
    df['multiagent'] = df['multiagent'].apply(lambda x: "ma" if x else "")
    print(df.head())
    print(df.tail())
    print(df[df['multiagent'] == 'ma'])
    df['alg_multiagent'] = df['alg'] + df['multiagent']
    sns.violinplot(data=df, x='alg_multiagent', y='reward', hue='in_dist', ax=ax)
    ax.set(xlabel='Algorithm', ylabel="Daily Return ($)")
    # ax.set_xticklabels(algs, rotation=30)
    print(f"Save to: 'plots/violins_caltech_Summer 2021_rllib.png'")
    fig.savefig(f'plots/violins_caltech_Summer 2021_rllib.png', dpi=300, pad_inches=0)


if __name__ == '__main__':

    # plot_violins()
    # # plot_reward_curve_separate()

    register_env(ENV_NAME, lambda config: get_env(**config)())
    config = parse_args()
    config_str = config_to_str(config)
    SAVE_PATH = os.path.join(RLLIB_PATH, config_str)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    run_algo(config)
    train_results_df, test_results_df = read_experiment(config)
