"""
This module implements utility functions for plotting.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

from sustaingym.envs.evcharging.utils import SiteStr, DefaultPeriodStr


def read_baseline(site: SiteStr, period: DefaultPeriodStr, algorithm: str) -> pd.DataFrame:
    """Read reward results from csv files."""
    return pd.read_csv(f'logs/baselines/{site}_{period}_{algorithm}.csv', compression='gzip')


def read_rl(experiment_identifier: int, rl_algorithm: str, csv_file: str = 'test_results') -> pd.DataFrame:
    """Read reward results from csv files."""
    return pd.read_csv(f'logs/{rl_algorithm}/exp{experiment_identifier}/{csv_file}.csv', compression='gzip')


def plot_lines(site: SiteStr, period: DefaultPeriodStr) -> None:
    """Plot line plots for baselines over evaluation period."""
    algs = ['offline_optimal', 'greedy', 'random', 'random_discrete', 'mpc_6']

    records = []
    for alg in algs:
        df = read_baseline(site, period, alg).rolling(7).median()
        df['day'] = df.index
        day = list(np.array(df.day))
        reward = list(np.array(df.reward))
        for r, di in zip(reward, day):
            records.append((alg, r, di))

    df = pd.DataFrame.from_records(records, columns=['alg', 'reward', 'day'])
    sns.lineplot(data=df, x="day", y="reward", hue="alg")
    plt.title('7-Day Rolling Median of Rewards')
    plt.ylabel("reward ($)")
    plt.savefig(f'plots/lines_{site}_{period}.png', dpi=300, pad_inches=0)


def plot_violins(site: SiteStr, period: DefaultPeriodStr) -> None:
    """Plot violin plots for baselines."""
    algs = ['offline_optimal', 'greedy', 'random', 'random_discrete']

    # Find best MPC
    best_window, best_mean_reward = 1, -100_000
    for window in [1, 3, 6, 12, 24]:
        df = read_baseline(site, period, f'mpc_{window}')
        mean_reward = np.mean(df.reward)
        if mean_reward > best_mean_reward:
            best_window, best_mean_reward = window, mean_reward
    algs.append(f'mpc_{best_window}')

    records = []
    # Read baselines
    for alg in algs:
        df = read_baseline(site, period, alg)
        reward = list(np.array(df.reward))
        for r in reward:
            alg_label = alg.replace('random', 'rand').replace('_', '\n')
            records.append((alg_label, r, 2021))   # baselines are neither in dist or out of dist

    # method, in dist vs. out of dist
    for alg in ['A2C', 'A2C_discrete', 'SAC', 'PPO', 'PPO_discrete']:
        algs.append(alg)

    for rl in ['A2C', 'SAC', 'PPO']:
        for exps in [(1000, 1002), (1003, 1005), (1006, 1008), (1009, 1011)]:
            if rl == 'SAC' and exps[0] >= 1006:
                continue  # no discrete SAC

            # Find best performing experiment of the three
            best_mean_reward = -100_000
            for exp in range(exps[0], exps[1] + 1):
                rld = rl if (exp <= 1005) else rl + '_discrete'
                df = read_rl(exp, rld)
                if np.mean(df.reward) > best_mean_reward:
                    best_mean_reward, reward = np.mean(df.reward), df.reward

            # Add reward to records
            reward = list(np.array(reward))
            for r in reward:
                rld_spaces = rld.replace('_', '\n')
                if exps[0] == 1000:
                    records.append((rld_spaces, r, 2019))
                elif exps[0] == 1003:
                    records.append((rld_spaces, r, 2021))
                elif exps[0] == 1006:
                    records.append((rld_spaces, r, 2019))
                elif exps[0] == 1009:
                    records.append((rld_spaces, r, 2021))
                else:
                    raise ValueError

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    df = pd.DataFrame.from_records(records, columns=['alg', 'reward', 'in_dist'])
    print(df.head())
    print(df.tail())
    sns.violinplot(data=df, x='alg', y='reward', hue='in_dist', ax=ax)
    ax.set(xlabel='Algorithm', ylabel="Daily Return ($)")
    # ax.set_xticklabels(algs, rotation=30)
    print(f"Save to: 'plots/violins_{site}_{period}.png'")
    fig.savefig(f'plots/violins_{site}_{period}.png', dpi=300, pad_inches=0)


def reward_curve_separate() -> None:
    """Plot reward curves separately for 'caltech' site in Summer 2021."""
    # reward curve w/o action projection on 2-week sample in period
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

    algs = {
        ('A2C', 'out of dist'): [1000, 1001, 1002],
        ('PPO', 'out of dist'): [1000, 1001, 1002],
        ('SAC', 'out of dist'): [1000, 1001, 1002],
        ('A2C', 'in dist'):     [1003, 1004, 1005],
        ('PPO', 'in dist'):     [1003, 1004, 1005],
        ('SAC', 'in dist'):     [1003, 1004, 1005],
    }
    best = {}

    # determine best in each group
    for alg in algs:
        best_exp, best_mean_reward = 0, -100_000
        for exp in algs[alg]:
            df = read_rl(exp, alg[0])
            if np.mean(df.reward) > best_mean_reward:
                best_exp, best_mean_reward = exp, np.mean(df.reward)
        best[alg] = best_exp

    for i, group in enumerate(best):
        evals = read_rl(best[group], group[0], csv_file='evaluations')
        timesteps = evals.timestep.unique()
        profit_means, profit_stds, cc_means, cc_stds, rew_means, rew_stds = [], [], [], [], [], []
        for timestep in timesteps:
            profit_means.append(evals[evals.timestep == timestep].profit.mean())
            profit_stds.append(evals[evals.timestep == timestep].profit.std())
            cc_means.append(evals[evals.timestep == timestep].carbon_cost.mean())
            cc_stds.append(evals[evals.timestep == timestep].carbon_cost.std())
            rew_means.append(evals[evals.timestep == timestep].reward.mean())
            rew_stds.append(evals[evals.timestep == timestep].reward.std())
        profit_means = np.array(profit_means)
        profit_stds = np.array(profit_stds)
        cc_means = np.array(cc_means)
        cc_stds = np.array(cc_stds)
        rew_means = np.array(rew_means)
        rew_stds = np.array(rew_stds)

        r = i // 3
        c = i % 3
        axes[0][0].set_ylabel('reward')
        axes[1][0].set_ylabel('reward')
        axes[r][c].plot(timesteps, profit_means, label='profit')
        axes[r][c].fill_between(timesteps, profit_means-profit_stds, profit_means+profit_stds, alpha=0.2)
        axes[r][c].plot(timesteps, cc_means, label='carbon cost')
        axes[r][c].fill_between(timesteps, cc_means-cc_stds, cc_means+cc_stds, alpha=0.2)
        axes[r][c].plot(timesteps, rew_means, label='reward')
        axes[r][c].fill_between(timesteps, rew_means-rew_stds, rew_means+rew_stds, alpha=0.2)
        axes[r][c].set_xlabel('timesteps')
        axes[r][c].legend()
        axes[r][c].set_title(f'{group[1]} {group[0]}')

    fig.suptitle('Reward Breakdown on 07/05/2021-07/18/2021 During Training')
    fig.tight_layout()
    print("Save to: 'plots/training_curves_separate.png'")
    fig.savefig('plots/training_curves_separate.png', dpi=300, bbox_inches='tight')


def reward_curve_all():
    """Plot reward curves together for 'caltech' site in Summer 2021."""
    # reward curve w/o action projection on 2-week sample in period
    algs = {
        ('A2C', 'out of dist'): [1000, 1001, 1002],
        ('A2C', 'in dist'):     [1003, 1004, 1005],
        ('PPO', 'out of dist'): [1000, 1001, 1002],
        ('PPO', 'in dist'):     [1003, 1004, 1005],
    }
    best = {}

    # determine best in each group
    for alg in algs:
        best_exp, best_mean_reward = 0, -100_000
        for exp in algs[alg]:
            df = read_rl(exp, alg[0])
            if np.mean(df.reward) > best_mean_reward:
                best_exp, best_mean_reward = exp, np.mean(df.reward)
        best[alg] = best_exp

    for group in best:
        evals = read_rl(best[group], group[0], 'evaluations')
        timesteps = evals.timestep.unique()
        reward_means, reward_stds = [], []
        for timestep in timesteps:
            timestep_rewards = evals[evals.timestep == timestep].reward
            reward_means.append(timestep_rewards.mean())
            reward_stds.append(timestep_rewards.std())
        reward_means, reward_stds = np.array(reward_means), np.array(reward_stds)

        plt.plot(timesteps, reward_means, label=f'{group[1]} {group[0]}')
        plt.fill_between(timesteps, reward_means-reward_stds, reward_means+reward_stds, alpha=0.2)
    plt.title('Reward on 07/05/2021-07/18/2021 During Training')
    plt.xlabel('timesteps')
    plt.ylabel('reward')
    plt.legend()
    print("Save to: 'plots/training_curves_all.png'")
    plt.savefig('plots/training_curves_all.png', dpi=300)


if __name__ == '__main__':
    # # Plot violin plot
    # plot_violins('caltech', 'Summer 2021')

    # # Plot line plot
    # plot_lines('caltech', 'Summer 2021')

    # # Plot training curves in one place
    # reward_curve_all()

    # Plot training curves separately
    reward_curve_separate()
