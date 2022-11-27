"""
This module implements utility functions for plotting.
"""
from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.mixture import GaussianMixture

from sustaingym.envs.evcharging import utils
from sustaingym.envs.evcharging.train_gmm_model import preprocess
from sustaingym.envs.evcharging.utils import \
    DATE_FORMAT, AM_LA, DEFAULT_PERIOD_TO_RANGE, SiteStr, DefaultPeriodStr


def plot_gmm_fit(site: SiteStr) -> None:
    """Plot actual arrival/departures against GMM PDF contours."""
    seasons = ['Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021']
    periods, dfs, gmms, pdfs = [], [], [], []

    # Display predicted scores by the model as a contour plot
    arr, dep = np.linspace(0.0, 1.0), np.linspace(0.0, 1.0)

    # Fit gmm on mesh to generate contour lines
    X, Y = np.meshgrid(arr, dep)
    Xr, Yr = X.ravel(), Y.ravel()
    x_b = np.vstack((Xr, Yr))  # shape (2, N)

    for season in seasons:
        period = (datetime.strptime(DEFAULT_PERIOD_TO_RANGE[season][0], DATE_FORMAT).replace(tzinfo=AM_LA),
                  datetime.strptime(DEFAULT_PERIOD_TO_RANGE[season][1], DATE_FORMAT).replace(tzinfo=AM_LA))
        periods.append(period)

        df = preprocess(utils.get_real_events(period[0], period[1], site))
        dfs.append(df)

        gmm_model = utils.load_gmm_model(site, period[0], period[1], 30)
        gmm: GaussianMixture = gmm_model['gmm']
        gmms.append(gmm)

        # Calculate conditional mean for estimated departure, requested energy
        weights = np.expand_dims(gmm.weights_, (1, 2))
        n_components = len(gmm.weights_)

        conditional_means = []
        for i in range(n_components):
            mu_a = np.expand_dims(gmm.means_[i, 2:], 1)
            mu_b = np.expand_dims(gmm.means_[i, :2], 1)
            mu_aa_precision = gmm.precisions_[i, 2:, 2:]
            mu_ab_precision = gmm.precisions_[i, 2:, :2]
            cond_a_b = mu_a - np.linalg.pinv(mu_aa_precision) @ mu_ab_precision @ (x_b - mu_b)
            conditional_means.append(cond_a_b)
        conditional_means = np.array(conditional_means)
        conditional_mean = np.sum(weights * conditional_means, axis=0).T  # N, 2

        gmm_input = pd.DataFrame({
            'arrival_time': Xr,
            'departure_time': Yr,
            'estimated_departure_time': conditional_mean[:, 0],
            'requested_energy (kWh)': conditional_mean[:, 1]
        })
        Z = gmm.score_samples(gmm_input).reshape(X.shape)

        pdfs.append((X, Y, Z))

    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(12, 3.5), tight_layout=True)
    for i, ax in enumerate(axs):
        X, Y, Z = pdfs[i]
        x = X * 288 # rescale to 5-min periods
        y = Y * 288
        # same with arrivals and departures
        arr_time = dfs[i].arrival_time * 288
        dep_time = dfs[i].departure_time * 288

        cp = ax.contourf(x, y, Z, cmap='Oranges', alpha=0.5)
        ax.scatter(arr_time, dep_time, marker='+', alpha=0.2, color='tab:blue')
        ax.set(xlabel='arrival time', title=f'{seasons[i]}')
        ax.set_aspect('equal', 'box')
        ax.set_xticks([0, 96, 192, 288], labels=['0:00', '8:00', '16:00', '24:00'])

    axs[0].set_ylabel('departure time')
    axs[0].set_yticks([0, 96, 192, 288], labels=['0:00', '8:00', '16:00', '24:00'])

    cbar_ax = fig.add_axes([1, 0.18, 0.02, 0.66])
    cbar = fig.colorbar(cp, cax=cbar_ax)
    cbar.ax.set_ylabel('log-likelihood', rotation=270)
    fig.savefig(f'plots/gmms_fit_{site}.png', dpi=300, pad_inches=0)


def read_baseline(site: SiteStr, period: DefaultPeriodStr, algorithm: str) -> pd.DataFrame:
    """Read reward results from csv files."""
    return pd.read_csv(f'logs/baselines/{site}_{period}_{algorithm}.csv', compression='gzip')


def read_rl(experiment_identifier: int, rl_algorithm: str, csv_file: str = 'test_results') -> pd.DataFrame:
    """Read reward results from csv files."""
    return pd.read_csv(f'logs/{rl_algorithm}/exp{experiment_identifier}/{csv_file}.csv', compression='gzip')


def plot_cross_scores():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('30-component GMM Average Log-Likelihood Scores')

    for site_number, site in enumerate(['caltech', 'jpl']):
        seasons = ['Summer 2019', 'Fall 2019', 'Spring 2020', 'Summer 2021']
        periods, dfs, gmms = [], [], []
        for season in seasons:
            period = (datetime.strptime(utils.DEFAULT_PERIOD_TO_RANGE[season][0], DATE_FORMAT).replace(tzinfo=AM_LA),
                    datetime.strptime(utils.DEFAULT_PERIOD_TO_RANGE[season][1], DATE_FORMAT).replace(tzinfo=AM_LA))
            periods.append(period)

            df = preprocess(utils.get_real_events(period[0], period[1], site))
            dfs.append(df)

            gmm = utils.load_gmm_model(site, period[0], period[1], 30)['gmm']
            gmms.append(gmm)

        train_cols = {season: [] for season in seasons}

        for i, train_season in enumerate(seasons):
            for j, _ in enumerate(seasons):
                train_cols[train_season].append(gmms[i].score(dfs[j]))

        cross_scores = pd.DataFrame(train_cols, index=seasons)
        print(cross_scores)
        sns.heatmap(cross_scores, cmap='Blues', annot=True, fmt='.3g', ax=axes[site_number])
        axes[site_number].set_title(site)
        axes[site_number].set_ylabel('Testing period')
        axes[site_number].set_xlabel('Training period')
        axes[site_number].tick_params(axis='x', labelrotation = 20)
    plt.savefig('plots/gmm_log_likelihoods.png', dpi=300, bbox_inches="tight")


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
    ax.set(xlabel='Algorithm', ylabel="Reward ($)")
    # ax.set_xticklabels(algs, rotation=30)
    print(f"Save to: 'plots/violins_{site}_{period}.png'")
    fig.savefig(f'plots/violins_{site}_{period}.png', dpi=300, pad_inches=0)


def reward_curve_separate() -> None:
    """Plot reward curves separately for 'caltech' site in Summer 2021."""
    # reward curve w/o action projection on 2-week sample in period
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)

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

    for i, group in enumerate(best):
        evals = read_rl(best[group], group[0], csv_file='evaluations')
        timesteps = evals.timestep.unique()
        profit_means, profit_stds, cc_means, cc_stds = [], [], [], []
        for timestep in timesteps:
            profit_means.append(evals[evals.timestep == timestep].profit.mean())
            profit_stds.append(evals[evals.timestep == timestep].profit.std())
            cc_means.append(evals[evals.timestep == timestep].carbon_cost.mean())
            cc_stds.append(evals[evals.timestep == timestep].carbon_cost.std())
        profit_means = np.array(profit_means)
        profit_stds = np.array(profit_stds)
        cc_means = np.array(cc_means)
        cc_stds = np.array(cc_stds)

        axes[i].plot(timesteps, profit_means, label='profit')
        axes[i].fill_between(timesteps, profit_means-profit_stds, profit_means+profit_stds, alpha=0.2)
        axes[i].plot(timesteps, cc_means, label='carbon cost')
        axes[i].fill_between(timesteps, cc_means-cc_stds, cc_means+cc_stds, alpha=0.2)
        axes[i].set_xlabel('timesteps')
        axes[0].set_ylabel('reward')
        axes[i].legend()
        axes[i].set_title(f'{group[1]} {group[0]}')

    fig.suptitle(f'Reward Breakdown on 07/05/2021-07/18/2021 During Training', y=1.07)
    print(f"Save to: 'plots/training_curves_separate.png'")
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
    plt.title(f'Reward on 07/05/2021-07/18/2021 During Training')
    plt.xlabel('timesteps')
    plt.ylabel('reward')
    plt.legend()
    print(f"Save to: 'plots/training_curves_all.png'")
    plt.savefig('plots/training_curves_all.png', dpi=300)



if __name__ == '__main__':
    # # Plot gmm fits
    # plot_gmm_fit('caltech')
    # plot_gmm_fit('jpl')

    # # Plot violin plot
    # plot_violins('caltech', 'Summer 2021')

    # # Plot cross scores
    # plot_cross_scores()

    # # Plot line plot
    # plot_lines('caltech', 'Summer 2021')

    # Plot training curves in one place
    reward_curve_all()

    # # Plot training curves separately
    # reward_curve_separate()
