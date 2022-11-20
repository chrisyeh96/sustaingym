"""
This module implements utility functions for plotting.
"""
from __future__ import annotations

from datetime import datetime
import pytz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

        gmm = utils.load_gmm_model(site, period[0], period[1], 30)['gmm']
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


def plot_violins(site: SiteStr, period: DefaultPeriodStr) -> None:
    """Plot violin plots for baselines."""
    algs = ['greedy', 'random_continuous', 'random_discrete']
    for window in [1, 3, 6, 12, 24]:
        algs.append(f'mpc_{window}')

    records = []
    for alg in algs:
        df = read_baseline(site, period, alg)
        reward = list(np.array(df.reward))
        for r in reward:
            records.append((alg, r))

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    df = pd.DataFrame.from_records(records, columns=['alg', 'reward'])
    sns.violinplot(data=df, x='alg', y='reward', ax=ax)
    ax.set(xlabel='Algorithm', ylabel="Reward ($)")
    ax.set_xticklabels(algs, rotation=30)
    fig.savefig(f'plots/violins_{site}_{period}.png', dpi=300, pad_inches=0)


if __name__ == '__main__':
    pass
    # # Plot gmm fits
    # plot_gmm_fit('caltech')
    # plot_gmm_fit('jpl')

    # # Plot violin plot
    # plot_violins('caltech', 'Summer 2021')
