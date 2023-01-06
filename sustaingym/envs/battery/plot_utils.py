"""
Plotting helper functions
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns

from sustaingym.envs import ElectricityMarketEnv


def training_eval_results(root_folder: str, relative_path: str) -> tuple:
    results = []
    fname = f'{root_folder}/{relative_path}/evaluations.npz'
    x = np.load(fname, allow_pickle=True)
    results.append(x['results'])

    timesteps = x['timesteps']
    results = np.hstack(results)
    y = results.mean(axis=1)
    error = results.std(axis=1)
    return timesteps, y, error


def plot_model_training_reward_curves(
        ax: plt.Axes, model: str, paths: Sequence[str], dists: Sequence[str], in_dist_year: int, out_dist_year: int | None = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    evals_lst = []
    err_lst = []

    if "in_dist" in dists:
        timesteps, y, err = training_eval_results(paths[0], f"eval{in_dist_year}-05")
        evals_lst.append(y)
        err_lst.append(err)
    
    min_len = len(timesteps)

    if "out_dist" in dists:
        timesteps, y, err = training_eval_results(paths[1], f"eval{in_dist_year}-05")
        evals_lst.append(y)
        err_lst.append(err)
    
    if len(timesteps) < min_len:
        min_len = len(timesteps)

    ax.plot(timesteps[:min_len], evals_lst[0][:min_len], label=f'train on May {in_dist_year}, evaluate on May {in_dist_year}')
    ax.fill_between(timesteps[:min_len], evals_lst[0][:min_len]-err_lst[0][:min_len], evals_lst[0][:min_len]+err_lst[0][:min_len], alpha=0.2)

    if len(dists) == 2:
        ax.plot(timesteps[:min_len], evals_lst[1][:min_len], label=f'train on May {out_dist_year}, evaluate on May {in_dist_year}')
        ax.fill_between(timesteps[:min_len], evals_lst[1][:min_len]-err_lst[1][:min_len],
                evals_lst[1][:min_len]+err_lst[1][:min_len], alpha=0.2)

    ax.set_title(f'{model} Training Reward Curves')
    ax.set_ylabel('reward ($)')
    ax.set_xlabel('timesteps')
    ax.legend()

    return ax


def plot_returns(results: Mapping[str, Mapping[str, np.ndarray]],
                 ylim: tuple[float, float] | None = None
                 ) -> tuple[plt.Figure, plt.Axes]:
    """Creates a violinplot of returns.

    Args:
        results: maps model name to results dict

    Returns: figure and axes
    """
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    rows = []
    for label, d in results.items():
        alg = ''
        for m in ['DQN', 'PPO', 'A2C', 'SAC']:
            if m in label:
                alg = label[:label.index(' (')]
                break
        if alg == '':
            alg = label
        alg = '\n'.join(alg.split(' '))

        year = 2019 if '2019' in label else 2021
        # returns = np.sum(d['rewards'], axis=1)
        returns = np.sum(d['results'], axis=1) # check to make sure this makes sense!
        rows.extend([
            (alg, year, r) for r in returns
        ])

    df = pd.DataFrame.from_records(rows, columns=['alg', 'year', 'return'])
    sns.violinplot(data=df, x='alg', y='return', hue='year', ax=ax)
    ax.set(xlabel='Algorithm', ylabel='Return ($)')
    ax.get_legend().set_title('Trained on')

    if ylim is not None:
        ax.set_ylim(*ylim)

    return fig, ax


def setup_episode_plot(env: ElectricityMarketEnv, month_str: str,
                       include_returns: bool, include_bids: bool
                       ) -> tuple[plt.Figure, dict[str, plt.Axes], list[datetime]]:
    """Sets up main plot. Plots demand + carbon cost curves.

    Plots are, in order from top to bottom:
        demand
        price & carbon price
        energy level
        return (aka. cumulative reward, optional)
        bids (optional)

    Args:
        env: already reset to desired day
        month_str: format 'YYYY-MM'
        include_bids: whether to include a final row of bids

    Returns:
        fig: matplotlib Figure
        ax_dict: dict, keys are ['demand', 'prices', 'energy'] and optionally
            include ['rewards', 'bids']. Values are matplotlib Axes
        times: list of datetime
    """
    nrows = 3 + include_returns + include_bids
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 2 * nrows), dpi=200, sharex=True, tight_layout=True)

    day = env.idx + 1
    fig.suptitle(f'Episode: {month_str}-{day:02d}')

    ax_dict = {}
    curr_ax = 0

    # demand
    ax = axs[curr_ax]
    ax_dict['demand'] = ax
    demand_df = env._get_demand_data()
    demand_forecast_df = env._get_demand_forecast_data()
    times = [datetime.strptime(t, '%H:%M') for t in demand_df.columns[:-1]]
    ax.plot(times, demand_df.iloc[env.idx, :-1], label='actual')
    ax.plot(times, demand_forecast_df.iloc[env.idx, :-1], label='forecasted')
    ax.set_ylabel('demand (MWh)')
    lines, labels = ax.get_legend_handles_labels()

    # MOER
    ax = ax.twinx()
    ax.set_ylabel('MOER (kg CO$_2$/kWh)')
    ax.plot(times, env.moer_arr[:-1, 0], color='grey', label='MOER')
    ax.grid(False)
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.2,1))

    curr_ax += 1
    ax = axs[curr_ax]
    ax_dict['prices'] = ax
    ax.set_ylabel('price ($)')

    # energy level
    curr_ax += 1
    ax = axs[curr_ax]
    ax_dict['energy'] = ax
    ax.set_ylabel('energy level (MWh)')
    ax.axhline(y=env.bats_capacity[-1]/2, color='grey')

    # return (aka. cumulative reward)
    if include_returns:
        curr_ax += 1
        ax = axs[curr_ax]
        ax_dict['rewards'] = ax
        ax.set_ylabel('return ($)')
        ax.axhline(y=0, color='grey')

    if include_bids:
        curr_ax += 1
        ax = axs[curr_ax]
        ax_dict['bids'] = ax
        ax.set_ylabel('bid price ($/MWh)')

    fmt = mdates.DateFormatter('%H:%M')
    loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
    for ax in axs:
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(loc)

    return fig, ax_dict, times


def plot_episode(axs: Mapping[str, plt.Axes],
                 times: Sequence[datetime],
                 model_name: str,
                 prices: np.ndarray,
                 energy: np.ndarray,
                 rewards: np.ndarray | None = None,
                 bids: np.ndarray | None = None,
                 **kwargs: Any) -> None:
    """
    Args:
        bids: array of shape [num_steps, 2]
    """
    axs['prices'].plot(times, prices, lw=1, label=model_name)
    axs['energy'].plot(times, energy, label=model_name)
    if 'rewards' in axs and rewards is not None:
        lines = axs['rewards'].plot(times, np.cumsum(rewards), label=model_name)
        axs['rewards'].plot(times[-1], np.sum(rewards), '.', markersize=8, c=lines[0].get_color(), zorder=1000)
    if bids is not None:
        axs['bids'].plot(times, bids[:,0], '.', markersize=2, label=f'{model_name}: buy price')
        axs['bids'].plot(times, bids[:,1], '.', markersize=2, label=f'{model_name}: sell price')
