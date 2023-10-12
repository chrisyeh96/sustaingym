"""
Plotting helper functions
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import seaborn as sns


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

        year = 2019 if '2019' in label else 2020
        returns = np.sum(d['rewards'], axis=1)
        # returns = np.sum(d['results'], axis=1) # check to make sure this makes sense!
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


def setup_episode_plot(env, month_str: str,
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
    # ax = axs[curr_ax]
    # ax_dict['demand'] = ax
    # demand_df = env._get_demand_data()
    # demand_forecast_df = env._get_demand_forecast_data()
    # times = [datetime.strptime(t, '%H:%M') for t in demand_df.columns[:-1]]
    # ax.plot(times, demand_df.iloc[env.idx, :-1], label='actual')
    # ax.plot(times, demand_forecast_df.iloc[env.idx, :-1], label='forecasted')
    # ax.set_ylabel('demand (MWh)')
    # lines, labels = ax.get_legend_handles_labels()

    ax = axs[curr_ax]
    ax_dict['demand'] = ax
    demand_df = env._get_demand_data()
    demand_forecast_df = env._get_demand_forecast_data()

    start = env.idx * env.MAX_STEPS_PER_EPISODE
    start_time = datetime(env.year, env.month, 1, 0, 0)

    times = [start_time + timedelta(minutes=5*step) for step in demand_df['Period']]
    ax.plot(times[:env.MAX_STEPS_PER_EPISODE], demand_df['1'].iloc[start:start + env.MAX_STEPS_PER_EPISODE].values, label='actual')
    ax.plot(times[:env.MAX_STEPS_PER_EPISODE], demand_df['1'].iloc[start:start + env.MAX_STEPS_PER_EPISODE].values, label='forecasted')
    ax.set_ylabel('demand (MWh)')
    lines, labels = ax.get_legend_handles_labels()

    # MOER
    ax = ax.twinx()
    ax.set_ylabel('MOER (kg CO$_2$/kWh)')
    ax.plot(times[:env.MAX_STEPS_PER_EPISODE], env.moer_arr[:-1, 0], color='grey', label='MOER')
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


class ElectricityMarketPlot:
    """Plots data from an ElectricityMarketEnv episode.

    Plots are, in order from top to bottom:
        demand
        price & carbon price
        energy level
        return (aka. cumulative reward, optional)
        bids (optional)

    Args:
        d: desired day
        soc_init: list of length N_B, initial soc of each battery
        include_returns: whether to include returns plot
        include_bids: whether to include a final row of bids
    """
    def __init__(self, d: date, soc_init: Sequence[float], include_returns: bool, include_bids: bool):
        self.d = d
        self.dt = datetime(d.year, d.month, d.day)
        self.N_B = len(soc_init)

        nrows = 3 + include_returns + include_bids
        self.fig, axs = plt.subplots(nrows, 1, figsize=(6, 2 * nrows), sharex=True, tight_layout=True)
        self.fig.suptitle(f'Episode: {d}')
        axs[-1].set(xlabel='time')

        self.axs = {
            'demand': axs[0],
            'moer': axs[0].twinx(),
            'price': axs[1],
            'soc': axs[2]
        }
        self.lines = {}

        ax = self.axs['demand']
        self.lines['demand: actual'], = ax.plot([], [], label='actual')
        self.lines['demand: forecasted'], = ax.plot([], [], label='forecasted')
        ax.set(ylabel='demand (MWh)')

        ax = self.axs['moer']
        self.lines['moer'], = ax.plot([], [], color='grey', label='MOER')
        ax.set(ylabel='MOER (kg CO$_2$/kWh)')

        lines = [self.lines[k] for k in ['demand: actual', 'demand: forecasted', 'moer']]
        ax.legend(lines, [l.get_label() for l in lines])

        ax = self.axs['price']
        for i in range(self.N_B):
            self.lines[f'price {i}'], = ax.plot([], [], label=f'{i}')
        ax.set(ylabel='price ($)')
        ax.legend(title='battery')

        ax = self.axs['soc']
        for i, _ in enumerate(soc_init):
            self.lines[f'soc {i}'], = ax.plot([], [], label=f'{i}')
        ax.set(ylabel='state of charge (MWh)')
        ax.legend(title='battery')

        if include_returns:
            ax = axs[3]
            self.axs['return'] = ax
            self.lines['return'], = ax.plot([], [], label='return')
            ax.set(ylabel='return ($)')

            ax = axs[3].twinx()
            self.axs['reward'] = ax
            self.lines['reward'], = ax.plot([], [], 'C1', label='reward')
            ax.set(ylabel='reward ($)')

            lines = [self.lines[k] for k in ['return', 'reward']]
            ax.legend(lines, [l.get_label() for l in lines])

        if include_bids:
            ax = axs[4]
            self.axs['bids'] = ax
            for i in range(self.N_B):
                self.lines[f'bids: buy price {i}'], = ax.plot([], [], '.', markersize=2, label=f'buy price {i}')
                self.lines[f'bids: sell price {i}'], = ax.plot([], [], '.', markersize=2, label=f'sell price {i}')
            ax.set(ylabel='bid price ($/MWh)')
            ax.legend()

        fmt = mdates.DateFormatter('%H:%M')
        loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
        for ax in axs:
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_major_locator(loc)

    def update(self,
               demand: np.ndarray,
               demand_forecast: np.ndarray,
               moer: np.ndarray,
               prices: np.ndarray,
               soc: np.ndarray,
               rewards: np.ndarray,
               bids: np.ndarray) -> None:
        """
        Args
        - demand: np.array, shape [T]
        - demand_forecast: np.array, shape [T]
        - moer: np.array, shape [T]
        - prices: np.array, shape [T, N_B]
        - soc: np.array, shape [T, N_B]
        - rewards: np.array, shape [T-1]
        - bids: np.array, shape [T-1, 2, N_B]
        """
        T = demand.shape[0]
        assert demand_forecast.shape[0] == T
        assert moer.shape[0] == T
        assert prices.shape[0] == T
        assert soc.shape[0] == T
        assert rewards.shape[0] == T - 1
        assert bids.shape[0] == T - 1

        FIVEMIN = timedelta(minutes=5)
        ts = pd.date_range(self.dt, periods=T, freq=FIVEMIN)

        self.lines['demand: actual'].set_data(ts, demand)
        self.lines['demand: forecasted'].set_data(ts, demand_forecast)
        self.lines['moer'].set_data(ts, moer)
        for i in range(self.N_B):
            self.lines[f'price {i}'].set_data(ts, prices[:, i])
            self.lines[f'soc {i}'].set_data(ts, soc[:, i])

            if 'bids' in self.axs:
                self.lines[f'bids: buy price {i}'].set_data(ts[1:], bids[:, 0, i])
                self.lines[f'bids: sell price {i}'].set_data(ts[1:], bids[:, 1, i])

        if 'return' in self.axs:
            self.lines['reward'].set_data(ts[1:], rewards)
            self.lines['return'].set_data(ts[1:], np.cumsum(rewards))

        for _, ax in self.axs.items():
            ax.relim()
            ax.autoscale_view()
