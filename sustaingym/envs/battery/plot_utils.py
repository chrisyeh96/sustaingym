"""
Plotting helper functions
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import gym
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

from sustaingym.envs import ElectricityMarketEnv


def training_eval_results(model: str, dist: str):
    assert model in ['PPO', 'A2C', 'DQN']
    assert dist in ['in_dist', 'out_dist']
    results = []
    fname = f'examples/discrete_logs_{model}/{dist}/evaluations.npz'
    x = np.load(fname, allow_pickle=True)
    results.append(x['results'])

    timesteps = x['timesteps']
    results = np.hstack(results)
    y = results.mean(axis=1)
    error = results.std(axis=1)
    return timesteps, y, error


def run_model_for_evaluation(
        model: Any,
        episodes: int,
        env: gym.Env,
        add_soc_and_prices: bool = False,
        per_time_step: bool = False,
        include_carbon_costs: bool = False):
    """
    Run a model for a number of episodes and return the mean and std of the reward.

    Args:
        model: (BaseRLModel object) the model to evaluate
        episodes: (int) number of episodes to evaluate for
        env: (Gym Environment) the environment to evaluate the model on

    Returns:
        rewards for episodes
    """
    episode_rewards = []
    prices = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    charges = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    carbon_costs = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    all_rewards = []

    for i in range(episodes):
        obs = env.reset(seed=i*10)
        charges[i, 0] = env.battery_charge[-1]
        done = False
        rewards = np.zeros(env.MAX_STEPS_PER_EPISODE)

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards[env.count] = reward
            prices[i, env.count] = info['price']
            charges[i, env.count] = env.battery_charge[-1]
            carbon_costs[i, env.count] = info['carbon reward']
        prices[i, 0] = prices[i, 1]
        episode_rewards.append(np.sum(rewards))
        all_rewards.append(rewards)

    if not include_carbon_costs:
        if not per_time_step:
            return episode_rewards if not add_soc_and_prices else (episode_rewards, prices, charges)

        return all_rewards if not add_soc_and_prices else (episode_rewards, prices, charges)
    else:
        if not per_time_step:
            return episode_rewards, carbon_costs  if not add_soc_and_prices else (episode_rewards, prices, charges)

        return all_rewards, carbon_costs if not add_soc_and_prices else (episode_rewards, prices, charges)


def get_offline_optimal(seeds: Sequence[int], env: gym.Env
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get offline optimal reward for a number of episodes.

    Args:
        seeds: seeds for environment reset, length is number of episodes to evaluate for
        env: environment to evaluate the model on

    Returns:
        rewards: list of array, array is rewards from an episode, shape [num_steps]
        prices: list of array, array is prices from an episode, shape [num_steps]
        net_prices: list of array, array is net prices from an episode, shape [num_steps]
        energies: list of array, array is energy level from an episode, shape [num_steps]
    """
    rewards = []
    prices = []
    net_prices = []
    energies = []
    dispatches = []

    for seed in seeds:
        env.reset(seed=seed)
        print('Day of month:', env.idx + 1)
        half = env.battery_capacity[-1] / 2.
        ep_prices = env._calculate_prices_without_agent()
        ep_rewards, dispatch, energy, net_price = env._calculate_price_taking_optimal(
            prices=ep_prices, init_charge=half, final_charge=half)
        ep_prices[0] = ep_prices[1]
        rewards.append(ep_rewards)
        prices.append(ep_prices)
        net_prices.append(net_price)
        energies.append(energy)
        dispatches.append(dispatch)

    return rewards, prices, net_prices, energies, dispatches


def get_random_action_rewards(episodes, env):
    episode_rewards = []
    for i in range(episodes):
        obs = env.reset(seed = i*10)
        done = False
        rewards = np.zeros(env.MAX_STEPS_PER_EPISODE)
        while not done:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            rewards[env.count] = reward
        episode_rewards.append(np.sum(rewards))
    return episode_rewards


def get_offline_time_step_rewards(env):
    env.reset(seed=0)
    prices = env._calculate_prices_without_agent()
    _, dispatches = env._calculate_price_taking_optimal(prices=prices,
            init_charge=env.battery_charge[-1], final_charge=env.battery_capacity[-1] / 2.)
    rewards = []
    rewards.append(0.)
    carbon_rewards = []
    carbon_rewards.append(0.)
    moers = env.moer_arr[:, 0]

    for i in range(1, env.MAX_STEPS_PER_EPISODE):
        rewards.append((prices[i] + env.CARBON_PRICE * moers[i]) * dispatches[i-1])
        carbon_rewards.append(env.CARBON_PRICE * moers[i] * dispatches[i-1])

    return rewards, carbon_rewards


def plot_model_training_reward_curves(ax: plt.Axes, model: str,
    dists: Sequence[str]) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    evals_lst = []
    err_lst = []

    for dist in dists:
        timesteps, y, err = training_eval_results(model, dist)
        evals_lst.append(y)
        err_lst.append(err)

    ax.plot(timesteps, evals_lst[0], label='train on May 2019, evaluate on May 2019') # too specific!!!
    ax.fill_between(timesteps, evals_lst[0]-err_lst[0], evals_lst[0]+err_lst[0], alpha=0.2)

    if len(dists) == 2:
        ax.plot(timesteps, evals_lst[1], label='train on May 2019, evaluate on May 2021') # too specific!!!
        ax.fill_between(timesteps, evals_lst[1]-err_lst[1], evals_lst[1]+err_lst[1], alpha=0.2)

    ax.set_title(f'{model} Training Reward Curves')
    ax.set_ylabel('reward ($)')
    ax.set_xlabel('timesteps')
    ax.legend()

    return ax


def plot_reward_distribution(
        ax: plt.Axes,
        eval_env: gym.Env,
        models: Sequence,
        model_labels: Sequence[str],
        n_episodes = 10,
        year = str) -> plt.Axes:
    """"
    Plot reward distributions for a given evaluation and training month where year is
    variable and month is fixed at May (for now).

    Args:
        ax: plt axes object to capture resulting reward distribution in a plot
        eval_env: gym environment representing the env to evaluate models on
        models: RL models trained on electricity market gym
        model_labels: corresponding labels for the "models" arg
        n_episodes: integer for number of episodes to evaluate over
    Return:
        populated plt axes object which plots the reward distribution

    Notes:
        Use "plt.xticks(rotation=30)" to rotate x axis labels for a more appealing
        orientation
    """
    assert len(models) == len(model_labels)

    if ax is None:
        fig, ax = plt.subplots()

    offline_rewards, _, _= get_offline_optimal(n_episodes, eval_env)
    random_rewards = get_random_action_rewards(n_episodes, eval_env)

    model_rewards = []

    for model in models:
        model_rewards.append(run_model_for_evaluation(model, n_episodes, eval_env))

    data = [offline_rewards, random_rewards]
    data.extend(model_rewards)
    labels = ['', 'offline', 'random']
    labels.extend(model_labels)

    ax.violinplot(data, showmedians=True)

    ax.set_ylabel('Reward ($)')
    ax.set_title(f'Reward Distribution for May {year}')
    x = np.arange(len(labels))  # the label locations

    ax.set_xticks(x, labels)
    fig.tight_layout()

    return ax


def plot_state_of_charge_and_prices(
        env, load_data: pd.DataFrame,
        model: Any | None = None, model_label: str | None = None,
        axs: Sequence[plt.Axes] | None = None
        ) -> tuple[plt.Axes, plt.Axes]:
    """Plots prices and battery energy level over time.

    Creates two axes. Top axes is prices (raw electricity price and net price),
    bottom axes is battery energy level.

    Args:
        env: ElectricityMarketEnv
        load_data:
        model: optional
        model_label: label for model, only needed if model is given
        axs: list of two plt.Axes
    """
    if axs is None:
        _, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
    assert len(axs) == 2
    ax, ax2 = axs

    time_arr = load_data.columns[:-1]
    timeArray = [datetime.strptime(t, '%H:%M') for t in time_arr]

    if model is not None:
        assert model_label is not None
        _, prices, charges = run_model_for_evaluation(model, 1, env, True)
        prices = np.reshape(prices, (env.MAX_STEPS_PER_EPISODE,))
        charges = np.reshape(charges, (env.MAX_STEPS_PER_EPISODE,))
        ax.plot(timeArray, prices, label=model_label)
        ax2.plot(timeArray, charges, label=model_label)

    _, offline_prices, offline_net_prices, offline_charges = get_offline_optimal(episodes=1, env=env)
    offline_prices = offline_prices[0]
    offline_net_prices = offline_net_prices[0]
    offline_charges = offline_charges[0]
    ax.plot(timeArray, offline_prices, label='offline prices')
    ax.plot(timeArray, offline_net_prices, label='offline net prices')
    ax2.plot(timeArray, offline_charges, label='offline charges')

    ax.set_ylabel('Prices ($)')
    ax2.set(xlabel='time', ylabel='State of Charge (MWh)', ylim=(0, 120))

    fmt = mdates.DateFormatter('%H:%M')
    loc = plticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    for ax in axs:
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(loc)
        ax.legend()

    return ax, ax2


def setup_episode_plot(env: ElectricityMarketEnv, month_str: str, include_bids: bool
                       ) -> tuple[list[plt.Axes], list[datetime]]:
    """Sets up main plot. Plots demand + carbon cost curves.

    Args:
        env: already reset to desired day
        month_str: format 'YYYY-MM'
        include_bids: whether to include a final row of bids
    """
    nrows = 4
    if include_bids:
        nrows += 1
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 8), dpi=200, sharex=True, tight_layout=True)

    day = env.idx + 1
    fig.suptitle(f'Episode: {month_str}-{day:02d}', y=1.02)

    # demand
    ax = axs[0]
    demand_df = env._get_demand_data()
    demand_forecast_df = env._get_demand_forecast_data()
    times = [datetime.strptime(t, '%H:%M') for t in demand_df.columns[:-1]]
    ax.plot(times, demand_df.iloc[env.idx, :-1], label='demand')
    ax.plot(times, demand_forecast_df.iloc[env.idx, :-1], label='forecasted demand')
    ax.set_ylabel('demand (MWh)')

    # carbon price
    ax = axs[1]
    ax.plot(times, env.moer_arr[:-1, 0] * env.CARBON_PRICE, color='grey', label=r'CO$_2$ price')
    ax.set_ylabel('price ($)')

    # energy level
    axs[2].set_ylabel('energy level (MWh)')
    axs[2].axhline(y=env.battery_capacity[-1]/2, color='grey')

    # return (aka. cumulative reward)
    axs[3].set_ylabel('return ($)')
    axs[3].axhline(y=0, color='grey')

    if include_bids:
        axs[4].set_ylabel('bid price ($/MWh)')

    fmt = mdates.DateFormatter('%H:%M')
    loc = plticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    for ax in axs:
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(loc)

    return fig, axs, times


def plot_episode(axs: Sequence[plt.Axes],
                 times: Sequence[datetime],
                 model_name: str,
                 prices: np.ndarray,
                 energy_level: np.ndarray,
                 rewards: np.ndarray,
                 bids: np.ndarray | None = None) -> None:
    """
    Args:
        bids: array of shape [num_steps, 2]
    """
    axs[1].plot(times, prices, label=model_name)
    axs[2].plot(times, energy_level, label=model_name)
    axs[3].plot(times, np.cumsum(rewards), label=model_name)
    if bids is not None:
        axs[4].plot(times, bids[:,0], label=f'{model_name}: buy price')
        axs[4].plot(times, bids[:,1], '.', label=f'{model_name}: sell price')


def plot_reward_over_episode(axs: Sequence[plt.Axes], model, env) -> plt.Axes:
    if axs is None:
        fig, axs = plt.subplots(2)

    assert len(axs) == 2
    ax, ax2 = axs
    delta_time = datetime.timedelta(minutes=5)
    curr_time = datetime(2022, 1, 1, 0, 0, 0, 0)
    times = []

    for _ in range(env.MAX_STEPS_PER_EPISODE):
        times.append(curr_time.strftime('%H:%M:%S'))
        curr_time += delta_time

    ppo_reward, ppo_carbon_costs = run_model_for_evaluation(model, 1, env, False, True, True)
    ppo_reward = np.array(ppo_reward[0])
    offline_reward, offline_carbon_costs = get_offline_time_step_rewards(env)
    offline_reward = np.array(offline_reward)

    cum_ppo_reward = np.cumsum(ppo_reward)
    cum_offline_reward = np.cumsum(offline_reward)
    cum_ppo_carbon_costs = np.cumsum(ppo_carbon_costs[0])
    cum_offline_carbon_costs = np.cumsum(offline_carbon_costs)

    ax.plot(times, cum_ppo_reward, label="ppo")
    ax.plot(times, cum_ppo_carbon_costs, label="ppo carbon costs")
    ax2.plot(times, cum_offline_reward, label='offline optimal')
    ax2.plot(times, cum_offline_carbon_costs, label='offline carbon costs')

    # naming the x axis
    ax2.set_xlabel('time')
    # naming the y axis
    ax.set_ylabel('reward ($)')

    for ax in axs:
        ax.legend()
        ax.set_xticks(times[::50])
        ax.set_xticklabels(times[::50])

    return ax, ax2
