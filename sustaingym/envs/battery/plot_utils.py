"""
Plotting helper functions
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from tqdm.auto import tqdm

import gym
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from sustaingym.envs import ElectricityMarketEnv
from sustaingym.envs.battery.wrapped import DiscreteActions


def training_eval_results(model: str, dist: str):
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
        seeds: Sequence[int],
        env: gym.Env,
        discrete: bool):
    """Run a model for a number of episodes and return results.

    Args:
        model: (BaseRLModel object) the model to evaluate
        episodes: (int) number of episodes to evaluate for
        env: environment to evaluate the model on
        discrete: flag for whether actions are discrete

    Returns: results dict, keys contain ['rewards', 'energies', 'prices', 'carbon rewards', 'actions'],
        values are arrays of shape [episodes, num_steps].
        If discrete, then contains key
            'actions' has shape [episodes, num_steps], type int32
        Otherwise, contains key
            'actions' has shape [episodes, num_steps, 2], type float64
    """
    num_eps = len(seeds)
    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energies = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    carbon_rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    if discrete:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE), dtype=np.int32)
    else:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE, 2))

    for ep, seed in enumerate(seeds):
        obs = env.reset(seed=ep*10)
        energies[ep, 0] = env.battery_charge[-1]
        prices[ep, 0] = obs['price previous'][0]

        for step in tqdm(range(1, env.MAX_STEPS_PER_EPISODE)):
            action, _ = model.predict(obs)
            if discrete:
                action = action.item()
            actions[ep, step] = action
            obs, reward, done, info = env.step(action)
            rewards[ep, step] = reward
            prices[ep, step] = obs['price previous'][0]
            energies[ep, step] = env.battery_charge[-1]
            carbon_rewards[ep, step] = info['carbon reward']
        assert done

    return {
        'rewards': rewards,
        'prices': prices,
        'energy': energies,
        'carbon rewards': carbon_rewards,
        'actions': actions
    }


def get_follow_offline_optimal(
        seeds: Sequence[int], env: gym.Env,
        opt_dispatches: np.ndarray,
        opt_energies: np.ndarray
        ) -> dict[str, np.ndarray]:
    """Get offline optimal reward for a number of episodes.

    Args:
        seeds: seeds for environment reset, length is number of episodes to evaluate for
        env: environment to evaluate the model on
        opt_dispatches: array of dispatches from the optimal agent, shape [len(seeds), num_steps]
        opt_energies: array of energy levels from the optimal agent, shape [len(seeds), num_steps]

    Returns:
        results dict, keys are ['rewards', 'prices', 'energy'],
            values are arrays of shape [num_episodes, num_steps]
    """
    num_eps = len(seeds)
    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energy = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    max_price = env.action_space.high[0]
    charge_action = (max_price, max_price)
    discharge_action = (0.01*max_price, 0.01*max_price)
    no_action = (0, max_price)

    for ep, seed in enumerate(seeds):
        obs = env.reset(seed=seed)
        energy[ep, 0] = obs['energy'][0]
        prices[ep, 0] = obs['price previous'][0]

        for i in range(1, 288):
            action = no_action
            if (opt_dispatches[ep, i] < -0.1) and (energy[ep, i-1] < opt_energies[ep, i]):
                action = charge_action
            elif (opt_dispatches[ep, i] > 0.1) and (energy[ep, i-1] > opt_energies[ep, i]):
                action = discharge_action
            obs, reward, _, _ = env.step(action)
            rewards[ep, i] = reward
            energy[ep, i] = obs['energy'][0]
            prices[ep, i] = obs['price previous'][0]

    return {
        'rewards': rewards,
        'prices': prices,
        'energy': energy
    }


def get_offline_optimal(seeds: Sequence[int], env: gym.Env
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get offline optimal reward for a number of episodes.

    Args:
        seeds: seeds for environment reset, length is number of episodes to evaluate for
        env: environment to evaluate the model on

    Returns:
        results dict, keys are ['rewards', 'prices', 'net_prices', 'energy', 'dispatch'],
            values are arrays of shape [num_episodes, num_steps]
    """
    num_episodes = len(seeds)
    all_results = {
        k: np.zeros((num_episodes, env.MAX_STEPS_PER_EPISODE))
        for k in ['rewards', 'prices', 'net_prices', 'energy', 'dispatch']
    }

    for i, seed in enumerate(seeds):
        env.reset(seed=seed)
        half = env.bats_capacity[-1] / 2.
        ep_prices = env._calculate_prices_without_agent()
        all_results['prices'][i] = ep_prices

        ep_results = env._calculate_price_taking_optimal(
            prices=ep_prices, init_charge=half, final_charge=half)
        for k in ['rewards', 'net_prices', 'energy', 'dispatch']:
            all_results[k][i] = ep_results[k]

    return all_results


def get_random(seeds: Sequence[int], env: gym.Env, discrete: bool) -> list[float]:
    num_eps = len(seeds)
    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energy = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    if discrete:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE), dtype=np.int32)
    else:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE, 2))

    for ep, seed in enumerate(seeds):
        env.reset(seed=seed)
        for i in tqdm(range(1, env.MAX_STEPS_PER_EPISODE)):
            action = env.action_space.sample()
            obs, reward, _, _ = env.step(action)
            rewards[ep, i] = reward
            energy[ep, i] = obs['energy'][0]
            prices[ep, i] = obs['price previous'][0]
            actions[ep, i] = action
    return {
        'rewards': rewards,
        'prices': prices,
        'energy': energy,
        'actions': actions
    }


def get_offline_time_step_rewards(env: ElectricityMarketEnv) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=0)
    prices = env._calculate_prices_without_agent()
    results = env._calculate_price_taking_optimal(
        prices=prices, init_charge=env.battery_charge[-1],
        final_charge=env.battery_capacity[-1] / 2.)
    carbon_rewards = env.CARBON_PRICE * env.moer_arr[:, 0] * results['dispatch']
    return results['rewards'], carbon_rewards


def plot_model_training_reward_curves(
        ax: plt.Axes, model: str, dists: Sequence[str]) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()

    evals_lst = []
    err_lst = []

    for dist in dists:
        timesteps, y, err = training_eval_results(model, dist)
        evals_lst.append(y)
        err_lst.append(err)

    ax.plot(timesteps, evals_lst[0], label='train on May 2019, evaluate on May 2019')  # too specific!!!
    ax.fill_between(timesteps, evals_lst[0]-err_lst[0], evals_lst[0]+err_lst[0], alpha=0.2)

    if len(dists) == 2:
        ax.plot(timesteps, evals_lst[1], label='train on May 2019, evaluate on May 2021')  # too specific!!!
        ax.fill_between(timesteps, evals_lst[1]-err_lst[1], evals_lst[1]+err_lst[1], alpha=0.2)

    ax.set_title(f'{model} Training Reward Curves')
    ax.set_ylabel('reward ($)')
    ax.set_xlabel('timesteps')
    ax.legend()

    return ax


def get_returns(
        env: gym.Env,
        discrete_env: gym.Env | None,
        models: Sequence,
        model_labels: Sequence[str],
        n_episodes: int = 10,
        run_base: Sequence[str] = ('offline', 'follow', 'random', 'random_discrete'),
        ) -> plt.Axes:
    """"Get returns.

    Plot reward distributions for a given evaluation and training month where year is
    variable and month is fixed at May (for now).

    Args:
        ax: plt axes object to capture resulting reward distribution in a plot
        env: gym environment representing the env to evaluate models on
        models: RL models trained on electricity market gym
        model_labels: corresponding labels for the "models" arg
        n_episodes: integer for number of episodes to evaluate over

    Returns:
        populated plt axes object which plots the reward distribution

    Notes:
        Use "plt.xticks(rotation=30)" to rotate x axis labels for a more appealing
        orientation
    """
    seeds = np.arange(n_episodes)
    data, labels = [], []

    if 'offline' in run_base:
        print('Running offline optimal')
        opt_results = get_offline_optimal(seeds=seeds, env=env)
        opt_returns = np.sum(opt_results['rewards'], axis=1)
        data.append(opt_returns)
        labels.append('offline')

    if 'follow' in run_base:
        print('Running follow offline optimal')
        follow_results = get_follow_offline_optimal(
            seeds=seeds, env=env,
            opt_dispatches=opt_results['dispatch'],
            opt_energies=opt_results['energy'])
        follow_returns = np.sum(follow_results['rewards'], axis=1)
        data.append(follow_returns)
        labels.append('follow offline')

    if 'random' in run_base:
        print('Running random')
        random_results = get_random(seeds, env, discrete=False)
        random_returns = np.sum(random_results['rewards'], axis=1)
        data.append(random_returns)
        labels.append('random')

    if 'random_discrete' in run_base:
        print('Running random (discrete)')
        discrete_random_results = get_random(seeds, discrete_env, discrete=True)
        discrete_random_returns = np.sum(discrete_random_results['rewards'], axis=1)
        data.append(discrete_random_returns)
        labels.append('random (discrete)')

    for model_label, model in zip(model_labels, models):
        print(f'Running {model_label}')
        discrete = isinstance(model.action_space, gym.spaces.Discrete)
        if discrete:
            results = run_model_for_evaluation(model, seeds, discrete_env, discrete)
        else:
            results = run_model_for_evaluation(model, seeds, env, discrete)
        data.append(np.sum(results['rewards'], axis=1))
        labels.append(model_label)

    return data, labels


def plot_returns(data, labels):
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    ax.violinplot(data, showmedians=True)
    ax.set(ylabel='Return ($)')
    x = np.arange(len(labels))  # the label locations
    ax.set_xticks(x, labels, rotation=45)
    return fig, ax


def plot_state_of_charge_and_prices(
        env: ElectricityMarketEnv, load_data: pd.DataFrame,
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
        results = run_model_for_evaluation(model, 1, env)
        prices = results['prices'][0]
        charges = results['energies'][0]
        ax.plot(timeArray, prices, label=model_label)
        ax2.plot(timeArray, charges, label=model_label)

    offline_results = get_offline_optimal(seeds=[0], env=env)
    ax.plot(timeArray, offline_results['prices'][0], label='offline prices')
    ax.plot(timeArray, offline_results['net_prices'][0], label='offline net prices')
    ax2.plot(timeArray, offline_results['energy'][0], label='offline charges')

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
                       ) -> tuple[plt.Figure, list[plt.Axes], list[datetime]]:
    """Sets up main plot. Plots demand + carbon cost curves.

    Plots are, in order from top to bottom:
        demand
        price & carbon price
        energy level
        return (aka. cumulative reward)
        bids (optional)

    Args:
        env: already reset to desired day
        month_str: format 'YYYY-MM'
        include_bids: whether to include a final row of bids
    """
    nrows = 4
    if include_bids:
        nrows += 1
    fig, axs = plt.subplots(nrows, 1, figsize=(6, 1.9 * nrows), dpi=200, sharex=True, tight_layout=True)

    day = env.idx + 1
    fig.suptitle(f'Episode: {month_str}-{day:02d}')

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
    axs[2].axhline(y=env.bats_capacity[-1]/2, color='grey')

    # return (aka. cumulative reward)
    axs[3].set_ylabel('return ($)')
    axs[3].axhline(y=0, color='grey')

    if include_bids:
        axs[4].set_ylabel('bid price ($/MWh)')

    fmt = mdates.DateFormatter('%H:%M')
    loc = plticker.MultipleLocator(base=0.25)  # this locator puts ticks at regular intervals
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
                 bids: np.ndarray | None = None,
                 **kwargs: Any) -> None:
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


def plot_reward_over_episode(model: Any, env: ElectricityMarketEnv, axs: Sequence[plt.Axes] | None = None) -> plt.Axes:
    if axs is None:
        fig, axs = plt.subplots(2)

    assert len(axs) == 2
    ax, ax2 = axs
    delta_time = timedelta(minutes=5)
    curr_time = datetime(2022, 1, 1, 0, 0, 0, 0)
    times = []

    for _ in range(env.MAX_STEPS_PER_EPISODE):
        times.append(curr_time.strftime('%H:%M'))
        curr_time += delta_time

    ppo_results = run_model_for_evaluation(model, 1, env)
    offline_reward, offline_carbon_costs = get_offline_time_step_rewards(env)
    offline_reward = np.array(offline_reward)

    cum_ppo_reward = np.cumsum(ppo_results['rewards'][0])
    cum_offline_reward = np.cumsum(offline_reward)
    cum_ppo_carbon_costs = np.cumsum(ppo_results['carbon rewards'][0])
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
