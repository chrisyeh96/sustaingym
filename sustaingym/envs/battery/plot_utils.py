"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from io import BytesIO
import os
import pkgutil
from socket import AF_AX25
import pytz
from typing import Any, List

import cvxpy as cp
import datetime
from gym import Env, spaces
from matplotlib.axes._axes import Axes as MplAxes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C


def training_eval_results(model: str, dist: str):
    assert model in ['PPO', 'A2C']
    assert dist in ['in_dist', 'out_dist']
    results = []
    fname = f'examples/logs_{model}/{dist}/evaluations.npz'
    x = np.load(fname, allow_pickle=True)
    results.append(x['results'])

    timesteps = x['timesteps']
    results = np.hstack(results)
    y = results.mean(axis=1)
    error = results.std(axis=1)
    return timesteps, y, error

def run_model_for_evaluation(model, episodes, env, add_soc_and_prices: bool = False,
    per_time_step: bool = False, include_carbon_costs: bool = False):
    """
    Run a model for a number of episodes and return the mean and std of the reward.
    :param model: (BaseRLModel object) the model to evaluate
    :param episodes: (int) number of episodes to evaluate for
    :param env: (Gym Environment) the environment to evaluate the model on
    :return: (np.ndarray) rewards for episodes
    """
    episode_rewards = []
    prices = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    charges = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    carbon_costs = np.zeros((episodes, env.MAX_STEPS_PER_EPISODE))
    all_rewards = []
    
    for i in range(episodes):
        obs = env.reset(seed = i*10)
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

def get_offline_optimal(episodes, env):
    """
    Get the offline optimal reward for a number of episodes.
    :param episodes: (int) number of episodes to evaluate for
    :param env: (Gym Environment) the environment to evaluate the model on
    :return: (np.ndarray) rewards for episodes
    """
    episode_rewards = []
    episode_charges = []
    episode_prices = []

    for i in range(episodes):
        obs = env.reset(seed = i*10)
        init_soc = env.battery_charge[-1]
        rewards, dispatches, prices = env._calculate_realistic_off_optimal_total_episode_reward()
        prices[0] = prices[1]
        episode_rewards.append(rewards)
        episode_prices.append(prices)
        
        curr_soc = init_soc
        charges = np.zeros(env.MAX_STEPS_PER_EPISODE)

        for i in range(len(dispatches)):
            charges[i] = curr_soc + -1. * dispatches[i]
            curr_soc += -1. * dispatches[i]

        episode_charges.append(charges)
        
    return episode_rewards, episode_prices, episode_charges

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
    _, dispatches, prices = env._calculate_realistic_off_optimal_total_episode_reward()
    rewards = []
    rewards.append(0.)
    carbon_costs = []
    carbon_costs.append(0.)
    moers = env.moer_arr[:, 0]

    for i in range(1, env.MAX_STEPS_PER_EPISODE):
        rewards.append((prices[i] + env.CARBON_COST * moers[i]) * dispatches[i-1])
        carbon_costs.append(env.CARBON_COST * moers[i] * dispatches[i-1])
    
    return rewards, carbon_costs

def plot_model_training_reward_curves(ax: MplAxes, model: str,
    dists: List[str]) -> MplAxes:
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

def plot_reward_distribution(ax: MplAxes, eval_env, models, model_labels, n_episodes = 10,
    year = str) -> MplAxes:
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

def plot_state_of_charge_and_prices(axes: List[MplAxes], load_data: pd.DataFrame, model,
    model_label, env) -> tuple[MplAxes, MplAxes]:

    if axes is None:
        fig, (ax, ax2) = plt.subplots(2)
        axes = (ax, ax2)
    
    assert len(axes) == 2
    ax, ax2 = axes
    fmt = mdates.DateFormatter('%H:%M:%S')

    time_arr = load_data.columns[:-1]
    time_arr = [t + ':00.0' for t in time_arr]
    timeArray = [datetime.datetime.strptime(i, '%H:%M:%S.%f') for i in time_arr]

    _, prices, charges = run_model_for_evaluation(model, 1, env, True)
    _, offline_prices, offline_charges = get_offline_optimal(1, env)

    prices = np.reshape(prices, (env.MAX_STEPS_PER_EPISODE,))
    offline_prices = np.reshape(offline_prices, (env.MAX_STEPS_PER_EPISODE,))
    charges = np.reshape(charges, (env.MAX_STEPS_PER_EPISODE,))
    offline_charges = np.reshape(offline_charges, (env.MAX_STEPS_PER_EPISODE,))

    ax.plot(timeArray, prices, label=model_label)
    ax.plot(timeArray, offline_prices, label='offline prices')
    ax2.plot(timeArray, charges, label=model_label)
    ax2.plot(timeArray, offline_charges, label='offline charges')

    # naming the x axis 
    ax.set_xlabel('time')
    # naming the y axis 
    ax.set_ylabel('Prices ($)')
    ax2.set_ylabel('State of Charge (MWh)')

    ax.xaxis.set_major_formatter(fmt)
    loc = plticker.MultipleLocator(base=0.2) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    ax2.xaxis.set_major_formatter(fmt)
    ax2.xaxis.set_major_locator(loc)

    ax.legend()
    ax2.legend()

    return ax, ax2

def plot_reward_over_episode(axes: List[MplAxes], model, env) -> MplAxes:
    if axes is None:
        fig, (ax, ax2) = plt.subplots(2)
        axes = (ax, ax2)
    
    assert len(axes) == 2
    ax, ax2 = axes
    delta_time = datetime.timedelta(minutes=5)
    curr_time = datetime.datetime(2022, 1, 1, 0, 0, 0, 0)
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

    ax.legend()
    ax2.legend()

    ax.set_xticks(times[::50])
    ax.set_xticklabels(times[::50])

    ax2.set_xticks(times[::50])
    ax2.set_xticklabels(times[::50])

    return ax, ax2

