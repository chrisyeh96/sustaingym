"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
import os
import pkgutil
import pytz
from typing import Any, List

import cvxpy as cp
from gym import Env, spaces
from matplotlib.axes._axes import Axes as MplAxes
import matplotlib.pyplot as plt
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

def run_model_for_evaluation(model, episodes, env):
    """
    Run a model for a number of episodes and return the mean and std of the reward.
    :param model: (BaseRLModel object) the model to evaluate
    :param episodes: (int) number of episodes to evaluate for
    :param env: (Gym Environment) the environment to evaluate the model on
    :return: (np.ndarray) rewards for episodes
    """
    episode_rewards = []
    for i in range(episodes):
        obs = env.reset(seed = i*10)
        done = False
        rewards = np.zeros(env.MAX_STEPS_PER_EPISODE)
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards[env.count] = reward
        episode_rewards.append(np.sum(rewards))
    return episode_rewards

def get_offline_optimal(episodes, env):
    """
    Get the offline optimal reward for a number of episodes.
    :param episodes: (int) number of episodes to evaluate for
    :param env: (Gym Environment) the environment to evaluate the model on
    :return: (np.ndarray) rewards for episodes
    """
    episode_rewards = []
    episode_actions = []

    for i in range(episodes):
        obs = env.reset(seed = i*10)
        rewards, actions = env._calculate_realistic_off_optimal_total_episode_reward()
        episode_rewards.append(rewards)
        episode_actions.append(actions)
    return episode_rewards, episode_actions

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
    
    ax.plot(timesteps, evals_lst[0], label='train on May 2019, evaluate on May 2019')
    ax.fill_between(timesteps, evals_lst[0]-err_lst[0], evals_lst[0]+err_lst[0], alpha=0.2)

    ax.plot(timesteps, evals_lst[1], label='train on May 2019, evaluate on May 2021')
    ax.fill_between(timesteps, evals_lst[1]-err_lst[1], evals_lst[1]+err_lst[1], alpha=0.2)

    ax.set_title(f'{model} Training Reward Curves')
    ax.set_ylabel('reward ($)')
    ax.set_xlabel('timesteps')
    ax.legend()

    return ax

