from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from typing import Any

import gym
import numpy as np
from tqdm.auto import tqdm


def run_offline_optimal(seeds: Sequence[int], env: gym.Env
                        ) -> dict[str, np.ndarray]:
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

    for i, seed in enumerate(tqdm(seeds)):
        print("seed number: ", i)
        env.reset(seed=seed)
        half = env.bats_capacity[-1] / 2.
        print("calculating baseline no agent prices...")
        ep_prices = env._calculate_prices_without_agent()
        all_results['prices'][i] = ep_prices

        print("calculating optimal...")
        ep_results = env._calculate_price_taking_optimal(
            prices=ep_prices, init_charge=half, final_charge=half)
        for k in ['rewards', 'net_prices', 'energy', 'dispatch']:
            all_results[k][i] = ep_results[k]

    return all_results


def run_follow_offline_optimal(
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
    assert opt_dispatches.shape[0] == num_eps
    assert opt_energies.shape[0] == num_eps

    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energy = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    max_price = env.action_space.high[0]
    charge_action = (max_price, max_price)
    discharge_action = (0.01*max_price, 0.01*max_price)
    no_action = (0, max_price)

    for ep, seed in enumerate(tqdm(seeds)):
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

def congested_run_follow_offline_optimal(
        seeds: Sequence[int], env: gym.Env,
        opt_dispatches: np.ndarray,
        opt_energies: np.ndarray
        ) -> dict[str, np.ndarray]:
    """Get offline optimal reward for a number of episodes in congestion case.

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
    assert opt_dispatches.shape[0] == num_eps
    assert opt_energies.shape[0] == num_eps

    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energy = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    max_price = env.action_space.high[0][0, 0]

    charge_action = np.zeros((2,1))
    charge_action[:, 0] = np.array([max_price, max_price])
    charge_action = np.array([[*charge_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

    discharge_action = np.zeros((2,1))
    discharge_action[:, 0] = np.array([0.001*max_price, 0.001*max_price])
    discharge_action = np.array([[*discharge_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

    no_action = np.zeros((2,1))
    no_action[:, 0] = np.array([0, max_price])
    no_action = np.array([[*no_action], ] * (env.settlement_interval+1)).transpose().reshape(2,1,(env.settlement_interval+1))

    for ep, seed in enumerate(tqdm(seeds)):
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


def run_random(seeds: Sequence[int], env: gym.Env, discrete: bool) -> dict[str, np.ndarray]:
    print("here")
    num_eps = len(seeds)
    rewards = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    energy = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))
    prices = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE))

    if discrete:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE), dtype=np.int32)
    else:
        actions = np.zeros((num_eps, env.MAX_STEPS_PER_EPISODE, 2))

    for ep, seed in tqdm(enumerate(seeds)):
        print("ep: ", ep)
        obs = env.reset(seed=seed)
        prices[ep, 0] = obs['price previous'][0]
        energy[ep, 0] = obs['energy'][0]
        for i in range(1, env.MAX_STEPS_PER_EPISODE):
            action = env.action_space.sample()
            obs, reward, _, _ = env.step(action)
            rewards[ep, i] = reward
            energy[ep, i] = obs['energy'][0]
            prices[ep, i] = obs['price previous'][0]
            
            if discrete:
                actions[ep, i] = action
            else:
                actions[ep, i] = action[:, 0, 0]
    
    return {
        'rewards': rewards,
        'prices': prices,
        'energy': energy,
        'actions': actions
    }


def run_model(
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

    for ep, seed in enumerate(tqdm(seeds)):
        obs = env.reset(seed=seed)
        energies[ep, 0] = env.battery_charge[-1]
        prices[ep, 0] = obs['price previous'][0]

        for step in range(1, env.MAX_STEPS_PER_EPISODE):
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


def save_results(results: Mapping[str, np.ndarray],
                 path: str,
                 seeds: Sequence[int] | None = None
                 ) -> None:
    if seeds is not None:
        num_eps = len(seeds)
        for arr in results.values():
            assert arr.shape[0] == num_eps

    if os.path.exists(path):
        print(f'File already exists at {path}! Not overwriting. Quitting.')
        return

    if seeds is None:
        np.savez_compressed(path, **results)
    else:
        np.savez_compressed(path, **results, seeds=seeds)
