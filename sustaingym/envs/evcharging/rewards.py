"""
This module contains the method that scores how well the agent is doing.
"""
from __future__ import annotations

from acnportal.acnsim.simulator import Simulator
import numpy as np


CHARGE_WEIGHT = 3
CONSTRAINT_VIOLATION_WEIGHT = 10


def get_rewards(simulator: Simulator, schedule: dict, prev_timestamp: int, timestamp: int, next_timestamp: int) -> float:
    """
    Get reward from charge received by EVs in previous timestep minus costs of
    violation constraints and amount of charge delivered in current timestep.

    Args:
    simulator (Simulator)
    schedule (dictionary) - maps EVSE charger to a single-element list of
        the pilot signal to that charger.
    prev_timestamp (int) - timestamp of previous action taken, needed to
        compute reward from charge received by EVs
    timestamp (int) - timestamp of current action taken
    next_timestamp (int) - timestamp of next action to be taken

    Returns:
    total_reward (float) - total reward awarded to current timestep
    """
    schedule = schedule_to_numpy(schedule)
    # Find charging cost based on amount of charge delivered
    charging_cost = -sum(schedule) * (next_timestamp - timestamp)

    # Get charging reward based on charging rates for previous interval
    charging_reward = (CHARGE_WEIGHT + 1) * np.sum(simulator.charging_rates[:, prev_timestamp:timestamp])

    # Find negative reward for current violation by finding sum of current
    # going over the constraints
    current_sum = np.real(simulator.network.constraint_current(schedule, linear=False))
    magnitudes = simulator.network.magnitudes
    over_current = np.maximum(current_sum - magnitudes, 0)
    constraint_punishment = -CONSTRAINT_VIOLATION_WEIGHT * sum(over_current) * (next_timestamp - timestamp)

    # Get total reward
    total_reward = charging_reward + charging_cost + constraint_punishment
    # normalize by number of charging stations
    total_reward /= len(simulator.network.station_ids)

    return total_reward


def schedule_to_numpy(schedule: dict) -> np.ndarray:
    """
    Convert schedule dictionary to usable numpy array. Helper to get_rewards.

    Args:
    schedule (dictionary) - maps EVSE charger to a single-element list of
        the pilot signal to that charger.

    Returns:
    (np.ndarray) - numpified version of schedule
    """
    return np.array(list(map(lambda x: x[0], schedule.values())))
