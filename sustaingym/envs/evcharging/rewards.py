"""
This module contains the method that scores how well the agent is doing.
"""
from __future__ import annotations

from typing import Any

from acnportal.acnsim.simulator import Simulator
import numpy as np


CHARGE_COST_WEIGHT = 1
DELIVERED_CHARGE_WEIGHT = 3
CONSTRAINT_VIOLATION_WEIGHT = 10


def get_rewards(simulator: Simulator,
                schedule: dict,
                prev_timestamp: int,
                timestamp: int,
                next_timestamp: int,
                get_info: bool = True
                ) -> float | tuple[float, dict[str, Any]]:
    """
    Return reward for scheduler's performance.

    Gets reward for charging EVs in previous timestep minus costs of
        violation constraints and amount of charge delivered in current
        timestep.

    Args:
        simulator: internal simulator from acnportal.acnsim
        schedule: dictionary mapping EVSE charger to a single-element list of
            the pilot signal to that charger.
        prev_timestamp: timestamp of previous action taken, needed to
            compute reward from charge received by EVs
        timestamp: timestamp of current action taken
        next_timestamp: timestamp of next action to be taken
        get_info: whether to return information about how reward is calculated

    Returns:
        - total reward awarded to current timestep
        - information on how reward is calculated
    """
    timestamp_diff = next_timestamp - timestamp

    schedule = schedule_to_numpy(schedule)
    # Find charging cost based on amount of charge delivered
    charging_cost = -sum(schedule) * timestamp_diff * CHARGE_COST_WEIGHT

    # Get charging reward based on charging rates for previous interval
    charging_reward = DELIVERED_CHARGE_WEIGHT * np.sum(simulator.charging_rates[:, prev_timestamp:timestamp])

    # Find negative reward for current violation by finding sum of current
    # going over the constraints
    current_sum = np.real(simulator.network.constraint_current(schedule, linear=False))
    magnitudes = simulator.network.magnitudes
    over_current = np.maximum(current_sum - magnitudes, 0)
    constraint_punishment = -CONSTRAINT_VIOLATION_WEIGHT * sum(over_current) * timestamp_diff

    # Get total reward
    total_reward = charging_reward + charging_cost + constraint_punishment

    if get_info:
        info = {
            "schedule": schedule,
            "charging_cost": charging_cost,
            "current_sum": current_sum,
            "magnitudes": magnitudes,
            "constraint_violation": over_current,
            "constraint_punishment": constraint_punishment,
            "charge_cost_weight": CHARGE_COST_WEIGHT,
            "delivered_charge_weight": DELIVERED_CHARGE_WEIGHT,
            "constraint_violation_weight": CONSTRAINT_VIOLATION_WEIGHT,
            "unnormalized_total_reward": total_reward,
        }

    # normalize by number of charging stations
    total_reward /= len(simulator.network.station_ids)

    if get_info:
        return total_reward, info
    else:
        return total_reward


def schedule_to_numpy(schedule: dict) -> np.ndarray:
    """
    Convert schedule dictionary to usable numpy array. Helper to get_rewards.

    Args:
        schedule: dictionary mapping EVSE charger to a single-element list of
            the pilot signal to that charger.

    Returns:
        numpified version of schedule
    """
    return np.array(list(map(lambda x: x[0], schedule.values())))
