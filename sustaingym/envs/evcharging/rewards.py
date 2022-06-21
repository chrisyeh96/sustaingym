"""
TODO
"""
from __future__ import annotations

from acnportal.acnsim.interface import Interface
from acnportal.acnsim.simulator import Simulator
import numpy as np


CHARGE_WEIGHT = 10
CONSTRAINT_VIOLATION_WEIGHT = 10

def get_rewards(interface: Interface, simulator: Simulator, schedule: dict, prev_timestamp: int, timestamp: int, next_timestamp: int) -> float:
    schedule = schedule_to_numpy(schedule)
    # Find charging cost based on amount of charge delivered
    charging_cost = -sum(schedule) * (next_timestamp - timestamp)

    # Get charging reward based on charging rates for previous interval
    charging_reward = (CHARGE_WEIGHT + 1) * np.sum(simulator.charging_rates[:, prev_timestamp:timestamp])

    # Find negative reward for current violation by finding sum of current
    # going over the constraints
    # TODO: ignores phase
    current_sum = np.real(simulator.network.constraint_current(schedule, linear=True))
    magnitudes = simulator.network.magnitudes
    over_current = np.maximum(current_sum - magnitudes, 0)
    constraint_punishment = -CONSTRAINT_VIOLATION_WEIGHT * sum(over_current) * (next_timestamp - timestamp)

    # Get total reward
    total_reward = charging_reward + charging_cost + constraint_punishment
    # normalize by number of charging stations
    total_reward /= len(simulator.network.station_ids)

    return total_reward


def schedule_to_numpy(schedule: dict) -> np.array:

    return np.array(list(map(lambda x: x[0], schedule.values())))