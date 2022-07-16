"""This module contains functions to calculate rewards."""
from __future__ import annotations

from typing import Any, List

from acnportal.acnsim.events import UnplugEvent
from acnportal.acnsim.simulator import Interface, Simulator
import acnportal.acnsim.models as acnm
import numpy as np


CHARGE_COST_WEIGHT = 1.
DELIVERED_CHARGE_WEIGHT = 2. * CHARGE_COST_WEIGHT
CONSTRAINT_VIOLATION_WEIGHT = 10.
UNCHARGED_PUNISHMENT_WEIGHT = 5.
weights = {
    "charge_cost": CHARGE_COST_WEIGHT,
    "delivered_charge": DELIVERED_CHARGE_WEIGHT,
    "constraint_violation": CONSTRAINT_VIOLATION_WEIGHT,
    "uncharged_punishment": UNCHARGED_PUNISHMENT_WEIGHT,
}


def get_rewards(interface: Interface,
                schedule: dict,
                prev_timestamp: int,
                timestamp: int,
                next_timestamp: int,
                period: int,
                done: bool,
                evs: List[acnm.ev.EV],
                get_info: bool = True,
                ) -> float | tuple[float, dict[str, Any]]:
    """
    Returns reward for scheduler's performance.

    Gets reward for charging EVs in previous timestep minus costs of
    violation constraints and amount of charge delivered in current
    timestep.

    Args:
        interface: acnportal.acnsim interface object
        schedule: dictionary mapping EVSE charger to a single-element list of
            the pilot signal to that charger.
        prev_timestamp: timestamp of previous action taken, needed to
            compute reward from charge received by EVs
        timestamp: timestamp of current action taken
        next_timestamp: timestamp of next action to be taken
        period: number of minutes per period
        done: whether simulation is done and customer satisfaction reward
            should be calculated
        evs: list of all EVs during the simulation
        get_info: whether to return information about how reward is calculated

    Returns:
        - total reward awarded to current timestep
        - information on how reward is calculated
    """
    simulator: Simulator = interface._simulator
    schedule = schedule_to_numpy(schedule)

    next_interval_mins = (next_timestamp - timestamp) * period

    # Charging cost: amount of charge sent out at current timestamp (A * mins)
    charging_cost = np.sum(schedule) * next_interval_mins

    # Immediate charging reward: amount of charge delivered to vehicles at previous timestamp (A * mins)
    charging_reward = np.sum(simulator.charging_rates[:, prev_timestamp:timestamp]) * period

    # Network constraint violation punishment: amount of charge over maximum allowed rates at previous timestamp (A * mins)
    current_sum = np.abs(simulator.network.constraint_current(schedule))
    over_current = np.maximum(current_sum - simulator.network.magnitudes, 0)
    constraint_punishment = sum(over_current) * next_interval_mins

    # Customer satisfaction reward: margin between energy requested and delivered (A * mins)
    # only computed once when simulation is finished
    charge_left_amp_mins = 0.
    if done:
        for ev in evs:
            charge_left_amp_mins += interface._convert_to_amp_periods(ev.remaining_demand, ev.station_id) * period

    # Get total reward
    total_reward = (
        - CHARGE_COST_WEIGHT * charging_cost
        + DELIVERED_CHARGE_WEIGHT * charging_reward
        - CONSTRAINT_VIOLATION_WEIGHT * constraint_punishment
        - UNCHARGED_PUNISHMENT_WEIGHT * charge_left_amp_mins)
                
    # convert to (kA * hrs)
    total_reward /= (60 * 1000)

    if get_info:
        info = {
            "weights": weights,
            "charging_cost": charging_cost,
            "charging_reward": charging_reward,
            "constraint_punishment": constraint_punishment,
            "remaining_charge_punishment": charge_left_amp_mins,
        }
        return total_reward, info
    else:
        return total_reward


def schedule_to_numpy(schedule: dict) -> np.ndarray:
    """
    Converts schedule dictionary to usable numpy array.

    Args:
        schedule: dictionary mapping EVSE charger to a single-element list of
            the pilot signal to that charger.

    Returns:
        numpified version of schedule
    """
    return np.array(list(map(lambda x: x[0], schedule.values())))
