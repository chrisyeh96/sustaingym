"""This module contains functions to calculate rewards."""
from __future__ import annotations

from typing import Any

from acnportal.acnsim.events import UnplugEvent
from acnportal.acnsim.simulator import Interface, Simulator
from acnportal.acnsim.events import Event
import numpy as np


CHARGE_COST_WEIGHT = 1
DELIVERED_CHARGE_WEIGHT = 1
CONSTRAINT_VIOLATION_WEIGHT = 1
UNCHARGED_WEIGHT = 1


def get_rewards(interface: Interface,
                schedule: dict,
                prev_timestamp: int,
                timestamp: int,
                next_timestamp: int,
                period: int,
                cur_event: Event = None,
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
        cur_event: index of unplugged station
        get_info: whether to return information about how reward is calculated

    Returns:
        - total reward awarded to current timestep
        - information on how reward is calculated
    """
    simulator: Simulator = interface._simulator
    schedule = schedule_to_numpy(schedule)

    next_interval_mins = (next_timestamp - timestamp) * period

    # Charging cost: amount of charge sent out at current timestamp (A * mins)
    charging_cost = - CHARGE_COST_WEIGHT * np.sum(schedule) * next_interval_mins

    # Immediate charging reward: amount of charge delivered to vehicles at previous timestamp (A * mins)
    charging_reward = DELIVERED_CHARGE_WEIGHT * np.sum(simulator.charging_rates[:, prev_timestamp:timestamp]) * period

    # Network constraint violation punishment: amount of charge over maximum allowed rates (A * mins)
    current_sum = np.abs(simulator.network.constraint_current(schedule, linear=True))
    over_current = np.maximum(current_sum - simulator.network.magnitudes, 0)
    constraint_punishment = - CONSTRAINT_VIOLATION_WEIGHT * sum(over_current) * next_interval_mins

    # TODO TODO TODO simulator may execute multiple events at once, don't know if an unplug event already got executed
    # Customer satisfaction reward: margin between energy requested and delivered
    if cur_event and isinstance(cur_event, UnplugEvent):
        # punish by how much more energy is requested than is actually delivered
        charge_left_kwh = min(0, cur_event.ev.energy_delivered - cur_event.ev.requested_energy)
        charge_left_amp_mins = interface._convert_to_amp_periods(charge_left_kwh, cur_event.ev.station_id) * period
        remaining_charge_punishment = UNCHARGED_WEIGHT * charge_left_amp_mins
        departure_event = True
    else:
        departure_event = False
        remaining_charge_punishment = 0

    timestamp_diff = next_timestamp - timestamp


    # Get total reward
    # total_reward = charging_cost + charging_reward + constraint_punishment + remaining_amp_periods_punishment
    total_reward = remaining_charge_punishment

    # normalize by number of charging stations
    # total_reward /= len(simulator.network.station_ids)

    if get_info:
        info = {
            "schedule": schedule,
            "current_sum": current_sum,
            "magnitudes": simulator.network.magnitudes,
            "constraint_violation": over_current,
            "charging_cost": charging_cost,
            "charging_reward": charging_reward,
            "constraint_punishment": constraint_punishment,
            "remaining_amp_periods_punishment": remaining_charge_punishment,
            "departure_event": departure_event,
            "charge_cost_weight": CHARGE_COST_WEIGHT,
            "delivered_charge_weight": DELIVERED_CHARGE_WEIGHT,
            "constraint_violation_weight": CONSTRAINT_VIOLATION_WEIGHT,
            "uncharged_weight": UNCHARGED_WEIGHT,
            "unnormalized_total_reward": total_reward,
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
