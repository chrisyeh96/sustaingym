"""
This module contains methods that define the action space and convert actions
to usable pilot signals for the Simulator class in acnportal.acnsim.
"""
from __future__ import annotations

from acnportal.acnsim.network import ChargingNetwork
from gym import spaces
import numpy as np

ACTION_SCALING_FACTOR = 8
DISCRETE_MULTIPLE = 8
MIN_PILOT_SIGNAL = 6
MAX_PILOT_SIGNAL = 32


def get_action_space(cn: ChargingNetwork, action_type: str) -> spaces.MultiDiscrete:
    """
    Returns action space for the charging network.

    Args:
        cn: charging network in environment simulation
        action_type: either 'discrete' or 'continuous'

    Returns:
        a space of shape (cn.station_ids,) where each entry takes on values in
        the set {0, 1, 2, 3, 4} or [0, 4], depending on action_type
    """
    if action_type == 'discrete':
        return spaces.MultiDiscrete(
            [5 for _ in range(len(cn.station_ids))]
        )
    elif action_type == 'continuous':
        return spaces.Box(
            low=0, high=4, shape=(len(cn.station_ids),), dtype=np.float32
        )
    else:
        raise ValueError("Only 'discrete' and 'continuous' action_types are allowed. ")


def to_schedule(action: np.ndarray, cn: ChargingNetwork, action_type: str) -> dict[str, list[float]]:
    """
    Given a numpy action, returns a dictionary usable for the Simulator class.

    Discrete actions are scaled up by a factor of 8 to the set
    {0, 8, 16, 24, 32}, a discretization of the set of allowed currents for AV
    (AeroVironment) and CC (ClipperCreek). Continuous actions are also scaled
    up by a factor of 8.

    Args:
        action: np.ndarray of shape (len(evses),)
            If action_type is 'discrete', expects actions in the set
                {0, 1, 2, 3, 4}.
            If action_type is 'continuous', expects actions in [0, 4].
        cn: charging network in environment simulation
        action_type: either 'discrete' or 'continuous'

    Returns:
        a dictionary mapping station ids to a schedule of pilot signals, as
            required by acnportal's Simulator
    """
    if action_type == 'discrete':
        return {e: [ACTION_SCALING_FACTOR * a] for a, e in zip(action, cn.station_ids)}
    elif action_type == 'continuous':
        action = np.round(action * ACTION_SCALING_FACTOR).astype(np.int32)  # round to nearest integer rate
        pilot_signals = {}
        for i in range(len(cn.station_ids)):
            station_id = cn.station_ids[i]
            # hacky way to determine allowable rates
            if cn._EVSEs[station_id].allowable_rates[1] == MIN_PILOT_SIGNAL:
                pilot_signals[station_id] = [action[i] if action[i] >= MIN_PILOT_SIGNAL else 0]  # signals less than minimum (of 6) are set to zero, rest are in action space
            else:
                pilot_signals[station_id] = [(np.round(action[i] / DISCRETE_MULTIPLE) * DISCRETE_MULTIPLE).astype(np.int32)]  # set to {0, 8, 16, 24, 32}
        return pilot_signals
    else:
        raise ValueError("Only 'discrete' and 'continuous' action_types are allowed. ")
