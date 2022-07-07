"""
This module contains methods for defining the action space of the simulation
and converting actions to usable pilot signals for the Simulator class in
acnportal.acnsim.
"""
from __future__ import annotations

from gym import spaces
import numpy as np

from acnportal.acnsim.network import ChargingNetwork


ACTION_DISCRETIZATION_FACTOR = 8
MIN_PILOT_SIGNAL = 8
MAX_PILOT_SIGNAL = 32


def get_action_space(cn: ChargingNetwork, action_type: str) -> spaces.MultiDiscrete:
    """
    Return discretized action space for a charging network.

    Args:
        num_stations: number of evse charging stations in network
        action_type: either 'discrete' or 'continuous'

    Returns:
        a space of shape (cn.station_ids,) where each entry
            takes on values in the set {0, 1, 2, 3, 4}
    """
    if action_type == 'discrete':
        return spaces.MultiDiscrete(
            [5 for _ in range(len(cn.station_ids))]
        )
    elif action_type == 'continuous':
        return spaces.Box(
            low=0, high=1, shape=(num_stations,), dtype=np.float32
        )
    else:
        raise ValueError("Only 'discrete' and 'continuous' action_types are allowed. ")


def to_schedule(action: np.ndarray, cn: ChargingNetwork, action_type: str) -> dict[str, list[float]]:
    """
    Returns a dictionary for pilot signals given the action.

    Entries of an action are in the set {0, 1, 2, 3, 4}, and they are scaled
    up by a factor of 8 to the set {0, 8, 16, 24, 32}, a discretization of the
    set of allowed currents for AV (AeroVironment) and CC (ClipperCreek).

    Args:
        action: np.ndarray of shape (len(evses),)
            entries of action are pilot signals from {0, 1, 2, 3, 4}
        evses: list of names of charging stations TODO
        action_type: either 'discrete' or 'continuous'


    Returns:
        a dictionary mapping station ids to a schedule of pilot signals, as
            required by the step function of
            acnportal.acnsim.simulator.Simulator
    """
    evses = cn.station_ids
    if action_type == 'discrete':
        return {e: [ACTION_DISCRETIZATION_FACTOR * a] for a, e in zip(action, evses)}
    elif action_type == 'continuous':
        action = np.round(action * MAX_PILOT_SIGNAL)
        action = np.where(action > MIN_PILOT_SIGNAL, action, 0)
        return {e: [a] for a, e in zip(action, evses)}
    else:
        raise ValueError("Only 'discrete' and 'continuous' action_types are allowed. ")
