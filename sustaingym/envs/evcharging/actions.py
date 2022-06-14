import numpy as np

from typing import List

import gym
from gym import spaces

from acnportal.acnsim.interface import Interface


def get_action_space(interface: Interface):
    """
    Returns discretized action space for a charging network.

    Args:
        interface (acnportal.acnsim.interface): interface between algorithms
            and simulation environment 
    
    Returns:
        MultiDiscrete: a space of shape (cn.station_ids,) where each entry
            takes on values in the set {0, 1, 2, 3, 4}
    """
    return spaces.MultiDiscrete(
        [5 for _ in range(len(interface._simulator.network.station_ids))]
    )


def to_schedule(action: np.ndarray, evses: List[str]):
    """
    Returns a dictionary for pilot signals given the action. Entries of action
    are in the set {0, 1, 2, 3, 4}, and they are scaled up by a factor of 8 to
    the set {0, 8, 16, 24, 32}, a discretization of the set of allowed
    currents for AV (AeroVironment) and CC (ClipperCreek).

    Args:
        action : np.ndarray of shape (len(evses),)
            entries of action are pilot signals from {0, 1, 2, 3, 4}
        evses (List) : list of names of charging stations
    
    Returns:
        Dictionary mapping station ids to a schedule of pilot signals, as
        required by the step function of acnportal.acnsim.simulator.Simulator
    """
    return {e: [8 * a] for a, e in zip(action, evses)}