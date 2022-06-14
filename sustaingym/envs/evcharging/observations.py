import numpy as np

import gym
from gym import spaces

from acnportal.acnsim.interface import Interface

MAX_INT = int(2**31 - 1)

def get_observation_space(interface: Interface):
    """
    Returns dictionary concerning the observation space for the network.

    Args:
        interface (acnportal.acnsim.interface): interface between algorithms
            and simulation environment 
    
    Returns:
        spaces.Dict(

        )
        MultiDiscrete: a space of shape (cn.station_ids,) where each entry
            takes on values in the set {0, 1, 2, 3, 4}
    """
    constraints = interface.get_constraints()
    num_constraints, num_stations = constraints.constraint_matrix.shape

    return spaces.Dict({
        "arrivals": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "departures": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
        "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
        "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
        "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
        "timestep": spaces.Box(low=0, high=MAX_INT, shape=(), dtype=np.int32),
    })


def get_observation(interface: Interface):
    constraints = interface.get_constraints()
    num_constraints, num_stations = constraints.constraint_matrix.shape

    return spaces.Dict({
        "arrivals": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "departures": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
        "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
        "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
        "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
        "timestep": spaces.Box(low=0, high=MAX_INT, shape=(), dtype=np.int32),
    }).sample()

