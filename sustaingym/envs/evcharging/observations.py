import numpy as np

import gym
from gym import spaces

from acnportal.acnsim.interface import Interface

from collections import OrderedDict

MAX_INT = int(2**31 - 1)

def get_observation_space(num_constraints: int, num_stations: int):
    """
    Returns dictionary concerning the observation space for the network.

    Args:
        num_constraints (int): number of constraints in charging network
        num_stations (int): number of evse charging stations in network
    
    Returns:
        spaces.Dict()
        MultiDiscrete: a space of shape (cn.station_ids,) where each entry
            takes on values in the set {0, 1, 2, 3, 4}
    """

    return spaces.Dict({
        "arrivals": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "departures": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
        "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
        "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
        "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
        "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
        "timestep": spaces.Box(low=0, high=MAX_INT, shape=(), dtype=np.int32),
    })


def get_observation(interface: Interface, num_constraints: int, num_stations: int, evse_name_to_idx: dict, timestep: int):
    # constraints = interface.get_constraints()
    # num_constraints, num_stations = constraints.constraint_matrix.shape
    # constraints.evse_index

    arrivals = np.zeros(shape=(num_stations,), dtype=np.int32)
    departures = np.zeros(shape=(num_stations,), dtype=np.int32)
    demands = np.zeros(shape=(num_stations,), dtype=np.float32)
    phases = np.zeros(shape=(num_stations,), dtype=np.float32)

    for session_info in interface.active_sessions():
        station_id = session_info.station_id
        station_idx = evse_name_to_idx[station_id]

        arrivals[station_idx] = session_info.arrival
        departures[station_idx] = session_info.departure
        demands[station_idx] = interface.remaining_amp_periods(session_info)
    
    for station_id, idx in evse_name_to_idx.items():
        phases[idx] = interface.evse_phase(station_id)
    
    constraint_matrix = interface.get_constraints().constraint_matrix
    magnitudes = interface.get_constraints().magnitudes

    return OrderedDict([
        ("arrivals", arrivals),
        ("departures", departures),
        ("constraint_matrix", constraint_matrix),
        ("magnitudes", magnitudes),
        ("demands", demands),
        ("phases", phases),
        ("timestep", timestep),
    ])


    # for evse in evse_name_to_idx.keys():
    # demands = interface.remaining_amp_periods()
    # # print(interface._simulator.get_active_evs())

    # return spaces.Dict({
    #     "arrivals": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
    #     "departures": spaces.Box(low=0, high=MAX_INT, shape=(num_stations,), dtype=np.int32),
    #     "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
    #     "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
    #     "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
    #     "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
    #     "timestep": spaces.Box(low=0, high=MAX_INT, shape=(), dtype=np.int32),
    # }).sample()

