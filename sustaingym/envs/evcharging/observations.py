"""
This module contains methods for defining the observation space of the
simulation and converting simulation information into usable obesrvations for
the gym.
"""
from __future__ import annotations

from acnportal.acnsim.interface import Interface
from gym import spaces
import numpy as np

MAX_INT = int(2**31 - 1)


def get_observation_space(num_constraints: int, num_stations: int) -> spaces.Dict:
    """
    Return dictionary concerning the observation space for the network.

    Args:
        num_constraints (int): number of constraints in charging network
        num_stations (int): number of evse charging stations in network

    Returns:
        (spaces.Dict()): observation space of simulator
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


def get_observation(interface: Interface, num_stations: int, evse_name_to_idx: dict, timestep: int) -> dict:
    """
    Return dictionary of observations.

    Args:
        interface (Interface) - interface of acnportal.acnsim Simulator
        num_stations (int) - number of stations in the garage
        evse_name_to_idx (dict) - dictionary defining the index of each EVSE
        timestep (int) - timestep of gym simulation

    Returns:
        (dict): observations from internal simulator to be used in gym
    """
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

    return {
        "arrivals": arrivals,
        "departures": departures,
        "constraint_matrix": constraint_matrix,
        "magnitudes": magnitudes,
        "demands": demands,
        "phases": phases,
        "timestep": timestep,
    }
