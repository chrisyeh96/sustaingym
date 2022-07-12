"""
This module contains methods for defining the observation space of the
simulation and converting simulation information into usable obesrvations for
the gym.
"""
from __future__ import annotations

from typing import Any

from acnportal.acnsim.interface import Interface
from acnportal.acnsim.network import ChargingNetwork
from gym import spaces
import numpy as np

LARGE_INT = 2 ** 30
MINS_IN_DAY = 1440


def get_observation_space(cn: ChargingNetwork) -> spaces.Dict:
    """
    Returns dictionary concerning the observation space for the network.

    Args:
        cn: charging network

    Returns:
        the observation space of the simulator.
    """
    num_constraints, num_stations = cn.constraint_matrix.shape
    return spaces.Dict({
        "arrivals": spaces.Box(low=-LARGE_INT, high=0, shape=(num_stations,), dtype=np.int32),
        "est_departures": spaces.Box(low=0, high=LARGE_INT, shape=(num_stations,), dtype=np.int32),
        "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
        "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
        "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
        "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
        "recompute_freq": spaces.Box(low=0, high=LARGE_INT, shape=(1,), dtype=np.int32),
        "timestep": spaces.Box(low=0, high=LARGE_INT, shape=(1,), dtype=np.int32),
    })


def get_observation(interface: Interface,
                    evse_name_to_idx: dict,
                    recompute_freq: int,
                    get_info: bool = True
                    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Returns dictionary of observations.

    Args:
        interface: interface of acnportal.acnsim Simulator.
        evse_name_to_idx: dictionary defining the index of each EVSE.
        recompute_freq: number of periods per recompute
        get_info: whether extra information should be returned as well.

    Returns:
        observations from internal simulator to be used in gym.
    """
    simulator = interface._simulator
    timestep = simulator._iteration
    cn = simulator.network

    num_stations = cn.constraint_matrix.shape[1]
    arrivals = np.zeros(shape=(num_stations,), dtype=np.int32)
    est_departures = np.zeros(shape=(num_stations,), dtype=np.int32)
    demands = np.zeros(shape=(num_stations,), dtype=np.float32)
    phases = np.zeros(shape=(num_stations,), dtype=np.float32)
    if get_info:
        actual_departures = np.zeros(shape=(num_stations,), dtype=np.int32)

    for session_info in interface.active_sessions():
        station_id = session_info.station_id
        station_idx = evse_name_to_idx[station_id]
        arrivals[station_idx] = session_info.arrival
        est_departures[station_idx] = session_info.estimated_departure
        demands[station_idx] = interface.remaining_amp_periods(session_info)
        if get_info:
            actual_departures[station_idx] = session_info.departure

    for station_id, idx in evse_name_to_idx.items():
        phases[idx] = interface.evse_phase(station_id)

    arrivals = np.where(arrivals, arrivals - timestep, arrivals)
    est_departures = np.where(est_departures, est_departures - timestep, est_departures)

    obs = {
        "arrivals": arrivals,
        "est_departures": est_departures,
        "constraint_matrix": cn.constraint_matrix,
        "magnitudes": cn.magnitudes,
        "demands": demands,
        "phases": phases,
        "recompute_freq": np.ones(shape=(1,), dtype=np.int32) * recompute_freq,
        "timestep": np.ones(shape=(1,), dtype=np.int32) * timestep,
    }

    if get_info:
        actual_departures = np.where(actual_departures, actual_departures - timestep, actual_departures)
        info = {
            "charging_rates": simulator.charging_rates_as_df(),
            "active_evs": simulator.get_active_evs(),
            "pilot_signals": simulator.pilot_signals_as_df(),
            "active_sessions": interface.active_sessions(),
            "departures": actual_departures,
        }
        return obs, info
    else:
        return obs
