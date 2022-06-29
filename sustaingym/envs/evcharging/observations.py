"""
This module contains methods for defining the observation space of the
simulation and converting simulation information into usable obesrvations for
the gym.
"""
from __future__ import annotations

from typing import Any

from acnportal.acnsim.interface import Interface
from gym import spaces
import numpy as np

LARGE_INT = 2 ** 30
MINS_IN_DAY = 1440


def get_observation_space(num_constraints: int, num_stations: int, period: int) -> spaces.Dict:
    """
    Return dictionary concerning the observation space for the network.

    Args:
        num_constraints: number of constraints in charging network.
        num_stations: number of evse charging stations in network.
        period: the number of minutes for the timestep interval.

    Returns:
        the observation space of the simulator.
    """
    num_periods = MINS_IN_DAY // period

    return spaces.Dict({
        "arrivals": spaces.Box(low=-num_periods, high=0, shape=(num_stations,), dtype=np.int32),
        "est_departures": spaces.Box(low=0, high=LARGE_INT, shape=(num_stations,), dtype=np.int32),
        "constraint_matrix": spaces.Box(low=-1.0, high=1.0, shape=(num_constraints, num_stations), dtype=np.float32),
        "magnitudes": spaces.Box(low=0.0, high=np.inf, shape=(num_constraints,), dtype=np.float32),
        "demands": spaces.Box(low=0.0, high=np.inf, shape=(num_stations,), dtype=np.float32),
        "phases": spaces.Box(low=-180.0, high=180.0, shape=(num_stations,), dtype=np.float32),
        "timestep": spaces.Box(low=0, high=LARGE_INT, shape=(1,), dtype=np.int32),
    })


def get_observation(interface: Interface,
                    num_stations: int,
                    evse_name_to_idx: dict,
                    timestep: int,
                    get_info: bool = False
                    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """
    Return dictionary of observations.

    Args:
        interface: interface of acnportal.acnsim Simulator.
        num_stations: number of stations in the garage.
        evse_name_to_idx: dictionary defining the index of each EVSE.
        timestep: timestep of gym simulation.
        get_info: whether extra information should be returned as well.

    Returns:
        observations from internal simulator to be used in gym.
    """
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

    constraint_matrix = interface.get_constraints().constraint_matrix
    magnitudes = interface.get_constraints().magnitudes

    arrivals = np.where(arrivals, arrivals - timestep, arrivals)
    est_departures = np.where(est_departures, est_departures - timestep, est_departures)

    np_timestep = np.ones(shape=(1,), dtype=np.int32) * timestep

    obs = {
        "arrivals": arrivals,
        "est_departures": est_departures,
        "constraint_matrix": constraint_matrix,
        "magnitudes": magnitudes,
        "demands": demands,
        "phases": phases,
        "timestep": np_timestep,
    }

    simulator = interface._simulator

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
