from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np


class Network:
    """Abstract base class defining a transmission network."""

    N: int                  # number of nodes (buses)
    M: int                  # number of lines
    N_G: int                # number of generators
    N_D: int                # number of loads
    N_B: int                # number of batteries

    edges: np.ndarray       # edges in network, shape [M, 2], 0-indexed nodes
    J: np.ndarray           # participant-to-node matrix denoted Σ in write-up,
                            #   values in {0, ±1}, shape [N, N_D + N_G + 2*N_B]
    H: np.ndarray           # generation_shift_factor_matrix, shape [M, N]
    g_min: np.ndarray       # min power for generators (MW), shape [N_G]
    g_max: np.ndarray       # max power for generators (MW), shape [N_G]
    bat_c_max: np.ndarray   # max battery charge power (MW), shape [N_B]
    bat_d_max: np.ndarray   # max battery discharge power (MW), shape [N_B]
    f_max: np.ndarray       # line maximum continuous power flow (MVA), shape [M]

    # cost of generation, shape [N_G, 2],
    # - 1st col (c_gen[:,0]): units of $/hr
    # - 2nd col (c_gen[:,1]): units of $/MWh
    # - if p (shape [N_G]) is power output (MW), then $/hr is
    #     c_gen[:,0] + c_gen[:,1] * p
    c_gen: np.ndarray

    eff_c: np.ndarray       # battery charging efficiency in (0, 1], shape [N_B]
    eff_d: np.ndarray       # battery discharging efficiency in (0, 1], shape [N_B]
    soc_min: np.ndarray     # min battery state-of-charge (MWh), shape [N_B]
    soc_max: np.ndarray     # max battery state-of-charge (MWh), shape [N_B]
    bat_nodes: np.ndarray   # nodes (0-indexed) of batteries, shape [N_B]

    def check_valid_period(self, start: datetime, end: datetime,
                           forecast_steps: int) -> bool:
        """Check whether [start, end) period is valid for this network, when
        accounting for forecast.
        """
        raise NotImplementedError

    def aggregate_demand(self, dt: datetime | Sequence[datetime],
                         forecast: bool = False) -> float | np.ndarray:
        """Get aggregate demand or aggregate demand forecast (in MW) for a
        given datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h, in year 2020 and
                aligned to 5-min frequency
            forecast: bool, whether to return forecast

        Returns: scalar or array of shape [h]
        """
        raise NotImplementedError

    def demand(self, dt: datetime | Sequence[datetime],
               forecast: bool = False) -> np.ndarray:
        """Calculate demand or demand forecast (in MW) at each load for a
        given datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h
            forecast: bool, whether to return forecast

        Returns: np.ndarray, shape [N_D] or [N_D, h]
        """
        raise NotImplementedError
