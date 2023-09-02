from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

class Network:
    N: int               # number of nodes (buses)
    M: int               # number of lines
    N_G: int             # number of generators
    N_D: int             # number of loads
    N_B: int             # number of batteries
    J: np.ndarray        # participant-to-node matrix, denoted Σ in write-up,
                         #   values in {-1, 0, 1}, shape [N, N_D + N_G + 2*N_B]
    H: np.ndarray        # generation_shift_factor_matrix, shape [M, N]
    g_min: np.ndarray    # min power for generators (MW), shape [N_G]
    g_max: np.ndarray    # max power for generators (MW), shape [N_G]
    bat_c_max: np.ndarray  # max battery charge power (MW), shape [N_B]
    bat_d_max: np.ndarray  # max battery discharge power (MW), shape [N_B]
    f_max: np.ndarray    # line maximum continuous power flow (MVA), shape [M]
    c_gen: np.ndarray    # cost of generation, shape [N_G, 2],
                         #   if p (shape [N_G]) is power output (MW), then $/hr
                         #   is c_gen[:,0] + c_gen[:,1] * p

    eff_c: np.ndarray    # battery charging efficiency in (0, 1], shape [N_B]
    eff_d: np.ndarray    # battery discharging efficiency in (0, 1], shape [N_B]
    soc_min: np.ndarray  # minimum battery state-of-charge in MWh, shape [N_B]
    soc_max: np.ndarray  # maximum battery state-of-charge in MWh, shape [N_B]

    def get_nodal_demand(self, dt: datetime | Sequence[datetime],
                         forecast: bool = False) -> np.ndarray:
        """Calculates nodal demand or nodal demand forecast (in MW) for a given
        datetime or range of datetimes.

        Args:
            dt: datetime or range of datetimes, length h
            forecast: bool, whether to return forecast

        Returns: np.ndarray, shape [N_D] or [N_D, h]
        """
        raise NotImplementedError
