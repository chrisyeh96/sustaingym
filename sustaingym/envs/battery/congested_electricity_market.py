"""
The module implements the ElectricityMarketEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
import os
import pkgutil
from typing import Any

import cvxpy as cp
from gym import Env, spaces
import numpy as np
import pandas as pd
import pandapower.networks as pn
from pandapower.auxiliary import pandapowerNet
import pytz

from sustaingym.data.load_moer import MOERLoader
from sustaingym.envs.utils import solve_mosek

BATTERY_STORAGE_MODULE = 'sustaingym.envs.battery'

class Case24_ieee_rts:
    def __init__(self):
        """Power flow data for the IEEE Reliability Test System.
        Please see L{caseformat} for details on the case file format.

        This system data is from the IEEE Reliability Test System, see

        IEEE Reliability Test System Task Force of the Applications of
        Probability Methods Subcommittee, I{"IEEE reliability test system,"}
        IEEE Transactions on Power Apparatus and Systems, Vol. 98, No. 6,
        Nov./Dec. 1979, pp. 2047-2054.

        IEEE Reliability Test System Task Force of Applications of
        Probability Methods Subcommittee, I{"IEEE reliability test system-96,"}
        IEEE Transactions on Power Systems, Vol. 14, No. 3, Aug. 1999,
        pp. 1010-1020.

        Cost data is from Web site run by Georgia Tech Power Systems Control
        and Automation Laboratory:
        U{http://pscal.ece.gatech.edu/testsys/index.html}

        MATPOWER case file data provided by Bruce Wollenberg.

        @return: Power flow data for the IEEE RELIABILITY TEST SYSTEM.
        """
        ppc = {"version": '2'}

        ##-----  Power Flow Data  -----##
        ## system MVA base
        ppc["baseMVA"] = 100.0

        ## bus data
        # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
        ppc["bus"] = np.array([
            [1,  2, 108, 22, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [2,  2,  97, 20, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [3,  1, 180, 37, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [4,  1,  74, 15, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [5,  1,  71, 14, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [6,  1, 136, 28, 0, -100, 2, 1, 0, 138, 1, 1.05, 0.95],
            [7,  2, 125, 25, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
            [8,  1, 171, 35, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
            [9,  1, 175, 36, 0, 0,    1, 1, 0, 138, 1, 1.05, 0.95],
            [10, 1, 195, 40, 0, 0,    2, 1, 0, 138, 1, 1.05, 0.95],
            [11, 1,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [12, 1,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [13, 3, 265, 54, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [14, 2, 194, 39, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [15, 2, 317, 64, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [16, 2, 100, 20, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [17, 1,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [18, 2, 333, 68, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [19, 1, 181, 37, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [20, 1, 128, 26, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [21, 2,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [22, 2,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95],
            [23, 2,   0,  0, 0, 0,    3, 1, 0, 230, 1, 1.05, 0.95],
            [24, 1,   0,  0, 0, 0,    4, 1, 0, 230, 1, 1.05, 0.95]
        ])

        ## generator data
        # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
        # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
        ppc["gen"] = np.array([
            [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [1,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [1,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [2,  10,   0,    10,   0, 1.035, 100, 1,  20,  16,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U20
            [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [2,  76,   0,    30, -25, 1.035, 100, 1,  76,  15.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U76
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [7,  80,   0,    60,   0, 1.025, 100, 1, 100,  25,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U100
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [13, 95.1, 0,    80,   0, 1.02,  100, 1, 197,  69,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U197
            [14, 0,   35.3, 200, -50, 0.98,  100, 1,   0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SynCond
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 12,   0,     6,   0, 1.014, 100, 1,  12,   2.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U12
            [15, 155,  0,    80, -50, 1.014, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [16, 155,  0,    80, -50, 1.017, 100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [18, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U400
            [21, 400,  0,   200, -50, 1.05,  100, 1, 400, 100,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U400
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [22, 50,   0,    16, -10, 1.05,  100, 1,  50,  10,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U50
            [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [23, 155,  0,    80, -50, 1.05,  100, 1, 155,  54.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # U155
            [23, 350,  0,   150, -25, 1.05,  100, 1, 350, 140,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # U350
        ])

        ## branch data
        # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
        ppc["branch"] = np.array([
            [1,   2, 0.0026, 0.0139, 0.4611, 175, 250, 200, 0,    0, 1, -360, 360],
            [1,   3, 0.0546, 0.2112, 0.0572, 175, 208, 220, 0,    0, 1, -360, 360],
            [1,   5, 0.0218, 0.0845, 0.0229, 175, 208, 220, 0,    0, 1, -360, 360],
            [2,   4, 0.0328, 0.1267, 0.0343, 175, 208, 220, 0,    0, 1, -360, 360],
            [2,   6, 0.0497, 0.192,  0.052,  175, 208, 220, 0,    0, 1, -360, 360],
            [3,   9, 0.0308, 0.119,  0.0322, 175, 208, 220, 0,    0, 1, -360, 360],
            [3,  24, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
            [4,   9, 0.0268, 0.1037, 0.0281, 175, 208, 220, 0,    0, 1, -360, 360],
            [5,  10, 0.0228, 0.0883, 0.0239, 175, 208, 220, 0,    0, 1, -360, 360],
            [6,  10, 0.0139, 0.0605, 2.459,  175, 193, 200, 0,    0, 1, -360, 360],
            [7,   8, 0.0159, 0.0614, 0.0166, 175, 208, 220, 0,    0, 1, -360, 360],
            [8,   9, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0,    0, 1, -360, 360],
            [8,  10, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0,    0, 1, -360, 360],
            [9,  11, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
            [9,  12, 0.0023, 0.0839, 0,      400, 510, 600, 1.03, 0, 1, -360, 360],
            [10, 11, 0.0023, 0.0839, 0,      400, 510, 600, 1.02, 0, 1, -360, 360],
            [10, 12, 0.0023, 0.0839, 0,      400, 510, 600, 1.02, 0, 1, -360, 360],
            [11, 13, 0.0061, 0.0476, 0.0999, 500, 600, 625, 0,    0, 1, -360, 360],
            [11, 14, 0.0054, 0.0418, 0.0879, 500, 625, 625, 0,    0, 1, -360, 360],
            [12, 13, 0.0061, 0.0476, 0.0999, 500, 625, 625, 0,    0, 1, -360, 360],
            [12, 23, 0.0124, 0.0966, 0.203,  500, 625, 625, 0,    0, 1, -360, 360],
            [13, 23, 0.0111, 0.0865, 0.1818, 500, 625, 625, 0,    0, 1, -360, 360],
            [14, 16, 0.005,  0.0389, 0.0818, 500, 625, 625, 0,    0, 1, -360, 360],
            [15, 16, 0.0022, 0.0173, 0.0364, 500, 600, 625, 0,    0, 1, -360, 360],
            [15, 21, 0.0063, 0.049,  0.103,  500, 600, 625, 0,    0, 1, -360, 360],
            [15, 21, 0.0063, 0.049,  0.103,  500, 600, 625, 0,    0, 1, -360, 360],
            [15, 24, 0.0067, 0.0519, 0.1091, 500, 600, 625, 0,    0, 1, -360, 360],
            [16, 17, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
            [16, 19, 0.003,  0.0231, 0.0485, 500, 600, 625, 0,    0, 1, -360, 360],
            [17, 18, 0.0018, 0.0144, 0.0303, 500, 600, 625, 0,    0, 1, -360, 360],
            [17, 22, 0.0135, 0.1053, 0.2212, 500, 600, 625, 0,    0, 1, -360, 360],
            [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
            [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0,    0, 1, -360, 360],
            [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0,    0, 1, -360, 360],
            [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0,    0, 1, -360, 360],
            [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0,    0, 1, -360, 360],
            [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0,    0, 1, -360, 360],
            [21, 22, 0.0087, 0.0678, 0.1424, 500, 600, 625, 0,    0, 1, -360, 360]
        ])

        ##-----  OPF Data  -----##
        ## area data
        # area refbus
        ppc["areas"] = np.array([
            [1, 1],
            [2, 3],
            [3, 8],
            [4, 6],
        ])

        ## generator cost data
        # 1 startup shutdown n x1 y1 ... xn yn
        # 2 startup shutdown n c(n-1) ... c0
        ppc["gencost"] = np.array([                           # bus Pmin  Pmax Qmin Qmax Unit Code
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  1,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  1,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  1,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  1,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  2,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0,        130,      400.6849],  #  2,  16,   20,   0,  10, U20
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  2,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.014142,  16.0811, 212.3076],  #  2,  15.2, 76, -25,  30, U76
            [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.052672,  43.6615*0.67, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.052672,  43.6615, 781.521],  #  7,  25,  100,   0,  60, U100
            [2, 1500, 0, 3, 0.00717,   48.5804+2, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0.00717,   48.5804 + 4, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0.00717,   48.5804 + 6, 832.7575],  # 13,  69,  197,   0,  80, U197
            [2, 1500, 0, 3, 0,          0,        0],      # 14                       SynCond
            [2, 1500, 0, 3, 0.328412,  56.564+1,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.328412,  56.564,   86.3852],  # 15,  2.4,  12,   0,   6, U12
            [2, 1500, 0, 3, 0.008342,  12.3883+2, 382.2391],  # 15, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.008342,  12.3883+3, 382.2391],  # 16, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.000213,   4.4231, 395.3749],  # 18, 100,  400, -50, 200, U400
            [2, 1500, 0, 3, 0.000213,   4.4231, 395.3749],  # 21, 100,  400, -50, 200, U400
            [2, 1500, 0, 3, 0,          1,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          2,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          3,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          4,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          5,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0,          6,    0.001],  # 22, 10,    50, -10,  16, U50
            [2, 1500, 0, 3, 0.008342,  12.3883, 382.2391], # 23, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.008342,  12.3883+1, 382.2391], # 23, 54.3, 155, -50,  80, U155
            [2, 1500, 0, 3, 0.004895,  11.8495, 665.1094], # 23, 140,  350, -25, 150, U350
        ])

        # Battery specs, to be added
        # node, Pmax, Pmin, eff_c, eff_d, soc, soc_f, soc_min, soc_max, cost_c, cost_d
        ppc["battery"] = np.array([
            [11, 20, -20, 0.95, 0.95, 40, 40, 0, 80, 3, 3]
        ])

        self.ppc = ppc

    def construct_network(self) -> dict[str, Any]:
        N = len(self.ppc['bus'])
        M = len(self.ppc['branch'])
        L = len(self.ppc['battery'])

        # load injection data
        p_l = []
        for i in range(N):
            bus_pd = self.ppc['bus'][i, 2]
            if bus_pd > 0:
                p_l.append((int(i+1), bus_pd))
        load_data = np.array(p_l)  # shape [D, 2]

        self.load_data = load_data[:, 1]  # shape [D]

        # C, B, f_max
        C = np.zeros((N, M))
        b = np.zeros(M)
        f_max = np.zeros(M)
        edges = []
        for j in range(M):
            fbus = int(self.ppc['branch'][j, 0] - 1)  # 0 indexing
            tbus = int(self.ppc['branch'][j, 1] - 1)  # 0 indexing
            v_mag_fbus = self.ppc['bus'][fbus, 7]
            v_mag_tbus = self.ppc['bus'][tbus, 7]
            f_max[j] = self.ppc['branch'][j, 5]  # rateA (long-term line rating)
            b[j] = self.ppc['branch'][j, 4] * abs(v_mag_fbus) * abs(v_mag_tbus)  # voltage-weighted susceptance
            C[fbus, j] = 1
            C[tbus, j] = -1
            edges.append((fbus+1, tbus+1))
        B = np.diag(b)
        H = B @ C.T @ np.linalg.pinv(C @ B @ C.T)


        # generators
        num_gen = len(self.ppc['gen'])
        gen_data = np.zeros((num_gen, 5))
        gen_nodes = []
        gen_edges = []
        for g in range(num_gen):
            gen_nodes.append(101 + g)
            gen_edges.append((101+g, int(self.ppc['gen'][g, 0])))

            gen_data[g, 0] = self.ppc['gen'][g, 9]  # p_min
            gen_data[g, 1] = self.ppc['gen'][g, 8]  # p_max
            gen_data[g, 2] = int(self.ppc['gen'][g, 0])  # bus id
            gen_data[g, 3] = self.ppc['gencost'][g, -1]  # c0
            gen_data[g, 4] = self.ppc['gencost'][g, -2]  # c1

        # J mapping matrix, shape = (N, D + G + 2 * L)
        D = len(self.load_data)
        G = len(gen_data)
        J = np.zeros((N, D + G + 2 * L)) # for the charge and discharge battery decisions

        for d in range(D):
            bus_idx = int(load_data[d, 0] - 1)
            J[bus_idx, d] = -1
        for g in range(G):
            bus_idx = int(gen_data[g, 2] - 1)
            J[bus_idx, D + g] = 1
        for l in range(L):
            bus_idx = int(self.ppc['battery'][l, 0] - 1)
            J[bus_idx, D + G + l] = -1     # charge
            J[bus_idx, D + G + L + l] = 1  # discharge

        # p_min, p_max for loads, gens, batteries (charge), batteries (discharge)
        p_min = np.concatenate([self.load_data, gen_data[:, 0], np.zeros(2*L)])
        p_max = np.concatenate([self.load_data, gen_data[:, 1], self.ppc['battery'][:,1], self.ppc['battery'][:,1]])

        c_gen = [[gen_data[g, 3], gen_data[g, 4]] for g in range(G)]
        c_bat = self.ppc['battery'][:L, -2:]

        soc = self.ppc['battery'][:L, 5]
        soc_min = self.ppc['battery'][:L, 7]
        soc_max = self.ppc['battery'][:L, 8]
        effs = self.ppc['battery'][:L, 3:5]

        # Network data dict
        network = {
            'name': 'IEE24RTS',
            'N': N,
            'M': M,
            'D': D,
            'G': G,
            'L': L,
            'B': B,
            'H': H,
            'J': J,
            'p_min': p_min,
            'p_max': p_max,
            'f_max': f_max,
            'c_gen': c_gen,
            'c_bat': c_bat,
            'soc': soc,
            'soc_min': soc_min,
            'soc_max': soc_max,
            'effs': effs
        }

        return network

class CongestedNetwork:
    """CongestedNetwork class."""
    def __init__(self, market_network: dict = None, load_data: np.ndarray = None):
        if market_network is not None:
            self.network = market_network
        else:
            case24_ieee = Case24_ieee_rts()
            self.network = case24_ieee.construct_network()

        if load_data is not None:
            self.default_load_split = load_data / np.sum(load_data)
        else:
            self.default_load_split = case24_ieee.load_data / np.sum(case24_ieee.load_data)

        self.load_split = self.default_load_split.copy()

        self.N = self.network['N']  # number of nodes
        self.M = self.network['M']  # number of lines
        self.N_G = self.network['G']  # number of generators
        self.N_D = self.network['D']  # number of loads
        self.N_B = self.network['L']  # number of batteries
        self.J = self.network['J']  # pow_inject_to_bus_injection_matrix
        self.H = self.network['H']  # generation_shift_factor_matrix
        self.pmin = self.network['p_min']  # minimum power for participants
        self.pmax = self.network['p_max']  # maximum power for participants
        self.cgen = np.array(self.network['c_gen'])  # cost of generation
        self.cbat = np.array(self.network['c_bat'])  # costs for battery charge decisions
        self.fmax = self.network['f_max']

    def update_load_split(self):
        # shift each +/- 10% of default load_split

        # load_split = np.zeros(self.D)

        # for i in range(self.D-1):
        #     load_split[i] = np.random.uniform(low=0.9*self.default_load_split[i], high=1.1*self.default_load_split[i])

        # load_split[-1] = 1 - np.sum(load_split)
        # self.load_split[:] = load_split

        self.load_split = self.default_load_split.copy()

class CongestedMarketOperator:
    """MarketOperator class."""
    def __init__(self, env: CongestedElectricityMarketEnv, network: CongestedNetwork = None):
        """
        Args:
            env: instance of ElectricityMarketEnv class
        """
        self.env = env
        if network is None:
            self.network = CongestedNetwork()

        h = self.env.settlement_interval  # horizon
        N_D = self.network.N_D  # num loads
        N_G = self.network.N_G  # num generators
        N_B = self.network.N_B  # num batteries

        # Variables
        # dispatch for participants (loads are trivially fixed by constraints)
        self.out = cp.Variable((N_D + N_G + 2*N_B, h + 1), name="dispatch")
        soc = cp.Variable((N_B, h + 2))

        gen_dis = self.out[N_D: N_D+N_G]
        bat_d = self.out[N_D+N_G: N_D+N_G+N_B]  # discharge
        bat_c = self.out[N_D+N_G+N_B:]  # charge

        # Parameters
        self.p_min = cp.Parameter((N_D + N_G + 2*N_B, h+1), name="minimum power")
        self.p_max = cp.Parameter((N_D + N_G + 2*N_B, h+1), nonneg=True, name="maximum power")
        self.cgen = cp.Parameter((N_G, 2), nonneg=True, name="generator production costs")  # assume time invariant
        self.cbat_d = cp.Parameter((N_B, h + 1), nonneg=True, name="battery discharge costs")  # discharge cost
        self.cbat_c = cp.Parameter((N_B, h + 1), nonneg=True, name="battery charge costs")  # charge cost
        self.soc_final = cp.Parameter(N_B, nonneg=True, name="soc final")  # final charge SOC(s)
        self.soc_max = cp.Parameter(N_B, nonneg=True, name="soc maximum")  # maximum SOC(s)

        # Constraints
        constraints = [
            # battery range
            0 <= soc,
            # soc <= self.soc_max,

            # initial and final soc
            soc[:, 0] == self.env.battery_charge,
            # soc[:, -1] >= self.soc_final,

            # charging dynamics
            # soc[:, 1:] == soc[:, :-1] + self.env.CHARGE_EFFICIENCY * bat_c - (1. / self.env.DISCHARGE_EFFICIENCY) * bat_d,

            # generation limits
            self.out >= self.p_min,
            self.out <= self.p_max,
        ]

        # power balance
        self.power_balance_constr = cp.sum(self.network.J @ self.out, axis=0) == 0
        constraints.append(self.power_balance_constr)

        if self.env.congestion:
            # power flow and line flow limits (aka power congestion constraints)
            Hp = self.network.H @ self.network.J @ self.out
            self.congestion_constrs = [
                Hp <= self.network.fmax,
                Hp >= -self.network.fmax
            ]
            constraints.extend(self.congestion_constrs)

        # Objective function
        obj = 0

        for tau in range(h+1):
            for g in range(N_G): # add up all generator costs
                obj += self.cgen[g, 0] + self.cgen[g, 1] * gen_dis[g, tau]
                # obj += self.cgen[g,1] * self.gen_dis[g, tau]

            for l in range(N_B):
                obj += self.cbat_d[l, tau] * bat_d[l, tau]
                obj -= self.cbat_c[l, tau] * bat_c[l, tau]

        # Solve problem
        self.prob = cp.Problem(cp.Minimize(obj), constraints)
        assert self.prob.is_dcp() and self.prob.is_dpp()

    def get_dispatch(self, agent_control: bool = True
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Determines dispatch values.

        Returns:
            x_gens: array of shape [num_gens], generator dispatch values
            x_bats: array of shape [num_bats], battery dispatch values
            price: float
        """

        # self.network.update_load_split() # update load split for randomness

        # print(self.env.demand_forecast.shape)
        # print((self.env.demand_forecast[0, 1] * self.network.load_split).shape)

        h = self.env.settlement_interval  # horizon
        N = self.network.N  # num buses
        N_D = self.network.N_D  # num loads
        N_G = self.network.N_G  # num generators
        N_B = self.network.N_B  # num batteries

        # shape [N_D, h+1]
        loads = np.empty([N_D, h + 1])
        loads[:, 0] = self.env.demand[0] * self.network.load_split
        loads[:, 1:] = self.env.demand_forecast[:h] * self.network.load_split[:, None]

        # for debugging purposes
        # loads[-1, :] = 0 * self.network.load_split

        # p_min, p_max shape: [N_D + N_G + 2*N_B, h+1]
        p_min = np.concatenate([
            loads,
            np.tile(self.network.pmin[N_D:, None], (1, h + 1))
        ])
        p_max = np.concatenate([
            loads,
            np.tile(self.network.pmax[N_D:, None], (1, h + 1))
        ])

        if not agent_control:
            p_min[-2*N_B:] = 0
            p_max[-2*N_B:] = 0

        # sanity checks
        assert np.array_equal(p_min[:N_D], p_max[:N_D])
        assert (p_min[N_D:] <= p_max[N_D:]).all()

        self.p_min.value = p_min
        self.p_max.value = p_max
        # cbats = self.network.cbat
        # costs[-1, 0] = self.env.action[0]
        # costs[-1, 1] = self.env.action[1]
        self.cgen.value = self.network.cgen

        # action has shape [2, N_B, h + 1]
        self.cbat_c.value = self.env.action[0, :, :]
        self.cbat_d.value = self.env.action[1, :, :]

        self.soc_final.value = self.env.bats_capacity / 2.
        self.soc_max.value = self.env.bats_capacity

        # print("demand: ", self.env.demand[0])
        # print("total gen: ", np.sum(self.network.pmax))

        # print("power max:", self.p_max.value[-1, :])
        # print("power min: ", self.p_min.value[-1, :])

        solve_mosek(self.prob)

        lam = -self.power_balance_constr.dual_value[0]

        if self.env.congestion:
            prices = lam + self.network.H.T @ (self.congestion_constrs[0].dual_value - self.congestion_constrs[1].dual_value)
        else:
            prices = lam * np.ones(N)

        gen_dis = self.out.value[N_D:N_D+N_G]
        bat_d = self.out.value[N_D+N_G:N_D+N_G+N_B]  # discharge
        bat_c = self.out.value[N_D+N_G+N_B:]  # charge

        print("gen dispatch: ", gen_dis)
        print("battery discharge: ", bat_d[0])
        print("battery charge: ", bat_c[0])

        return gen_dis, bat_d[0], bat_c[0], prices


class CongestedElectricityMarketEnv(Env):
    """
    Actions:
        Type: Box(2)
        Action                              Min                     Max
        a ($ / MWh)                         -Inf                    Inf
        b ($ / MWh)                         -Inf                    Inf
    Observation:
        Type: Box(9)
                                            Min                     Max
        Energy storage level (MWh)           0                     Max Capacity
        Time (fraction of day)               0                       1
        Previous charge cost ($ / MWh)       0                     Max Cost
        Previous discharge cost ($ / MWh)    0                     Max Cost
        Previous agent dispatch (MWh)        Max Charge            Max Discharge
        Previous load demand (MWh)           0                     Inf
        Load Forecast (MWh)                  0                     Inf
        Previous MOER value                  0                     Inf
        MOER Forecast                        0                     Inf
    """

    # Time step duration in hours (corresponds to 5 minutes)
    TIME_STEP_DURATION = 5 / 60
    # Each trajectories is one day (1440 minutes)
    MAX_STEPS_PER_EPISODE = 288

    # charge efficiency for all batteries
    CHARGE_EFFICIENCY = 0.95
    # discharge efficiency for all batteries
    DISCHARGE_EFFICIENCY = 0.95

    # default max production rates (MW)
    # DEFAULT_GEN_MAX_RATES = (36.8, 31.19, 3.8, 9.92, 49.0, 50.0, 50.0, 15.0, 48.5, 56.7)
    # default max discharging rates for batteries (MW), from the perspective
    #   of the market operator
    # DEFAULT_BAT_MAX_DISCHARGE = (20.0, 29.7, 7.5, 2.0, 30.0)
    # default capacity for batteries (MWh)
    # DEFAULT_BAT_CAPACITY = (80, 20, 30, 0.95, 120)
    DEFAULT_BAT_CAPACITY = [80]
    # defaul initial energy level for batteries (MWh)
    # DEFAULT_BAT_INIT_ENERGY = (40, 10, 15, 0.475, 60)
    DEFAULT_BAT_INIT_ENERGY = [40]
    # default range for max charging and discharging rates for batteries (MW)
    #   assuming symmetric range
    # DEFAULT_BAT_MAX_RATES = tuple((-val, val) for val in DEFAULT_BAT_MAX_DISCHARGE)
    # price of carbon ($ / mT of CO2), 1 mT = 1000 kg
    CARBON_PRICE = 30.85

    def __init__(self,
                 congestion: bool = True,
                #  gen_max_production: Sequence[float] = DEFAULT_GEN_MAX_RATES,
                 gens_costs: np.ndarray | None = None,
                 bats_capacity: Sequence[float] = DEFAULT_BAT_CAPACITY,
                 bats_init_energy: Sequence[float] = DEFAULT_BAT_INIT_ENERGY,
                 # bats_max_rates: Sequence[Sequence[float]] = DEFAULT_BAT_MAX_RATES,
                 bats_costs: np.ndarray | None = None,
                 randomize_costs: Sequence[str] = (),
                 use_intermediate_rewards: bool = True,
                 month: str = '2020-05',
                 moer_forecast_steps: int = 36,
                 load_forecast_steps: int = 36,
                 # settlement_interval: int = 36,
                 settlement_interval: int = 22,
                 seed: int | None = None,
                 LOCAL_FILE_PATH: str | None = None):
        """
        Args:
            congestion: bool, whether or not to incorporate congestion constraints
            gen_max_production: shape [num_gens], maximum production of each generator (MW)
            gens_costs: shape [num_gens], costs of each generator ($/MWh)
            bats_capacity: shape [num_bats], capacity of each battery (MWh)
            bats_max_rates: shape [num_bats, 2],
                maximum charge (-) and discharge (+) rates of each battery (MW)
            bats_costs: shape [num_bats - 1, 2], charging and discharging costs
                for each battery, excluding the agent-controlled battery ($/MWh)
            randomize_costs: list of str, chosen from ['gens', 'bats'], which
                costs should be randomly scaled
            use_intermediate_rewards: bool, whether or not to calculate intermediate rewards
            month: year and month to load moer and net demand data from, format YYYY-MM
            moer_forecast_steps: number of steps of MOER forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            load_forecast_steps: number of steps of load forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            settlement_interval: number of 5-minute settlements the multi-interval settlement
                in congestion case considers, default of 12 intervals (one hour lookahead)
            seed: random seed
            LOCAL_FILE_PATH: string representing the relative path of personal dataset
        """
        # if LOCAL_FILE_PATH is None:
        #     assert month in ['2019-05', '2020-05', '2021-05']  # update for future dates

        if LOCAL_FILE_PATH is None:
            assert month[:4] == '2020' # make sure the month is in 2020

        # self.num_gens = len(gen_max_production)
        # self.num_bats = len(bats_capacity)
        self.num_bats = 1

        self.randomize_costs = randomize_costs
        self.use_intermediate_rewards = use_intermediate_rewards
        self.moer_forecast_steps = moer_forecast_steps
        self.load_forecast_steps = load_forecast_steps
        self.LOCAL_PATH = LOCAL_FILE_PATH
        self.congestion = congestion
        self.settlement_interval = settlement_interval

        self.month = int(month[-2:])
        self.year = int(month[:4])
        self.date = month

        self.rng = np.random.default_rng(seed)
        rng = self.rng

        self.network = CongestedNetwork()
        N_B = self.network.N_B  # num batteries

        # generators
        # self.gen_max_production = np.array(gen_max_production, dtype=np.float32)

        # if gens_costs is None:
        #     self.gens_base_costs = rng.uniform(50, 150, size=self.num_gens)
        # else:
        #     assert len(gens_costs) == self.num_gens
        #     self.gens_base_costs = gens_costs

        # batteries
        self.bats_capacity = np.array(bats_capacity, dtype=np.float32)
        assert len(bats_init_energy) == self.num_bats
        self.bats_init_energy = np.array(bats_init_energy, dtype=np.float32)

        # for debugging purposes
        self.battery_charge = self.bats_init_energy.copy()

        assert (self.bats_init_energy >= 0).all()
        assert (self.bats_init_energy <= self.bats_capacity).all()

        # self.bats_max_rates = np.array(bats_max_rates, dtype=np.float32)
        # assert self.bats_max_rates.shape == (self.num_bats, 2)
        # assert (self.bats_max_rates[:, 0] < 0).all()
        # assert (self.bats_max_rates[:, 1] > 0).all()

        self.bats_base_costs = np.zeros((self.num_bats, 2))
        if bats_costs is None:
            self.bats_base_costs[:-1, 0] = rng.uniform(2.8, 3.2, size=self.num_bats-1)  # discharging
            self.bats_base_costs[:-1, 1] = 0.75 * self.bats_base_costs[:-1, 1]  # charging
        else:
            self.bats_base_costs[:-1] = bats_costs
        assert (self.bats_base_costs >= 0).all()

        # determine the maximum possible cost of energy ($ / MWh)
        max_cost = 1.25 * max(max(self.network.cgen[:, 1]), self.network.cgen[-1, 0], self.network.cgen[-1, 1]) # edit for general n battery case later!!!

        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0, high=max_cost, shape=(2, N_B, self.settlement_interval + 1), dtype=np.float32)

        # observation space is current energy level, current time, previous (a, b, x)
        # from dispatch and previous load demand value
        self.observation_space = spaces.Dict({
            'energy': spaces.Box(low=0, high=self.bats_capacity[-1], shape=(1,), dtype=float),
            'time': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'previous action': spaces.Box(low=0, high=np.inf, shape=(2, N_B, self.settlement_interval+1), dtype=float),
            'previous agent dispatch': spaces.Box(low=self.network.pmin[-1] * self.TIME_STEP_DURATION,
                                                  high=self.network.pmax[-1] * self.TIME_STEP_DURATION,
                                                  shape=(1,), dtype=float),
            'demand previous': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
            'demand forecast': spaces.Box(low=0, high=np.inf, shape=(load_forecast_steps,), dtype=float),
            'moer previous': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'moer forecast': spaces.Box(low=0, high=1, shape=(moer_forecast_steps,), dtype=float),
            'prices previous': spaces.Box(low=0, high=max_cost, shape=(1,), dtype=float)
        })
        self.init = False
        self.df_demand = self._get_demand_data()
        self.df_demand_forecast = self._get_demand_forecast_data()

        starttime = datetime.strptime(month, '%Y-%m').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_loader = MOERLoader(
            starttime=starttime, endtime=starttime,
            ba='SGIP_CAISO_SCE', save_dir='sustaingym/data/moer')

    # def _get_demand_data(self) -> pd.DataFrame:
    #     """Get demand data.

    #     Returns:
    #         DataFrame with demand, columns are 'HH:MM' at 5-min intervals
    #     """
    #     if self.LOCAL_PATH is not None:
    #         return pd.read_csv(self.LOCAL_PATH)
    #     else:
    #         csv_path = os.path.join('data', 'demand_data', f'CAISO-demand-{self.month}.csv.gz')
    #         bytes_data = pkgutil.get_data('sustaingym', csv_path)
    #         assert bytes_data is not None
    #         df_demand = pd.read_csv(BytesIO(bytes_data), compression='gzip', index_col=0)
    #         assert df_demand.shape == (31, 289)
    #         return df_demand / 1800.

    # def _get_demand_forecast_data(self) -> pd.DataFrame:
    #     """Get temporal load forecast data.

    #     Returns:
    #         Dataframe with demand forecast, columns are 'HH:MM' at 5-min intervals
    #     """
    #     if self.LOCAL_PATH is not None:
    #         return pd.read_csv(self.LOCAL_PATH)
    #     else:
    #         csv_path = f'data/demand_forecast_data/CAISO-demand-forecast-{self.month}.csv.gz'
    #         bytes_data = pkgutil.get_data('sustaingym', csv_path)
    #         assert bytes_data is not None
    #         df_demand_forecast = pd.read_csv(BytesIO(bytes_data), compression='gzip', index_col=0)
    #         assert df_demand_forecast.shape == (31, 289)
    #         return df_demand_forecast / 1800.

    def _get_demand_data(self) -> pd.DataFrame:
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'demand_data', f'REAL_TIME_regional_Load.csv')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            df_demand = pd.read_csv(BytesIO(bytes_data), index_col=0)

            df_demand = df_demand.loc[df_demand['Month'] == self.month]

            return df_demand

    def _get_demand_forecast_data(self) -> pd.DataFrame:
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'demand_data', f'REAL_TIME_regional_Load.csv')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            df_demand_forecast = pd.read_csv(BytesIO(bytes_data), index_col=0)

            df_demand_forecast = df_demand_forecast.loc[df_demand_forecast['Month'] == self.month]

            return df_demand_forecast

    def _generate_load_data(self, count: int) -> float:
        """Generate net demand for the time step associated with the given count.

        Args:
            count: integer representing a given time step

        Returns:
            net demand for the given time step (MWh) *currently demand*
        """
        # use demand from Zone 1
        return self.df_demand['1'].iloc[self.idx * self.MAX_STEPS_PER_EPISODE + count]

    # def _generate_load_forecast_data(self, count: int) -> float:
    #     """Generate hour ahead forecast of the net demand for the time step associated
    #     with the given count.

    #     Args:
    #         count: integer representing a given time step

    #     Returns:
    #         net demand for the given time step (MWh) *currently demand*
    #     """
    #     if count > self.df_demand_forecast.shape[1]:
    #         return np.nan
    #     return self.df_demand_forecast.iloc[self.idx, count]

    def _generate_load_forecast_data(self, count: int, lookahead_steps: int) -> np.ndarray:
        """Generate hour ahead forecast of the net demand for the time step associated
        with the given count.

        Args:
            count: integer representing a given time step
            lookahead_steps: integer representing number of time steps to look ahead

        Returns:
            array of shape [lookahead_steps], net demands for the given lookahead
        """
        num_days_total = self.df_demand_forecast.shape[0]
        start = self.idx * self.MAX_STEPS_PER_EPISODE + count

        if count > num_days_total: # outside of bounds
            return np.array([np.nan]*lookahead_steps)

        elif count + lookahead_steps > num_days_total:  # part of lookahead outside of bounds
            return np.concatenate([
                self.df_demand['1'].iloc[start:],
                [self.df_demand['1'].iloc[-1]] * (lookahead_steps + count - num_days_total)
            ])
        else:
            return self.df_demand['1'].iloc[start:start + lookahead_steps].values

    def _get_time(self) -> float:
        """Determine the fraction of the day that has elapsed based on the current
        count.

        Returns:
            fraction of the day that has elapsed
        """
        return self.count / self.MAX_STEPS_PER_EPISODE

    def reset(self, seed: int | None = None, return_info: bool = False,
              options: dict | None = None
              ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an instance of an episode for the BatteryStorageEnv.

        Args:
            seed: optional seed value for controlling seed of np_random attributes
            return_info: determines if returned observation includes additional
                info or not
            options: includes optional settings like reward type

        Returns:
            tuple containing the initial observation for env's episode
        """
        self.rng = np.random.default_rng(seed)
        rng = self.rng

        # initialize gen costs, battery charge costs, and battery discharge costs for all time steps
        # if 'gens' in self.randomize_costs:
        #     self.all_gens_costs = self.gens_base_costs[:, None] * rng.uniform(0.8, 1.25, size=(self.num_gens, self.MAX_STEPS_PER_EPISODE))
        # else:
        #     self.all_gens_costs = self.gens_base_costs[:, None] * np.ones((self.num_gens, self.MAX_STEPS_PER_EPISODE))
        # if 'bats' in self.randomize_costs:
        #     self.all_bats_costs = self.bats_base_costs[:, None, :] * rng.uniform(0.8, 1.25, size=(self.num_bats, self.MAX_STEPS_PER_EPISODE, 2))
        # else:
        #     self.all_bats_costs = self.bats_base_costs[:, None, :] * np.ones((self.num_bats, self.MAX_STEPS_PER_EPISODE, 2))

        # enforce convexity of battery bids:
        # discharge costs must be >= charge costs
        # self.all_bats_costs[:, :, 1] = np.maximum(
        #     self.all_bats_costs[:, :, 0], self.all_bats_costs[:, :, 1])

        # randomly pick a day for the episode, among the days with complete demand data

        self.idx = rng.choice(int(len(self.df_demand) / 288))
        day = self.idx + 1

        date = datetime.strptime(f'{self.date}-{day:02d}', '%Y-%m-%d').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_arr = self.moer_loader.retrieve(date).astype(np.float32)
        self.load_arr = self._generate_load_forecast_data(1, self.load_forecast_steps)

        self.action = np.zeros((2, self.network.N_B, self.settlement_interval + 1), dtype=np.float32)
        self.dispatch = np.zeros(1, dtype=np.float32)
        self.count = 0  # counter for the step in current episode
        self.battery_charge = self.bats_init_energy.copy()

        self.init = True
        self.demand = np.array([self._generate_load_data(self.count)], dtype=np.float32)
        self.demand_forecast = np.array(self._generate_load_forecast_data(self.count+1, self.load_forecast_steps), dtype=np.float32)
        self.moer = self.moer_arr[0:1, 0]
        self.moer_forecast = self.moer_arr[0, 1:self.moer_forecast_steps + 1]
        self.time = np.array([self._get_time()], dtype=np.float32)

        self.market_op = CongestedMarketOperator(self)

        # self.price = 0  # TODO: remove this line!
        self.price = np.array([self._calculate_dispatch_without_agent(self.count)[3]], dtype=np.float32)

        # print("action shape: ", self.action.shape)
        # print("dispatch shape: ", self.dispatch.shape)
        # print("demand: ", self.demand)
        # print("demand forecast: ", self.demand_forecast)
        # print("price: ", self.price)


        # set up observations
        self.obs = {
            'energy': self.battery_charge[-1:],
            'time': self.time,
            'previous action': self.action,
            'previous agent dispatch': self.dispatch,
            'demand previous': self.demand,
            'demand forecast': self.demand_forecast,
            'moer previous': self.moer,
            'moer forecast': self.moer_forecast,
            'price previous': self.price
        }

        self.intermediate_rewards = {
            'net': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'energy': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'carbon': np.zeros(self.MAX_STEPS_PER_EPISODE),
            'terminal': None
        }

        info = {
            'energy reward': None,
            'carbon reward': None,
            'terminal reward': None
        }
        return self.obs if not return_info else (self.obs, info)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Executes a single time step in the environments current trajectory.
        Assumes action is in environment's action space.

        Args:
            action: array of shape [2, N_B, h + 1], two float values representing
                charging and discharging bid prices ($/MWh) for this time step
            TODO: check the dimensions

        Returns:
            obs: dict representing the resulting state from that action
            reward: reward from action
            done: whether the episode is done
            info: additional info (currently empty)
        """

        assert self.init
        assert action.shape == (2, self.network.N_B, self.settlement_interval)

        self.count += 1

        # convert action shape to (2 * self.network.N_B, self.settlement_interval)
        # action = action.reshape((2 * self.network.N_B, self.settlement_interval))

        # ensure selling cost (discharging) is at least as large as buying cost (charging)
        # print(action.shape)
        self.action[:] = action

        for i in range(self.action.shape[1]):
            for j in range(self.action.shape[2]):
                if action[1, i, j] < action[0, i, j]:
                    self.action[1, i, j] = action[0, i, j]

        # self.gens_costs = self.all_gens_costs[:, self.count]
        # self.bats_costs = self.all_bats_costs[:, self.count]
        # self.bats_costs[-1] = self.action

        # assert (self.bats_costs[:, 1] >= self.bats_costs[:, 0]).all()

        self.demand[:] = self._generate_load_data(self.count)
        self.moer[:] = self.moer_arr[self.count:self.count + 1, 0]

        # rates in MW
        self.bats_max_charge = np.maximum(
            self.network.pmin[-1],
            -(self.bats_capacity - self.battery_charge) / (self.TIME_STEP_DURATION * self.CHARGE_EFFICIENCY))
        self.bats_max_discharge = np.minimum(
            self.network.pmax[-1],
            self.battery_charge * self.DISCHARGE_EFFICIENCY / self.TIME_STEP_DURATION)

        _, x_bats, price = self.market_op.get_dispatch(self.congestion)
        x_agent = x_bats[-1]
        self.dispatch[:] = x_agent
        self.price[:] = price

        # update battery charges
        charging = (x_bats < 0)
        self.battery_charge[charging] -= self.CHARGE_EFFICIENCY * x_bats[charging]
        self.battery_charge[~charging] -= (1. / self.DISCHARGE_EFFICIENCY) * x_bats[~charging]
        self.battery_charge[:] = self.battery_charge.clip(0, self.bats_capacity)

        # get forecasts for next time step
        self.time[:] = self._get_time()
        self.demand_forecast[:] = self._generate_load_forecast_data(self.count + 1, self.load_forecast_steps)
        self.moer_forecast[:] = self.moer_arr[self.count, 1:self.moer_forecast_steps + 1]

        energy_reward = price * x_agent
        carbon_reward = self.CARBON_PRICE * self.moer[0] * x_agent
        reward = energy_reward + carbon_reward

        self.intermediate_rewards['energy'][self.count] = energy_reward
        self.intermediate_rewards['carbon'][self.count] = carbon_reward
        self.intermediate_rewards['net'][self.count] = reward

        done = (self.count + 1 >= self.MAX_STEPS_PER_EPISODE)

        if done:
            terminal_cost = self._calculate_terminal_cost(self.battery_charge[-1])
            reward -= terminal_cost

            self.intermediate_rewards['terminal'] = terminal_cost
            self.intermediate_rewards['net'][self.count] = reward

            if not self.use_intermediate_rewards:
                reward = np.sum(self.intermediate_rewards['net'])
        else:
            if not self.use_intermediate_rewards:
                reward = 0
            terminal_cost = None

        info = {
            'energy reward': energy_reward,
            'carbon reward': carbon_reward,
            'terminal cost': terminal_cost,
        }
        return self.obs, reward, done, info

    def _calculate_dispatch_without_agent(self, count: int
                                          ) -> tuple[np.ndarray, np.ndarray, float]:
        """Calculates market price and dispatch at given time step, without
        agent participation.

        The only state variable that is modified is self.demand.

        Args:
            count: time step

        Returns:
            x_gens: array of shape [num_gens], generator dispatch values
            x_bats: array of shape [num_bats], battery dispatch values
            price: float
        """
        # self.gens_costs = self.all_gens_costs[:, count]
        # self.bats_costs = self.all_bats_costs[:, count]

        # update charging range for each battery
        # self.bats_max_charge = np.maximum(
        #     self.network.pmin[-1],
        #     -(self.bats_capacity - self.battery_charge) / (self.TIME_STEP_DURATION * self.CHARGE_EFFICIENCY))
        # self.bats_max_discharge = np.minimum(
        #     self.network.pmax[-1],
        #     self.battery_charge * self.DISCHARGE_EFFICIENCY / self.TIME_STEP_DURATION)

        # prevent agent-battery from participating in market
        # self.bats_max_charge[-1] = 0
        # self.bats_max_discharge[-1] = 0

        self.demand[:] = self._generate_load_data(count)
        x_gens, x_bat_d, x_bat_c, prices = self.market_op.get_dispatch(agent_control=False)

        print(prices)

        # sanity checks
        assert np.all(0 <= prices), 'prices should be nonnegative'
        assert np.isclose(x_bat_d, 0).all() and np.isclose(x_bat_c, 0).all()

        x_bat_d[:] = np.zeros(self.network.N_B)
        x_bat_c[:] = np.zeros(self.network.N_B)

        return x_gens, x_bat_d, x_bat_c, prices

    def _calculate_prices_without_agent(self) -> np.ndarray:
        """Calculates market prices, as if the agent did not participate.

        The only state variable that is modified is self.demand.

        Returns:
            np.ndarray, shape [num_steps], type float64
        """
        battery_charge_save = self.battery_charge.copy()
        prices = np.zeros((self.network.N, self.MAX_STEPS_PER_EPISODE))

        # get prices from market for all time steps
        for count in range(self.MAX_STEPS_PER_EPISODE):
            _, x_bat_d, x_bat_c, prices = self._calculate_dispatch_without_agent(count)
            prices[:, count] = prices

            # update battery charges
            self.battery_charge += self.CHARGE_EFFICIENCY * x_bat_c
            self.battery_charge -= (1. / self.DISCHARGE_EFFICIENCY) * x_bat_d
            self.battery_charge[:] = self.battery_charge.clip(0, self.bats_capacity)

        self.battery_charge = battery_charge_save
        return prices

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float) -> dict[str, np.ndarray]:
        """Calculates optimal episode, under price-taking assumption.

        Args:
            prices: array of shape [num_steps], fixed prices at each time step
            init_charge: float in [0, self.bats_capacity[-1]],
                initial energy level for agent battery
            final_charge: float, minimum final energy level of agent battery

        Returns:
            results dict, keys are ['rewards', 'dispatch', 'energy', 'net_prices'],
                values are arrays of shape [num_steps].
                'net_prices' is prices + carbon cost
        """
        c = cp.Variable(self.MAX_STEPS_PER_EPISODE)  # charging (in MWh)
        d = cp.Variable(self.MAX_STEPS_PER_EPISODE)  # discharging (in MWh)
        x = d * self.DISCHARGE_EFFICIENCY - c / self.CHARGE_EFFICIENCY  # dispatch (in MWh)
        delta_energy = cp.cumsum(c) - cp.cumsum(d)

        constraints = [
            c[0] == 0, d[0] == 0,  # do nothing on 1st time step

            0 <= init_charge + delta_energy,
            init_charge + delta_energy <= self.bats_capacity[-1],

            # rate constraints
            0 <= c,
            c / self.CHARGE_EFFICIENCY <= -self.network.pmax[-1] * self.TIME_STEP_DURATION,
            0 <= d,
            d * self.DISCHARGE_EFFICIENCY <= self.network.pmin[-1] * self.TIME_STEP_DURATION
        ]
        if final_charge > 0:
            constraints.append(final_charge <= init_charge + delta_energy[-1])

        moers = self.moer_arr[:-1, 0]
        net_price = prices + self.CARBON_PRICE * moers
        obj = net_price @ x
        prob = cp.Problem(objective=cp.Maximize(obj), constraints=constraints)
        assert prob.is_dcp() and prob.is_dpp()
        solve_mosek(prob)

        rewards = net_price * x.value
        energy = init_charge + delta_energy.value
        return dict(rewards=rewards, dispatch=x.value, energy=energy, net_prices=net_price)

    def _calculate_terminal_cost(self, agent_energy_level: float) -> float:
        """Calculates terminal cost term.

        Args:
            agent_energy_level: initial energy level (MWh) in the
                agent-controlled battery

        Returns:
            terminal cost for the current episode's reward function,
                always nonnegative
        """
        desired_charge = self.bats_init_energy[-1]
        if agent_energy_level >= desired_charge:
            return 0

        prices = self._calculate_prices_without_agent()
        future_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=agent_energy_level, final_charge=desired_charge)['rewards']
        potential_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=desired_charge, final_charge=desired_charge)['rewards']

        # added factor to ensure terminal costs motivates charging actions
        penalty = max(0, prices[-1] * (desired_charge - agent_energy_level) / self.CHARGE_EFFICIENCY)

        future_return = np.sum(future_rewards)
        potential_return = np.sum(potential_rewards)
        return max(0, potential_return - future_return) + penalty

    def render(self):
        raise NotImplementedError

    def close(self):
        return