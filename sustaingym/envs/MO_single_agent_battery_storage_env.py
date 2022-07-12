"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import cvxpy as cp
import gym
from gym import spaces
import pandas as pd
import pkgutil
from io import StringIO
import sys
sys.path.append('../')

class MarketOperator():
    """
    MarketOperator class.
    """
    def __init__(self, env: BatteryStorageInGridEnv):
        """
        Constructs instance of MarketOperator class.
        """
        self.env = env
    
    
        

class BatteryStorageInGridEnv(gym.Env):
    """
    Actions:
        Type: Box(2)
        Action                              Min                     Max
        a ($ / MWh)                         -Inf                    Inf
        b ($ / MWh)                         -Inf                    Inf
    Observation:
        Type: Box(2)
                                            Min                     Max
        Energy storage level (MWh)          -Inf                    Inf
        Current Electricity Price ($/MWh)   -Inf                    Inf
        Time (fraction of day)               0                       1
    """