"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple

import math
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

        Args:
            env: instance of BatteryStorageInGridEnv class
        """
        self.env = env
    
    def get_dispatch(self) -> Tuple:
        """
        Returns dispatch values.

        Returns:
            Tuple: (generator dispatch values, battery dispatch values, and agent battery
            dispath value)
        """
        x_i = cp.Variable(self.env.num_gens)
        x_j = cp.Variable(self.env.num_batteries)
        x_tilda = cp.Variable(1)
        a_i = self.env.gen_costs

        constraints = []
        time_step = self.env.TIME_STEP_DURATION / 60

        for i in range(self.env.num_gens):
            constraints.append(self.env.gen_max_production[i]*time_step >= x_i[i] >= 0)
        
        for i in range(self.env.num_batteries):
            constraints.append(self.env.battery_max_discharge[i]*time_step >= x_j[i] >= 
                               self.env.battery_max_charge[i]*time_step)
        constraints.append(self.env.agent_battery_max_discharge*time_step >= x_tilda >=
                          self.env.agent_battery_max_charge*time_step)
        constraints.append(x_i.sum() + x_j.sum() + x_tilda ==
                           self.env._generate_load_data())

        # ask Chris how to deal with the piecewise functions for batteries in the
        # minimization problem formulation
        objective = cp.Problem(cp.Minimize(a_i.T@x_i), constraints)

    

        

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

    # Charge efficiency
    CHARGE_EFFICIENCY = 0.4
    # Discharge efficiency
    DISCHARGE_EFFICIENCY = 0.6
    # Time step duration (min)
    TIME_STEP_DURATION = 5
    # Each trajectories is one day (1440 minutes)
    MAX_STEPS_PER_EPISODE = 288
    # probability for running price average in state/observation space
    eta = 0.5

    def __init__(self, render_mode: Optional[str] = None, num_gens: int | None = None,
        gen_max_production: np.ndarray | None = None, gen_costs: np.ndarray | None
         = None, num_batteries: int | None = None, battery_max_capacity: np.ndarray | None
         = None, battery_min_capacity: np.ndarray | None = None,
         battery_max_discharge: np.ndarray | None = None,
         battery_max_charge: np.ndarray | None = None,
         battery_charge_costs: np.ndarray | None = None,
         battery_discharge_costs: np.ndarray | None = None,
         agent_battery_max_capacity: float | None = None,
         agent_battery_min_capacity: float | None = None,
         agent_battery_max_discharge: float | None = None,
         agent_battery_max_charge: float | None = None,
         agent_battery_max_cost: float | None = None):
        """
        Constructs instance of BatteryStorageInGridEnv class.

        Args:
            num_gens: number of generators
            gen_max_production: maximum production of each generator (MW)
            gen_costs: costs of each generator ($/MWh)
            num_batteries: number of batteries
            battery_max_capacity: maximum capacity of each battery (MWh)
            battery_min_capacity: minimum capacity of each battery (MWh)
            battery_max_discharge: maximum discharge rate of each battery (MW)
            battery_max_charge: maximum charge rate of each battery (MW)
            battery_charge_costs: cost of charging for each battery ($/MWh)
            battery_discharge_costs: cost of discharging for each battery ($/MWh)
            agent_battery_max_capacity: maximum capacity of agent's battery (MWh)
            agent_battery_min_capacity: minimum capacity of agent's battery (MWh)
            agent_battery_max_discharge: maximum discharge rate of agent's battery (MW)
            agent_battery_max_charge: maximum charge rate of agent's battery (MW)
            agent_battery_max_cost: maximum cost for charging or discharging
            for agent's battery ($/MWh)
        """
        if num_gens == None:
            self.num_gens = 10
        else:
            self.num_gens = num_gens

        if gen_max_production == None:
            self.gen_max_production = np.random.uniform(5, 10, size=(self.num_gens,))
        else:
            assert len(gen_max_production) == self.num_gens
            self.gen_max_production = gen_max_production
        
        if gen_costs == None:
            self.gen_costs = np.random.uniform(0, 5, size=(self.num_gens,))
        else:
            assert len(gen_costs) == self.num_gens
            self.gen_costs = gen_costs
        
        if num_batteries == None:
            self.num_batteries = 4
        else:
            self.num_batteries = num_batteries
        
        if battery_max_capacity == None:
            self.battery_max_capacity = np.random.uniform(30, 50, size=(self.num_batteries,))
        else:
            assert len(battery_max_capacity) == self.num_batteries
            self.battery_max_capacity = battery_max_capacity
        
        if battery_min_capacity == None:
            self.battery_min_capacity = np.array([0] * self.num_batteries)
        else:
            assert len(battery_min_capacity) == self.num_batteries
            self.battery_min_capacity = battery_min_capacity
        
        if battery_max_discharge == None:
            self.battery_max_discharge = np.random.uniform(2, 4, size=(self.num_batteries,))
        else:
            assert len(battery_max_discharge) == self.num_batteries
            self.battery_max_discharge = battery_max_discharge
        
        if battery_max_discharge == None:
            self.battery_max_discharge = np.random.uniform(2, 4, size=(self.num_batteries,))
        else:
            assert len(battery_max_discharge) == self.num_batteries
            self.battery_max_discharge = battery_max_discharge

        if battery_max_charge == None:
            self.battery_max_charge = np.random.uniform(-3, -1, size=(self.num_batteries,))
        else:
            assert len(battery_max_charge) == self.num_batteries
            self.battery_max_charge = battery_max_charge
        
        if battery_charge_costs == None:
            self.battery_charge_costs = np.random.uniform(0, 5, size=(self.num_batteries,))
        else:
            assert len(battery_charge_costs) == self.num_batteries
            self.battery_charge_costs = battery_charge_costs
        
        if battery_discharge_costs == None:
            self.battery_discharge_costs = np.random.uniform(0, 4, size=(self.num_batteries,))
        else:
            assert len(battery_discharge_costs) == self.num_batteries
            self.battery_charge_costs = battery_discharge_costs
        
        self.battery_init_charge = np.zeros((self.num_batteries, ))

        for i in range(self.num_batteries):
            self.battery_init_charge[i] = (self.battery_min_capacity[i] +
                                            self.battery_max_capacity[i]) / 2.0
        
        if agent_battery_max_capacity == None:
            self.agent_battery_max_capacity = np.random.uniform(30, 50)
        else:
            self.agent_battery_max_capacity = agent_battery_max_capacity
        
        if agent_battery_min_capacity == None:
            self.agent_battery_min_capacity = 0.0
        else:
            self.agent_battery_min_capacity = agent_battery_min_capacity
        
        self.agent_init_battery_cahrge = (self.agent_battery_min_capacity +
                                        self.agent_battery_max_capacity) / 2.0
        
        if agent_battery_max_discharge == None:
            self.agent_battery_max_discharge = np.random.uniform(2, 4)
        else:
            self.agent_battery_max_discharge = agent_battery_max_discharge
        
        if agent_battery_max_charge == None:
            self.agent_battery_max_charge = np.random.uniform(-3, -1)
        else:
            self.agent_battery_max_charge = agent_battery_max_charge
        
        if agent_battery_max_cost == None:
            self.agent_battery_max_cost = np.random.uniform(10, 20)
        else:
            self.agent_battery_max_cost = agent_battery_max_cost
        
        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0,
                                       high=self.agent_battery_max_cost, shape=(2, ),
                                       dtype=np.float32)
        # observation space is current energy level and current price
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3, ), dtype=np.float32
        )
        self.init = False
        self.count = 0
    
    def _generate_load_data(self) -> float:
        return 30*math.sin(2*math.pi*self.count/288) + 30

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> tuple:
        """
        Initialize or restart an instance of an episode for the BatteryStorageEnv.

        Args:
            seed: optional seed value for controlling seed of np_random attributes
            return_info: determines if returned observation includes additional
            info or not
            options: includes optional settings like reward type
        Returns:
            tuple containing the initial observation for env's episode
        """


