"""
The module implements the BatteryStorageInGridEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import pkgutil
from io import StringIO
import sys
sys.path.append('../')

import cvxpy as cp
from gym import Env, spaces
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class MarketOperator:
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

        # x: energy in MWh, positive means generation, negative (for battery) means charging
        self.x = cp.Variable(env.num_gens + env.num_bats)
        x_gens = self.x[:env.num_gens]
        x_bats = self.x[env.num_gens:]

        # time-dependent parameters
        self.gen_max_production = cp.Parameter(env.num_gens, nonneg=True)
        self.bats_max_charge = cp.Parameter(env.num_bats, nonpos=True)  # rate in MW
        self.bats_max_discharge = cp.Parameter(env.num_bats, nonneg=True)
        self.bats_charge_costs = cp.Parameter(env.num_bats, nonneg=True)
        self.bats_discharge_costs = cp.Parameter(env.num_bats, nonneg=True)
        self.gen_costs = cp.Parameter(env.num_gens, nonneg=True)
        self.load = cp.Parameter(nonneg=True)

        time_step = env.TIME_STEP_DURATION / 60  # in hours
        constraints = [
            cp.sum(self.x) == self.load,  # supply = demand

            0 <= x_gens,
            x_gens <= self.gen_max_production * time_step,

            self.bats_max_charge * time_step <= x_bats,
            x_bats <= self.bats_max_discharge * time_step,
        ]

        obj = self.gen_costs @ x_gens
        obj += cp.sum(cp.maximum(cp.multiply(self.bats_discharge_costs, x_bats), cp.multiply(self.bats_charge_costs, x_bats)))
        self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)
        assert self.prob.is_dcp()
        assert self.prob.is_dpp()

    def get_dispatch(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Returns dispatch values.

        Returns:
            Tuple: (generator dispatch values, battery dispatch values, and agent battery
            dispath value)
        """

        # print("step: ", self.env.count)
        # print("gen max production: ", self.env.gen_max_production)
        # # print("battery max charge: ", self.env.bats_max_charge)
        # print("battery max discharge: ", self.env.bats_max_discharge)
        # print("battery discharge costs: ", self.env.bats_discharge_costs)
        # print("battery charge costs: ", self.env.bats_charge_costs)
        # print("gen costs: ", self.env.gen_costs)
        # print("load: ", self.env.load_demand)

        self.gen_max_production.value = self.env.gen_max_production
        self.gen_costs.value = self.env.gen_costs
        self.bats_max_charge.value = self.env.bats_max_charge
        self.bats_max_discharge.value = self.env.bats_max_discharge
        self.bats_charge_costs.value = self.env.bats_charge_costs
        self.bats_discharge_costs.value = self.env.bats_discharge_costs
        self.load.value = self.env.load_demand
        self.prob.solve()
        # print("status: ", self.prob.status)
        # print("x value: ", self.x.value)
        price = -1*self.prob.constraints[0].dual_value # negative because of minimizing objective in LP
        # print("price:", price)
        x_gens = self.x.value[:self.env.num_gens]
        x_bats = self.x.value[self.env.num_gens:self.env.num_gens + self.env.num_bats]
        return x_gens, x_bats, price
    
    def get_dispatch_no_agent(self):
        """
        TODO
        """
        # print("step: ", self.env.count)
        # print("gen max production: ", self.env.gen_max_production)
        # # print("battery max charge: ", self.env.bats_max_charge)
        # print("battery max discharge: ", self.env.bats_max_discharge)
        # print("battery discharge costs: ", self.env.bats_discharge_costs)
        # print("battery charge costs: ", self.env.bats_charge_costs)
        # print("gen costs: ", self.env.gen_costs)
        # print("load: ", self.env.load_demand)

        bats_max_charge = self.env.bats_max_charge_cpy
        bats_max_discharge = self.env.bats_max_discharge_cpy
        # restrict agent to not charge or discharge
        bats_max_charge[-1] = 0
        bats_max_discharge[-1] = 0

        bats_charge_costs = self.env.bats_charge_costs_cpy
        bats_discharge_costs = self.env.bats_discharge_costs_cpy
        # restrict agent to not charge or discharge
        bats_charge_costs[-1] = 10**20
        bats_discharge_costs[-1] = 0

        self.gen_max_production.value = self.env.gen_max_production
        self.gen_costs.value = self.env.gen_costs_cpy

        self.bats_max_charge.value = bats_max_charge
        self.bats_max_discharge.value = bats_max_discharge

        self.bats_charge_costs.value = bats_charge_costs
        self.bats_discharge_costs.value = bats_discharge_costs

        self.load.value = self.env.load_demand_cpy
        self.prob.solve()
        # print("status: ", self.prob.status)
        # print("x value: ", self.x.value)
        price = -1*self.prob.constraints[0].dual_value # negative because of minimizing objective in LP
        # print("price:", price)
        x_gens = self.x.value[:self.env.num_gens]
        x_bats = self.x.value[self.env.num_gens:self.env.num_gens + self.env.num_bats]
        return x_gens, x_bats, price


class BatteryStorageInGridEnv(Env):
    """
    Actions:
        Type: Box(2)
        Action                              Min                     Max
        a ($ / MWh)                         -Inf                    Inf
        b ($ / MWh)                         -Inf                    Inf
    Observation:
        Type: Box(5)
                                            Min                     Max
        Energy storage level (MWh)           0                     Max Capacity
        Time (fraction of day)               0                       1
        Previous charge cost ($ / MWh)       0                     Max Cost
        Previous discharge cost ($ / MWh)    0                     Max Cost
        Previous agent dispatch (MWh)        Max Charge            Max Discharge
        Previous load demand (MWh)           0                     55
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
    # default max production rates (MW)
    DEFAULT_GEN_MAX_PRODUCTION = (36.8, 31.19, 3.8, 9.92, 49.0, 50.0, 50.0, 15.0, 48.5, 56.7)
    # default max discharging rates for batteries (MW)
    DEFAULT_BAT_MAX_DISCHARGE = (30.0, 20.0, 29.7, 7.5, 2.0)
    # default max capacity for batteries (MWh)
    DEFAULT_BAT_CAPACITY = (120.0, 80.0, 118.8, 30.0, 8.0)


    def __init__(self, render_mode: str | None = None, num_gens: int = 10,
        gen_max_production: tuple = DEFAULT_GEN_MAX_PRODUCTION, gen_costs: np.ndarray | None
         = None, num_bats: int = 5, date: str = '2019-05',
         battery_capacity: np.ndarray | None
         = None,
         bats_max_discharge_range: np.ndarray | None = None,
         bats_charge_costs: np.ndarray | None = None,
         bats_discharge_costs: np.ndarray | None = None,
         seed: int | None = None, LOCAL_FILE_PATH: str | None = None):
        """
        Constructs instance of BatteryStorageInGridEnv class.

        Args:
            num_gens: number of generators
            gen_max_production: shape [num_gens],
                maximum production of each generator (MW)
            gen_costs: costs of each generator ($/MWh)
            num_bats: number of batteries (last battery is agent-controlled)
            battery_capacity: shape [num_bats], capacity of each battery (MWh)
            bats_max_discharge_range: shape [num_bats, 2],
                maximum charge (-) and discharge (+) rates of each battery (MW)
            bats_charge_costs: shape [num_bats - 1], charging cost for each battery,
                excluding the agent-controlled battery ($/MWh)
            bats_discharge_costs: shape [num_bats - 1], discharging cost for each battery,
                excluding the agent-controlled battery ($/MWh)
            seed: random seed
        """
        if LOCAL_FILE_PATH is not None:
            assert date in ['2019-05', '2020-05', '2021-05'] # update for future dates
        
        self.date = date

        self.LOCAL_PATH = LOCAL_FILE_PATH

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        rng = self.rng

        self.num_gens = num_gens

        # if gen_max_production is None:
        #     self.gen_max_production = np.zeros((self.num_gens,))
        #     self.gen_max_production[:3] = rng.uniform(3, 5, size=(3,))
        #     self.gen_max_production[3:6] = rng.uniform(6, 7, size=(3,))
        #     self.gen_max_production[6:] = rng.uniform(8, 15, size=(self.num_gens - 6,))
        # else:
        assert len(gen_max_production) == self.num_gens
        self.gen_max_production = np.array(gen_max_production)
        
        self.init_gen_max_production = self.gen_max_production.copy()

        if gen_costs is None:
            self.init_gen_costs = rng.uniform(50, 100, size=(self.num_gens,))
        else:
            assert len(gen_costs) == self.num_gens
            self.init_gen_costs = gen_costs

        self.num_bats = num_bats

        if battery_capacity is None:
            self.battery_capacity = rng.uniform(5, 10, size=(self.num_bats,))
        else:
            assert len(battery_capacity) == self.num_bats
            self.battery_capacity = battery_capacity

        if bats_max_discharge_range is None:
            self.bats_max_discharge_range = np.zeros([self.num_bats, 2])
            self.bats_max_discharge_range[:, 0] = rng.uniform(-1.5, -0.5, size=[self.num_bats])
            self.bats_max_discharge_range[:, 1] = rng.uniform(0.75, 2, size=[self.num_bats])
        else:
            assert bats_max_discharge_range.shape == (self.num_bats, 2)
            assert (bats_max_discharge_range[:, 0] < 0).all()
            assert (bats_max_discharge_range[:, 1] > 0).all()
            self.bats_max_discharge_range = bats_max_discharge_range

        if bats_charge_costs is None:
            bats_charge_costs = np.zeros((self.num_bats, ))
            bats_charge_costs[:-1] = rng.uniform(2, 4, size=self.num_bats-1)
        else:
            assert len(bats_charge_costs) == self.num_bats - 1
            bats_charge_costs[:-1] = bats_charge_costs
        
        self.init_bats_charge_costs = bats_charge_costs.copy()

        self.bats_discharge_costs = np.zeros((self.num_bats, ))
        if bats_discharge_costs is None:
            self.bats_discharge_costs[:-1] = rng.uniform(5, 10, size=self.num_bats-1)
        else:
            assert len(bats_discharge_costs) == self.num_bats - 1
            self.bats_discharge_costs[:-1] = bats_discharge_costs
        
        self.init_bats_discharge_costs = self.bats_discharge_costs.copy()

        self.battery_charge = np.zeros((self.num_bats, ))

        for i in range(self.num_bats):
            self.battery_charge[i] = (0 + self.battery_capacity[i]) / 2.0
        
        self.init_battery_charge = self.battery_charge.copy()

        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)
        # observation space is current energy level, current time, previous (a, b, x)
        # from dispatch and previous load demand value
        time_step = self.TIME_STEP_DURATION / 60
        self.observation_space = spaces.Dict({
            "current_energy_level": spaces.Box(low=0, high=self.battery_capacity[-1],
                                                shape=(1,), dtype=float),
            "current_time": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            "previous action": spaces.Box(low=0, high=np.inf, shape=(2,), dtype=float),
            "previous agent dispatch": spaces.Box(low=self.bats_max_discharge_range[-1, 0]*time_step,
                                                high=self.bats_max_discharge_range[-1, 1]*time_step,
                                                shape=(1,), dtype=float),
            "previous load demand": spaces.Box(low=0, high=55,
                                                shape=(1,), dtype=float)
        })
        self.init = False
        self.count = 1
        self.market_op = MarketOperator(self)
        self.df_load = self._get_load_data()
        self.df_moer = self._get_moer_data()
    
    def _get_load_data(self) -> pd.DataFrame:
        """
        Generates temporal load data.

        Args:
            N/A
        Returns:
            pandas dataframe containing load data
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            bytes_data = pkgutil.get_data(__name__, 'data/CAISO-netdemand-' + self.date + 
                                        '.csv')
            s = bytes_data.decode('utf-8')
            data = StringIO(s)
            return pd.read_csv(data)
    
    def _get_moer_data(self) -> pd.DataFrame:
        """
        Generates temporal moer data.

        Args:
            N/A
        Returns:
            pandas dataframe containing moer data
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            bytes_data = pkgutil.get_data(__name__, 'data/SGIP_CAISO_SCE_' + self.date + 
                                        '.csv')
            s = bytes_data.decode('utf-8')
            data = StringIO(s)
            return pd.read_csv(data)
    
    def _generate_load_data(self, count: int) -> float:
        """
        TODO
        """
        if self.count == 1:
            # random index for the day in May
            pos_ids = [idx for idx in np.arange(len(
                self.df_load.iloc[:,0])) if not pd.isnull(self.df_load.iloc[
                    idx, :-1]).any()]
            self.idx = self.rng.choice(pos_ids)
        if self.LOCAL_PATH is not None:
            return self.df_load.iloc[self.idx, count]
        return self.df_load.iloc[self.idx, count] / 6000.0 # scale to small grid scale
    
    def _generate_moer_data(self, count: int) -> float:
        """
        TODO
        """
        if self.LOCAL_PATH is not None:
            return self.df_moer.iloc[self.idx, count-1]
        return self.df_moer.iloc[self.MAX_STEPS_PER_EPISODE*(self.idx+1) + (count-1), 1]

    def _generate_load_data2(self) -> float:
        # TODO: describe this function
        return np.sin(2 * np.pi * self.count / 288) + 3.5
    
    def _get_time(self) -> float:
        # Describe this function
        return self.count / self.MAX_STEPS_PER_EPISODE

    def reset(self, *,
              return_info: bool = False,
              options: dict | None = None) -> dict[str, Any] | tuple[dict[str,
               Any], dict[str, Any]]:
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

        # initialize gen costs for all time steps
        self.all_gen_costs = np.zeros((self.num_gens, self.MAX_STEPS_PER_EPISODE))

        for i in range(self.num_gens):
            self.all_gen_costs[i, :] = self.rng.uniform(0.8*self.init_gen_costs[i],
                                        1.25*self.init_gen_costs[i], size=self.MAX_STEPS_PER_EPISODE)
        
        # initialize battery charge costs for all time steps
        self.all_bats_charge_costs = np.zeros((self.num_bats, self.MAX_STEPS_PER_EPISODE))

        for i in range(self.num_bats):
            self.all_bats_charge_costs[i, :] = self.rng.uniform(0.8*self.init_bats_charge_costs[i],
                                        1.25*self.init_bats_charge_costs[i], size=self.MAX_STEPS_PER_EPISODE)

        # initialize battery discharge costs for all time steps
        self.all_bats_discharge_costs = np.zeros((self.num_bats, self.MAX_STEPS_PER_EPISODE))

        for i in range(self.num_bats):
            self.all_bats_discharge_costs[i, :] = self.rng.uniform(0.8*self.init_bats_discharge_costs[i],
                                        1.25*self.init_bats_discharge_costs[i], size=self.MAX_STEPS_PER_EPISODE)

        self.gen_costs = self.all_gen_costs[:, 0]
        self.bats_charge_costs = self.all_bats_charge_costs[:, 0]
        self.bats_discharge_costs = self.all_bats_discharge_costs[:, 0]

        self.a = 0.0
        self.b = 0.0
        self.dispatch = 0.0
        self.count = 1  # counter for the step in current episode
        self.gen_max_production = self.init_gen_max_production.copy()
        self.battery_charge = self.init_battery_charge.copy()

        # enforce convexity for battery bids
        self.bats_discharge_costs = np.maximum(self.bats_discharge_costs, self.bats_charge_costs)

        self.reward_type = 0  # default reward type without moving price average
        if options and 'reward' in options.keys():
            if options.get('reward') == 1:
                self.reward_type = 1

        self.init = True
        self.load_demand = self._generate_load_data(self.count)
        obs = {
            "current_energy_level": np.array([self.battery_charge[-1]], dtype=np.float32),
            "current_time": np.array([self._get_time()], dtype=np.float32),
            "previous action": np.array([self.a, self.b], dtype=np.float32),
            "previous agent dispatch": np.array([self.dispatch], dtype=np.float32),
            "previous load demand": np.array([self.load_demand], dtype=np.float32),
            "previous moer value": np.array([0], dtype=np.float32)
        }
        # TODO: figure what additional info could be helpful here
        info = {"curr info": None}
        return obs if not return_info else (obs, info)

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any],
              float, bool, dict[Any, Any]]:
        """
        Executes a single time step in the environments current trajectory.

        Args:
            action: array of two float values representing the charging and discharging
            costs during this time step
        Returns:
            dict representing the resulting state from that action
        """
        assert self.init
        assert self.reward_type == 0
        assert self.action_space.contains(action)

        self.count += 1

        # ensure selling cost (a) for charge is at least as large as buying cost (b)
        if action[0] < action[1]:
            # print('Warning: selling cost (a) is less than buying cost (b)')
            action[0] = action[1]

        self.gen_costs = self.all_gen_costs[:, self.count - 1]
        self.bats_charge_costs = self.all_bats_charge_costs[:, self.count - 1]
        self.bats_discharge_costs = self.all_bats_discharge_costs[:, self.count - 1]
        self.bats_charge_costs[-1] = action[1]
        self.bats_discharge_costs[-1] = action[0]

        # print('charge costs: ', self.bats_charge_costs)
        # print('discharge costs before: ', self.bats_discharge_costs)
        # enforce convexity for battery bids
        self.bats_discharge_costs = np.maximum(self.bats_discharge_costs, self.bats_charge_costs)
        # print('discharge costs after: ', self.bats_discharge_costs)

        time_step = self.TIME_STEP_DURATION / 60  # min -> hr

        prev_a, prev_b = self.a, self.b
        self.a, self.b = action[0], action[1]

        prev_dispatch = self.dispatch
        
        self.bats_max_charge = np.maximum(
            self.bats_max_discharge_range[:, 0],
            -(self.battery_capacity - self.battery_charge) / (time_step * self.CHARGE_EFFICIENCY))
        self.bats_max_discharge = np.minimum(
            self.bats_max_discharge_range[:, 1],
            self.battery_charge * self.DISCHARGE_EFFICIENCY / time_step)
        
        # print("agent max discharge: ", self.bats_max_discharge[-1])
        # print("agent max charge: ", self.bats_max_charge[-1])
        # print("agent curr charge: ", self.battery_charge[-1])
        # print("agent charge cost: ", self.bats_charge_costs[-1])
        # print("agent discharge cost: ", self.bats_discharge_costs[-1])
        
        _, x_bats, price = self.market_op.get_dispatch()
        x_agent = x_bats[-1]

        self.dispatch = np.array([x_agent], dtype=np.float32)
        prev_load_demand = self.load_demand
        self.load_demand = self._generate_load_data(self.count)

        for i in range(self.num_bats):
            if 0 <= x_bats[i]:
                self.battery_charge[i] -= (1. / self.DISCHARGE_EFFICIENCY) * x_bats[i]
                # ensure non-negative battery charge
                if self.battery_charge[i] < 0:
                    self.battery_charge[i] = 0.0
            else:
                self.battery_charge[i] -= self.CHARGE_EFFICIENCY * x_bats[i]

        done = (self.count >= self.MAX_STEPS_PER_EPISODE)
        moer = self._generate_moer_data(self.count)
        obs = {
            "current_energy_level": np.array([self.battery_charge[-1]], dtype=np.float32),
            "current_time": np.array([self._get_time()], dtype=np.float32),
            "previous action": np.array([prev_a, prev_b], dtype=np.float32),
            "previous agent dispatch": np.array([prev_dispatch], dtype=np.float32),
            "previous load demand": np.array([prev_load_demand], dtype=np.float32),
            "previous moer value": np.array([moer], dtype=np.float32)
        }
        # TODO: figure what additional info could be helpful here
        info = {"curr info": None}
        reward = (price + 30.85 * moer) * x_agent
        return obs, reward, done, info
    
    def _calculate_off_optimal_total_episode_reward(self) -> float:
        """
        TODO
        """

        prices = np.zeros((self.MAX_STEPS_PER_EPISODE,))
        self.bats_max_charge_cpy = self.bats_max_charge.copy()
        self.bats_max_discharge_cpy = self.bats_max_discharge.copy()
        self.battery_charge_cpy = self.battery_charge.copy()

        
        # get prices from market for all time steps

        for i in range(self.MAX_STEPS_PER_EPISODE):
            count = i + 1
            time_step = self.TIME_STEP_DURATION / 60  # min -> hr

            self.bats_charge_costs_cpy = self.all_bats_charge_costs[:, count - 1]
            self.bats_discharge_costs_cpy = self.all_bats_discharge_costs[:, count - 1]
            self.gen_costs_cpy = self.all_gen_costs[:, count - 1]

            # enforce convexity for battery bids
            self.bats_discharge_costs_cpy = np.maximum(self.bats_discharge_costs,
                                                       self.bats_charge_costs)

            # update charging range for each battery
            self.bats_max_charge_cpy = np.maximum(
                    self.bats_max_discharge_range[:, 0],
                    -(self.battery_capacity - self.battery_charge_cpy) / (
                        time_step * self.CHARGE_EFFICIENCY))
            self.bats_max_discharge_cpy = np.minimum(
                    self.bats_max_discharge_range[:, 1],
                    self.battery_charge_cpy * self.DISCHARGE_EFFICIENCY / time_step)

            self.load_demand_cpy = self._generate_load_data(count)
            _, x_bats, price = self.market_op.get_dispatch_no_agent()
            
            # print("agent change: ", x_bats[-1])
            # print("price: ", price)

            # sanity checks
            assert abs(x_bats[-1] - 0) <= 10**-10 and 0 <= price

            # update battery charges
            for i in range(self.num_bats):
                if 0 <= x_bats[i]:
                    self.battery_charge_cpy[i] -= (1. / self.DISCHARGE_EFFICIENCY) * x_bats[i]
                    # ensure non-negative battery charge
                    if self.battery_charge_cpy[i] < 0:
                        self.battery_charge_cpy[i] = 0.0
                else:
                    self.battery_charge_cpy[i] -= self.CHARGE_EFFICIENCY * x_bats[i]

            prices[i] = price

        # find optimal total reward for episode based on prices (price taking assumption)

        x = cp.Variable(self.MAX_STEPS_PER_EPISODE)

        constraints = [
            0 <= self.init_battery_charge[-1] + cp.cumsum(-1*x),
            self.init_battery_charge[-1] + cp.cumsum(-1*x) <= self.battery_capacity[-1],
        ]

        moers = np.zeros((self.MAX_STEPS_PER_EPISODE,))

        for i in range(self.MAX_STEPS_PER_EPISODE):
            moers[i] = self._generate_moer_data(i + 1)

        obj = cp.sum(cp.multiply(prices, x))
        obj += 30.85*cp.sum(cp.multiply(moers, x))
        prob = cp.Problem(objective=cp.Maximize(obj), constraints=constraints)
        assert prob.is_dcp()
        assert prob.is_dpp()
        
        prob.solve()
        return prob.value

    def render(self):
        raise NotImplementedError

    def close(self):
        return


# if __name__ == '__main__':
#     env = BatteryStorageInGridEnv()

#     episodes = 100

#     rewards_lst_1 = []

#     for i in tqdm(range(episodes)):
#         ob = env.reset()
#         done = False
#         rewards = np.zeros(env.MAX_STEPS_PER_EPISODE)

#         while not done:
#             # random action as policy
#             action = env.action_space.sample()
#             state, reward, done, info = env.step(action)
#             rewards[env.count - 1] = reward
#         # print("episode {} mean reward: {}".format(i, np.mean(rewards)))
#         rewards_lst_1.append(np.sum(rewards))

#     # print(rewards_lst_1)
#     # plot episode # versus total episode reward
#     plt.bar(list(range(episodes)), rewards_lst_1, width=0.5)

    # # naming the x axis 
    # plt.xlabel('episode #') 
    # # naming the y axis 
    # plt.ylabel('total reward')

    # plt.show()