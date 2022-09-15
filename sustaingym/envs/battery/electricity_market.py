"""
The module implements the BatteryStorageInGridEnv class.
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
import pytz

from sustaingym.data.load_moer import MOERLoader
from sustaingym.envs.utils import solve_mosek

BATTERY_STORAGE_MODULE = 'sustaingym.envs.battery'


class MarketOperator:
    """MarketOperator class."""
    def __init__(self, env: ElectricityMarketEnv):
        """
        Args:
            env: instance of ElectricityMarketEnv class
        """
        self.env = env

        # x: energy in MWh, positive means generation, negative (for battery) means charging
        self.x = cp.Variable(env.num_gens + env.num_bats)
        x_gens = self.x[:env.num_gens]
        x_bats = self.x[env.num_gens:]

        # time-dependent parameters
        self.gen_max_production = cp.Parameter(env.num_gens, nonneg=True)  # rate in MW
        self.bats_max_charge = cp.Parameter(env.num_bats, nonpos=True)  # rate in MW
        self.bats_max_discharge = cp.Parameter(env.num_bats, nonneg=True)
        self.bats_charge_costs = cp.Parameter(env.num_bats, nonneg=True)
        self.bats_discharge_costs = cp.Parameter(env.num_bats, nonneg=True)
        self.gen_costs = cp.Parameter(env.num_gens, nonneg=True)
        self.demand = cp.Parameter()  # net demand: usually +, but could be -

        constraints = [
            cp.sum(self.x) == self.demand,  # supply = demand

            0 <= x_gens,
            x_gens <= self.gen_max_production * env.TIME_STEP_DURATION,

            self.bats_max_charge * env.TIME_STEP_DURATION <= x_bats,
            x_bats <= self.bats_max_discharge * env.TIME_STEP_DURATION,
        ]

        obj = self.gen_costs @ x_gens
        obj += cp.sum(cp.maximum(cp.multiply(self.bats_discharge_costs, x_bats), cp.multiply(self.bats_charge_costs, x_bats)))
        self.prob = cp.Problem(objective=cp.Minimize(obj), constraints=constraints)
        assert self.prob.is_dcp() and self.prob.is_dpp()

    def get_dispatch(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Determines dispatch values.

        Returns:
            x_gens: array of shape [num_gens], generator dispatch values
            x_bats: array of shape [num_bats], battery dispatch values
            price: float
        """
        self.gen_max_production.value = self.env.gen_max_production
        self.gen_costs.value = self.env.gen_costs
        self.bats_max_charge.value = self.env.bats_max_charge
        self.bats_max_discharge.value = self.env.bats_max_discharge
        self.bats_discharge_costs.value = self.env.bats_costs[:, 0]
        self.bats_charge_costs.value = self.env.bats_costs[:, 1]
        self.demand.value = self.env.demand[0]
        # if self.env.count == 287:
        #     print("battery charge: ", self.env.battery_charge[-1])
        #     print("gen_max_production: ", self.env.gen_max_production)
        #     print("gen_costs: ", self.env.gen_costs)
        #     print("bats_max_charge: ", self.env.bats_max_charge)
        #     print("bats_max_discharge: ", self.env.bats_max_discharge)
        #     print("bats_charge_costs: ", self.env.bats_costs[:, 0])
        #     print("bats_discharge_costs: ", self.env.bats_costs[:, 1])
        #     print("load: ", self.env.demand)
        solve_mosek(self.prob)
        # print("status: ", self.prob.status)
        # print("x value: ", self.x.value)
        # print("price:", -1*self.prob.constraints[0].dual_value)
        price = -self.prob.constraints[0].dual_value  # negative because of minimizing objective in LP
        x_gens = self.x.value[:self.env.num_gens]
        x_bats = self.x.value[self.env.num_gens:]

        # if self.env.count == 287:
        #     print("battery dispatch: ", x_bats[-1])

        return x_gens, x_bats, price


class ElectricityMarketEnv(Env):
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
    DEFAULT_GEN_MAX_RATES = (36.8, 31.19, 3.8, 9.92, 49.0, 50.0, 50.0, 15.0, 48.5, 56.7)
    # default max discharging rates for batteries (MW), from the perspective
    #   of the market operator
    DEFAULT_BAT_MAX_DISCHARGE = (20.0, 29.7, 7.5, 2.0, 30.0)
    # default max capacity for batteries (MWh)
    DEFAULT_BAT_CAPACITY = (80, 20, 30, 0.95, 120)
    # default range for max charging and discharging rates for batteries (MW)
    #   assuming symmetric range
    DEFAULT_BAT_MAX_RATES = tuple((-val, val) for val in DEFAULT_BAT_MAX_DISCHARGE)
    # price of carbon ($ / mT of CO2), 1 mT = 1000 kg
    CARBON_PRICE = 30.85

    def __init__(self, num_gens: int = 10,
                 gen_max_production: Sequence[float] = DEFAULT_GEN_MAX_RATES,
                 gen_costs: np.ndarray | None = None,
                 num_bats: int = 5,
                 battery_capacity: Sequence[float] = DEFAULT_BAT_CAPACITY,
                 bats_max_rates: Sequence[Sequence[float]] = DEFAULT_BAT_MAX_RATES,
                 bats_costs: np.ndarray | None = None,
                 randomize_costs: Sequence[str] = (),
                 month: str = '2019-05',
                 moer_forecast_steps: int = 36,
                 seed: int | None = None,
                 LOCAL_FILE_PATH: str | None = None):
        """
        Args:
            num_gens: number of generators
            gen_max_production: shape [num_gens], maximum production of each generator (MW)
            gen_costs: shape [num_gens], costs of each generator ($/MWh)
            num_bats: number of batteries (last battery is agent-controlled)
            battery_capacity: shape [num_bats], capacity of each battery (MWh)
            bats_max_rates: shape [num_bats, 2],
                maximum charge (-) and discharge (+) rates of each battery (MW)
            bats_costs: shape [num_bats - 1, 2], charging and discharging costs
                for each battery, excluding the agent-controlled battery ($/MWh)
            randomize_costs: list of str, chosen from ['gens', 'bats'], which
                costs should be randomly scaled
            month: year and month to load moer and net demand data from, format YYYY-MM
            moer_forecast_steps: number of steps of MOER forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            seed: random seed
        """
        if LOCAL_FILE_PATH is not None:
            assert month in ['2019-05', '2020-05', '2021-05']  # update for future dates

        self.num_gens = num_gens
        self.num_bats = num_bats
        self.randomize_costs = randomize_costs
        self.month = month
        self.moer_forecast_steps = moer_forecast_steps
        self.LOCAL_PATH = LOCAL_FILE_PATH

        self.rng = np.random.default_rng(seed)
        rng = self.rng

        assert len(gen_max_production) == self.num_gens
        self.gen_max_production = np.array(gen_max_production, dtype=np.float32)

        if gen_costs is None:
            self.init_gen_costs = rng.uniform(50, 150, size=self.num_gens)
        else:
            assert len(gen_costs) == self.num_gens
            self.init_gen_costs = gen_costs

        assert len(battery_capacity) == self.num_bats
        self.battery_capacity = np.array(battery_capacity, dtype=np.float32)

        self.bats_max_rates = np.array(bats_max_rates, dtype=np.float32)
        assert self.bats_max_rates.shape == (self.num_bats, 2)
        assert (self.bats_max_rates[:, 0] < 0).all()
        assert (self.bats_max_rates[:, 1] > 0).all()

        self.init_bats_costs = np.zeros((self.num_bats, 2))
        if bats_costs is None:
            self.init_bats_costs[:-1, 1] = rng.uniform(50, 100, size=self.num_bats-1)  # discharging
            self.init_bats_costs[:-1, 0] = 0.75 * self.init_bats_costs[:-1, 1]  # charging
        else:
            self.init_bats_costs[:-1] = bats_costs
        assert (self.init_bats_costs >= 0).all()

        # determine the maximum possible cost of energy ($ / MWh)
        max_cost = 1.25 * max(self.init_gen_costs.max(), self.init_bats_costs.max())

        # action space is two values for the charging and discharging costs
        self.action_space = spaces.Box(low=0, high=max_cost, shape=(2,), dtype=np.float32)

        # observation space is current energy level, current time, previous (a, b, x)
        # from dispatch and previous load demand value
        self.observation_space = spaces.Dict({
            'energy':          spaces.Box(low=0, high=self.battery_capacity[-1], shape=(1,), dtype=float),
            'time':            spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'previous action': spaces.Box(low=0, high=np.inf, shape=(2,), dtype=float),
            'previous agent dispatch': spaces.Box(low=self.bats_max_rates[-1, 0] * self.TIME_STEP_DURATION,
                                                  high=self.bats_max_rates[-1, 1] * self.TIME_STEP_DURATION,
                                                  shape=(1,), dtype=float),
            'demand previous': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
            'demand forecast': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float),
            'moer previous':   spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'moer forecast':   spaces.Box(low=0, high=1, shape=(moer_forecast_steps,), dtype=float),
            'price previous':  spaces.Box(low=0, high=max_cost, shape=(1,), dtype=float)
        })
        self.init = False
        self.market_op = MarketOperator(self)
        self.df_demand = self._get_demand_data()
        self.df_demand_forecast = self._get_demand_forecast_data()

        starttime = datetime.strptime(self.month, '%Y-%m').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_loader = MOERLoader(
            starttime=starttime, endtime=starttime,
            ba='SGIP_CAISO_SCE', save_dir='sustaingym/data/moer')

    def _get_demand_data(self) -> pd.DataFrame:
        """Get net demand data.

        TODO: verify data is actually net demand

        Returns:
            DataFrame with net demand, columns are 'HH:MM' at 5-min intervals
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'demand_data', f'CAISO-demand-{self.month}.csv.gz')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            df_demand = pd.read_csv(BytesIO(bytes_data), compression='gzip', index_col=0)
            # TODO: assert shape of DataFrame
            return df_demand / 1800.

    def _get_demand_forecast_data(self) -> pd.DataFrame:
        """Get temporal load forecast data.

        TODO: verify data is actually net demand forecast

        Returns:
            Dataframe with demand forecast, columns are 'HH:MM' at 5-min intervals
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = f'data/demand_forecast_data/CAISO-demand-forecast-{self.month}.csv.gz'
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            # bytes_data = pkgutil.get_data(__name__, csv_path)
            assert bytes_data is not None
            df_demand_forecast = pd.read_csv(BytesIO(bytes_data), compression='gzip', index_col=0)
            # TODO: assert shape of DataFrame
            return df_demand_forecast / 1800.

    def _generate_load_data(self, count: int) -> float:
        """Generate net demand for the time step associated with the given count.

        Args:
            count: integer representing a given time step

        Returns:
            net demand for the given time step (MWh) *currently demand*
        """
        return self.df_demand.iloc[self.idx, count]

    def _generate_load_forecast_data(self, count: int) -> float:
        """Generate hour ahead forecast of the net demand for the time step associated
        with the given count.

        Args:
            count: integer representing a given time step

        Returns:
            net demand for the given time step (MWh) *currently demand*
        """
        if count > self.df_demand_forecast.shape[1]:
            return np.nan
        return self.df_demand_forecast.iloc[self.idx, count]

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
        if 'gens' in self.randomize_costs:
            self.all_gen_costs = self.init_gen_costs[:, None] * rng.uniform(0.8, 1.25, size=(self.num_gens, self.MAX_STEPS_PER_EPISODE))
        else:
            self.all_gen_costs = self.init_gen_costs[:, None] * np.ones((self.num_gens, self.MAX_STEPS_PER_EPISODE))
        if 'bats' in self.randomize_costs:
            self.all_bats_costs = self.init_bats_costs[:, None, :] * rng.uniform(0.8, 1.25, size=(self.num_bats, self.MAX_STEPS_PER_EPISODE, 2))
        else:
            self.all_bats_costs = self.init_bats_costs[:, None, :] * np.ones((self.num_bats, self.MAX_STEPS_PER_EPISODE, 2))

        # enforce convexity of battery bids:
        # discharge costs must be >= charge costs
        self.all_bats_costs[:, :, 1] = np.maximum(
            self.all_bats_costs[:, :, 0], self.all_bats_costs[:, :, 1])

        # randomly pick a day for the episode, among the days with complete demand data
        pos_ids = [
            idx for idx in range(len(self.df_demand))
            if not pd.isnull(self.df_demand.iloc[idx, :-1]).any()
        ]
        self.idx = rng.choice(pos_ids)
        day = self.idx + 1
        date = datetime.strptime(f'{self.month}-{day:02d}', '%Y-%m-%d').replace(tzinfo=pytz.timezone('America/Los_Angeles'))
        self.moer_arr = self.moer_loader.retrieve(date).astype(np.float32)

        self.action = np.zeros(2, dtype=np.float32)
        self.dispatch = np.zeros(1, dtype=np.float32)
        self.count = 0  # counter for the step in current episode
        self.battery_charge = self.battery_capacity / 2.

        self.init = True
        self.demand = np.array([self._generate_load_data(self.count)], dtype=np.float32)
        self.demand_forecast = np.array([self._generate_load_forecast_data(self.count+1)], dtype=np.float32)
        self.moer = self.moer_arr[0:1, 0]
        self.moer_forecast = self.moer_arr[0, 1:self.moer_forecast_steps + 1]
        self.time = np.array([self._get_time()], dtype=np.float32)

        self.price = np.array([self._calculate_dispatch_without_agent(self.count)[2]], dtype=np.float32)

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

        info = {
            'energy reward': None,
            'carbon reward': None,
            'terminal reward': None
        }
        return self.obs if not return_info else (self.obs, info)

    def step(self, action: Sequence[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Executes a single time step in the environments current trajectory.

        Args:
            action: array of shape [2], two float values representing
                charging and discharging bid prices ($/MWh) for this time step

        Returns:
            obs: dict representing the resulting state from that action
            reward: reward from action
            done: whether the episode is done
            info: additional info (currently empty)
        """
        # print("action: ", action)
        # print("action space: ", self.action_space)
        assert self.init
        # assert self.action_space.contains(action)

        self.count += 1

        # ensure selling cost (discharging) is at least as large as buying cost (charging)
        self.action[:] = action
        if action[1] < action[0]:
            # print('Warning: selling cost (discharging) is less than buying cost (charging)')
            self.action[1] = action[0]

        self.gen_costs = self.all_gen_costs[:, self.count]
        self.bats_costs = self.all_bats_costs[:, self.count]
        self.bats_costs[-1] = self.action

        assert (self.bats_costs[:, 1] >= self.bats_costs[:, 0]).all()

        self.demand[:] = self._generate_load_data(self.count)
        self.moer[:] = self.moer_arr[self.count:self.count + 1, 0]

        # rates in MW
        self.bats_max_charge = np.maximum(
            self.bats_max_rates[:, 0],
            -(self.battery_capacity - self.battery_charge) / (self.TIME_STEP_DURATION * self.CHARGE_EFFICIENCY))
        self.bats_max_discharge = np.minimum(
            self.bats_max_rates[:, 1],
            self.battery_charge * self.DISCHARGE_EFFICIENCY / self.TIME_STEP_DURATION)

        _, x_bats, price = self.market_op.get_dispatch()
        x_agent = x_bats[-1]
        self.dispatch[:] = x_agent
        self.price[:] = price

        # update battery charges
        charging = (x_bats < 0)
        self.battery_charge[charging] -= self.CHARGE_EFFICIENCY * x_bats[charging]
        self.battery_charge[~charging] -= (1. / self.DISCHARGE_EFFICIENCY) * x_bats[~charging]
        self.battery_charge[:] = self.battery_charge.clip(0, self.battery_capacity)

        # get forecasts for next time step
        self.time[:] = self._get_time()
        self.demand_forecast[:] = self._generate_load_forecast_data(self.count + 1)
        self.moer_forecast[:] = self.moer_arr[self.count, 1:self.moer_forecast_steps + 1]

        energy_reward = price * x_agent
        carbon_reward = self.CARBON_PRICE * self.moer[0] * x_agent
        reward = energy_reward + carbon_reward
        done = (self.count + 1 >= self.MAX_STEPS_PER_EPISODE)

        if done:
            terminal_cost = self._calculate_terminal_cost(self.battery_charge[-1])
            reward -= terminal_cost
        else:
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
        self.gen_costs = self.all_gen_costs[:, count]
        self.bats_costs = self.all_bats_costs[:, count]

        # update charging range for each battery
        self.bats_max_charge = np.maximum(
            self.bats_max_rates[:, 0],
            -(self.battery_capacity - self.battery_charge) / (self.TIME_STEP_DURATION * self.CHARGE_EFFICIENCY))
        self.bats_max_discharge = np.minimum(
            self.bats_max_rates[:, 1],
            self.battery_charge * self.DISCHARGE_EFFICIENCY / self.TIME_STEP_DURATION)

        # prevent agent-battery from participating in market
        self.bats_max_charge[-1] = 0
        self.bats_max_discharge[-1] = 0

        self.demand[:] = self._generate_load_data(count)
        x_gens, x_bats, price = self.market_op.get_dispatch()

        # sanity checks
        assert np.isclose(x_bats[-1], 0) and 0 <= price
        x_bats[-1] = 0
        return x_gens, x_bats, price

    def _calculate_prices_without_agent(self) -> np.ndarray:
        """Calculates market prices, as if the agent did not participate.

        The only state variable that is modified is self.demand.

        Returns:
            np.ndarray, shape [num_steps], type float64
        """
        battery_charge_save = self.battery_charge.copy()
        prices = np.zeros(self.MAX_STEPS_PER_EPISODE)

        # get prices from market for all time steps
        for count in range(self.MAX_STEPS_PER_EPISODE):
            _, x_bats, price = self._calculate_dispatch_without_agent(count)
            prices[count] = price

            # update battery charges
            charging = (x_bats < 0)
            self.battery_charge[charging] -= self.CHARGE_EFFICIENCY * x_bats[charging]
            self.battery_charge[~charging] -= (1. / self.DISCHARGE_EFFICIENCY) * x_bats[~charging]
            self.battery_charge[:] = self.battery_charge.clip(0, self.battery_capacity)

        self.battery_charge = battery_charge_save
        return prices

    def _calculate_price_taking_optimal(
            self, prices: np.ndarray, init_charge: float,
            final_charge: float) -> dict[str, np.ndarray]:
        """Calculates optimal episode, under price-taking assumption.

        Args:
            prices: array of shape [num_steps], fixed prices at each time step
            init_charge: float in [0, self.battery_capacity[-1]],
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
            init_charge + delta_energy <= self.battery_capacity[-1],

            # rate constraints
            0 <= c,
            c / self.CHARGE_EFFICIENCY <= -self.bats_max_rates[-1, 0] * self.TIME_STEP_DURATION,
            0 <= d,
            d * self.DISCHARGE_EFFICIENCY <= self.bats_max_rates[-1, 1] * self.TIME_STEP_DURATION
            # self.bats_max_rates[-1, 0] * self.TIME_STEP_DURATION <= x,
            # x <= self.bats_max_rates[-1, 1] * self.TIME_STEP_DURATION
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
        prices = self._calculate_prices_without_agent()
        half_charge = self.battery_capacity[-1] / 2.
        future_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=agent_energy_level, final_charge=half_charge)['rewards']
        potential_rewards = self._calculate_price_taking_optimal(
            prices, init_charge=half_charge, final_charge=half_charge)['rewards']

        # added factor to ensure terminal costs motivates charging actions
        penalty = max(0, prices[-1] * (half_charge - agent_energy_level))

        future_return = np.sum(future_rewards)
        potential_return = np.sum(potential_rewards)
        return max(0, potential_return - future_return) + penalty

    def render(self):
        raise NotImplementedError

    def close(self):
        return
