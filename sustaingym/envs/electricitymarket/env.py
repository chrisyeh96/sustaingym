"""
The module implements the ElectricityMarketEnv class.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any

import cvxpy as cp
from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec
import numpy as np
import pandas as pd
import pytz

from sustaingym.data.load_moer import MOERLoader
from .market_operator import MarketOperator
from .plot_utils import ElectricityMarketPlot
from ..utils import solve_mosek


AM_LA = pytz.timezone('America/Los_Angeles')
FIVEMIN = timedelta(minutes=5)
DEFAULT_CARBON_PRICE = 30.85


def _datetime_from_date(date: date) -> datetime:
    """Converts date to datetime (at midnight, without timezone).

    For example, date(2010, 1, 1) becomes datetime(2010, 1, 1, 0, 0).
    """
    return datetime(date.year, date.month, date.day)


class ElectricityMarketEnv(Env):
    """
    Actions:

    .. code:: none

        Type: Box(2, number of batteries, settlement interval + 1)
        Discharge and Charge action per battery for each time step in the lookahead.
        Action                              Min                     Max
        ($ / MWh)                        -Max Cost                Max Cost

    Observation:

    .. code:: none
     
        Type: Dict
                                            Min                     Max
        Time (fraction of day)               0                       1
        Energy storage level (MWh)           0                     Max Capacity
        Previous action ($ / MWh)         -Max Cost                Max Cost
        Previous agent dispatch (MWh)      Max Charge              Max Discharge
        Previous load demand (MWh)           0                     Inf
        Load Forecast (MWh)                  0                     Inf
        Previous MOER value                  0                     Inf
        MOER Forecast                        0                     Inf

    The environment operates as s(0), a(1), [s(1), r(1)], a(2), [s(2), r(2)], ...

    1. ``env.reset()`` does the following:

        - The environment sets time t=0. It observes the true demand at t=0 and the
          demand forecast for t=1 to t=self.load_forecast_steps. The market operator
          computes the nodal prices, assuming that the batteries don't participate.
        - Once everything has finished in the t=0 time step, the observation
          ```obs[0]``` is returned to the agent. This includes:

            - "time" starts at 0, normalized as ``t / T``
            - "previous action" is set to zeros (the agent did not participate at t=0)
            - "soc" is the amount of battery energy that the agent starts with
            - "agg demand previous" was the demand at ``t=0``
            - "agg demand forecast" is for ``t=1`` to ``t=self.load_forecast_steps`` (inclusive)
            - "moer previous" was the t=0 MOER
            - "moer forecast" is for ``t=1`` to ``t=self.moer_forecast_steps`` (inclusive)
        - Finally, based on this observation, the agent must decide an action to be
          taken (and used by the market operator) at t=1.
            
    2. At time t=1, the environment observes a new demand and MOER. The env then
        applies the agent's action, resulting in ``obs[1], info[1] = env.step(action)``.
    
        - "time" is set to ``1/self.T``
        - "previous action" is set to the action that the agent just submitted
        - "soc" is the new energy level after applying the agent's action
        - "agg demand previous" is the new demand that was observed at ``t=1``
        - "agg demand forecast" is for ``t=2`` to ``t=(self.load_forecast_steps + 1)``
        - "moer previous" is the MOER observed at t=1
        - "moer forecast" is for ``t=2`` to ``t=(self.moer_forecast_steps + 1)``
    3. This continues until time t=288: ``obs[288], info[288] = env.step(action)``.

        - "time" is set to 1
        - ``terminated`` is set to True 
        
    Args:
        period: tuple of (start_date, end_date), end_date is exclusive
        market_operator: optional MarketOperator, defaults to
            MarketOperator with IEEE24_Network(zone=1) network
        use_intermediate_rewards: bool, whether to calculate intermediate
            rewards
        moer_forecast_steps: # of steps of MOER forecast to include,
            maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
        load_forecast_steps: # of steps of load forecast to include,
            maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
        settlement_interval: # of 5-minute intervals considered in the
            multi-interval economic dispatch problem, defaults to 12 intervals
            (1h lookahead), must be ≤ min(moer_forecast_steps, load_forecast_steps)
        carbon_price: float, price of carbon ($ / mT CO2), 1 mT = 1000 kg

    Attributes:
        # attributes required by gym.Env
        action_space: spaces.Box, structure of actions expected by env
        observation_space: spaces.Dict, structure of observations
        reward_range: tuple[float, float], min and max rewards
        spec: EnvSpec, info used to initialize env from gymnasium.make()
        metadata: dict[str, Any], unused
        np_random: np.random.Generator, random number generator for the env

        # attributes specific to ElectricityMarketEnv
        t: current time step, in {0, 1, ..., 288}
        market_operator: MarketOperator, solves the SCED problem and determines
            nodal prices at each time step
        period: tuple[datetime, datetime]
        history: dict[str, np.ndarray]

            - 'dispatch gens': shape [t+1, N_G]
            - 'dispatch bats': shape [t+1, N_B]
            - 'nodal prices': shape [t+1, N]
            - 'soc': shape [t+1, N_B]
            - 'moer': shape [t+1, 1 + moer_forecast_steps]
            - 'agg demand': shape [t+1]
            - 'agg demand forecast': shape [t+1]
            - 'action': shape [t, 2, N_B, settlement_interval + 1]
        intermediate_rewards: dict[str, np.ndarray | float]
    """

    # Time step duration in hours (corresponds to 5 minutes)
    TIME_STEP_DURATION = 5 / 60
    # Each trajectory is one day (1440 minutes)
    T = 288

    def __init__(self,
                 period: tuple[date, date],
                 market_operator: MarketOperator | None = None,
                 soc_init: Sequence[float] | None = None,
                 use_intermediate_rewards: bool = True,
                 moer_forecast_steps: int = 36,
                 load_forecast_steps: int = 36,
                 settlement_interval: int = 12,
                 carbon_price: float = DEFAULT_CARBON_PRICE):
        self.period = (_datetime_from_date(period[0]), _datetime_from_date(period[1]))

        if market_operator is None:
            from .networks.ieee24 import IEEE24_Network
            network = IEEE24_Network(zone=1)
            market_operator = MarketOperator(
                network, congestion=True, time_step_duration=5/60,
                h=settlement_interval, milp=True)
        self.market_operator = market_operator
        self.time_step_duration = market_operator.time_step_duration
        net = market_operator.network
        assert net.check_valid_period(
            self.period[0], self.period[1], forecast_steps=load_forecast_steps)

        N, N_G, N_B = net.N, net.N_G, net.N_B  # nodes, num gens, num batteries
        T = self.T

        if soc_init is None:
            soc_init = 0.5 * net.soc_max
        self.soc_init = np.array(soc_init, dtype=np.float32)
        assert self.soc_init.shape == (N_B,)
        assert (self.soc_init >= net.soc_min).all()
        assert (self.soc_init <= net.soc_max).all()

        self.use_intermediate_rewards = use_intermediate_rewards

        assert settlement_interval <= min(moer_forecast_steps, load_forecast_steps)
        self.moer_forecast_steps = moer_forecast_steps
        self.load_forecast_steps = load_forecast_steps
        self.settlement_interval = settlement_interval
        self.carbon_price = carbon_price

        # MOERLoader uses timezone info
        self.moer_loader = MOERLoader(
            starttime=self.period[0].replace(tzinfo=AM_LA),
            endtime=self.period[1].replace(tzinfo=AM_LA),
            ba='SGIP_CAISO_SCE', save_dir='sustaingym/data/moer')

        # determine the maximum possible cost of energy ($ / MWh)
        self.max_cost = 1.25 * np.max(net.c_gen[:, 1])

        # action space: charge and discharge bid prices ($ / MWh) for all batteries
        self.action_space = spaces.Box(
            low=-self.max_cost, high=self.max_cost, dtype=np.float32,
            shape=(2, N_B, self.settlement_interval + 1))

        self.observation_space = spaces.Dict({
            'time': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32),
            'soc': spaces.Box(  # units MWh
                low=0, high=net.soc_max, shape=(N_B,), dtype=np.float32),
            'previous action': self.action_space,
            'previous agent dispatch': spaces.Box(  # units MWh
                low=-net.bat_c_max * self.time_step_duration,
                high=net.bat_d_max * self.time_step_duration,
                shape=(N_B,), dtype=np.float32),
            'agg demand previous': spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'agg demand forecast': spaces.Box(
                low=0, high=np.inf, shape=(load_forecast_steps,),
                dtype=np.float32),
            'moer previous': spaces.Box(
                low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'moer forecast': spaces.Box(
                low=0, high=np.inf, shape=(moer_forecast_steps,),
                dtype=np.float32),
            'prices previous': spaces.Box(
                low=-self.max_cost, high=self.max_cost, shape=(N_B,),
                dtype=np.float32)
        })

        # track values over time
        self._hist_dispatch_gens = np.zeros([T+1, N_G], dtype=np.float32)  # in MW
        self._hist_dispatch_bats = np.zeros([T+1, N_B], dtype=np.float32)  # in MWh
        self._hist_nodal_prices = np.zeros([T+1, N], dtype=np.float32)  # in $/MWh
        self._hist_soc = np.zeros([T+1, N_B], dtype=np.float32)  # in MWh
        self._hist_action = np.zeros([T, 2, N_B, settlement_interval + 1], dtype=np.float32)  # in $/MWh

        # Define environment spec
        self.spec = EnvSpec(
            id='sustaingym/ElectricityMarket-v0',
            entry_point='sustaingym.envs:ElectricityMarketEnv',
            nondeterministic=False,
            max_episode_steps=self.T)

    def reset(self, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an instance of an episode.

        Args:
            seed: if given, sets day to ``seed`` days after self.period[0]
            options: includes optional settings like reward type

        Returns:
            tuple containing the initial observation for env's episode
        """
        super().reset(seed=seed)
        self._plot = None

        net = self.market_operator.network
        N_B = net.N_B
        T = self.T

        num_days = (self.period[1] - self.period[0]).days
        if seed is None:
            # randomly pick a day within self.period for the episode
            offset = self.np_random.choice(num_days)  # in {0, ..., num_days-1}
        else:
            offset = seed % num_days
        self.datetime = self.period[0] + timedelta(days=offset)
        self.date = self.datetime.date()

        self.moer_arr = self.moer_loader.retrieve(
            self.datetime.replace(tzinfo=AM_LA)
        ).astype(np.float32)  # shape [289, 1 + self.moer_forecast_steps]

        self.action = np.zeros((2, N_B, self.settlement_interval + 1), dtype=np.float32)
        self.dispatch = np.zeros(N_B, dtype=np.float32)  # (+) to discharge, (-) to charge
        self.t = 0  # counter for the step in current episode
        self.soc = self.soc_init.copy()

        self.agg_demand = self._get_aggregate_demand()
        self.agg_demand_forecast = self._get_aggregate_demand(forecast=True)
        self.moer = self.moer_arr[0:1, 0]
        self.moer_forecast = self.moer_arr[0, 1 : 1 + self.moer_forecast_steps]
        self.time = np.array([self.t / T], dtype=np.float32)

        x_gens, nodal_prices = self.calculate_dispatch_without_agent()
        self.prices = nodal_prices[net.bat_nodes].astype(np.float32)

        # set up observations
        self.obs = {
            'time': self.time,
            'soc': self.soc,
            'previous action': self.action,
            'previous agent dispatch': self.dispatch,
            'agg demand previous': self.agg_demand,
            'agg demand forecast': self.agg_demand_forecast,
            'moer previous': self.moer,
            'moer forecast': self.moer_forecast,
            'prices previous': self.prices
        }

        self.intermediate_rewards = {
            'net': np.zeros(T),
            'energy': np.zeros(T),
            'carbon': np.zeros(T),
            'terminal': None
        }

        # store history
        dts = pd.date_range(self.datetime, periods=self.T + 1, freq=FIVEMIN)
        self._hist_dispatch_gens[0] = x_gens
        self._hist_dispatch_bats[0] = self.dispatch
        self._hist_nodal_prices[0] = nodal_prices
        self._hist_soc[0] = self.soc
        self._hist_agg_demand = net.aggregate_demand(dts).astype(np.float32)  # in MW
        self._hist_agg_demand_forecast = net.aggregate_demand(dts, forecast=True).astype(np.float32)  # in MW
        self._update_history()

        info = {
            'energy reward': None,
            'carbon reward': None,
            'terminal reward': None,
            'history': self.history
        }
        return self.obs, info

    def step(self, action: np.ndarray
             ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Executes a single time step in the environments current trajectory.
        Assumes action is in environment's action space.

        Args:
            action: array of shape [2, N_B, h + 1], float values representing
                charging and discharging bid prices ($/MWh) for each battery and
                time step in the lookahead

        Returns:
            obs: observation
            reward: reward, 0 if ``self.use_intermediate_rewards == False`` and
                ``terminated == False``
            terminated: whether the episode has reached after 288 steps
            truncated: always ``False``, since there is no intermediate stopping condition
            info: additional info
        """
        net = self.market_operator.network
        assert action.shape == (2, net.N_B, self.settlement_interval+1)

        self.t += 1
        t = self.t
        self.datetime += FIVEMIN

        # ensure selling cost (discharging) is at least as large as buying cost (charging)
        self.action[:] = action

        self.agg_demand[:] = self._get_aggregate_demand()
        self.agg_demand_forecast[:] = self._get_aggregate_demand(forecast=True)
        self.moer[:] = self.moer_arr[self.t:self.t + 1, 0]

        x_gens, x_bat_d, x_bat_c, nodal_prices = self.market_operator.get_dispatch(
            demand=self._get_demand(),
            demand_forecast=self._get_demand(forecast=True, steps=self.settlement_interval),
            c_bat=action, soc=self.soc, soc_final=net.soc_max / 2,
            verbose=False)
        self.dispatch = x_bat_d - x_bat_c
        self.prices[:] = nodal_prices[net.bat_nodes]

        # update battery charges
        self.soc += net.eff_c * x_bat_c - (1. / net.eff_d) * x_bat_d
        self.soc[:] = self.soc.clip(0, net.soc_max)

        # get forecasts for next time step
        self.time[:] = t / self.T
        self.moer_forecast[:] = self.moer_arr[t, 1:self.moer_forecast_steps + 1]

        energy_reward = self.prices @ self.dispatch
        carbon_reward = self.carbon_price * self.moer[0] * self.dispatch.sum()
        reward = energy_reward + carbon_reward

        self.intermediate_rewards['energy'][t-1] = energy_reward
        self.intermediate_rewards['carbon'][t-1] = carbon_reward
        self.intermediate_rewards['net'][t-1] = reward

        terminated = (t >= self.T)

        if terminated:
            # terminal cost = cost to charge batteries to soc_init
            terminal_cost = np.sum(np.maximum(0, self.soc_init - self.soc) * self.prices)
            reward -= terminal_cost

            self.intermediate_rewards['terminal'] = terminal_cost
            self.intermediate_rewards['net'][t-1] = reward

            if not self.use_intermediate_rewards:
                reward = np.sum(self.intermediate_rewards['net'])
        else:
            if not self.use_intermediate_rewards:
                reward = 0
            terminal_cost = None

        self._hist_dispatch_gens[t] = x_gens
        self._hist_dispatch_bats[t] = self.dispatch
        self._hist_nodal_prices[t] = nodal_prices
        self._hist_soc[t] = self.soc
        self._hist_action[t-1] = action
        self._update_history()

        info = {
            'energy reward': energy_reward,
            'carbon reward': carbon_reward,
            'terminal cost': terminal_cost,
            'history': self.history
        }

        truncated = False  # always False, no intermediate stopping conditions
        return self.obs, reward, terminated, truncated, info

    def _update_history(self) -> None:
        t = self.t
        self.history = {
            'dispatch gens': self._hist_dispatch_gens[:t+1],
            'dispatch bats': self._hist_dispatch_bats[:t+1],
            'nodal prices': self._hist_nodal_prices[:t+1],
            'soc': self._hist_soc[:t+1],
            'moer': self.moer_arr[:t+1],
            'agg demand': self._hist_agg_demand[:t+1],
            'agg demand forecast': self._hist_agg_demand_forecast[:t+1],
            'action': self._hist_action[:t],
        }

    def _get_demand(self, dt: datetime | None = None, forecast: bool = False,
                    steps: int | None = None) -> np.ndarray:
        """Get demand or forecasted demand (in MW) at each load.

        Args:
            dt: datetime, defaults to current datetime
            forecast: whether to return forecast. If not forecast, returns
                array of shape [N_D] containing actual demand at time dt. If
                forecast, returns array of shape [N_D, steps] containing
                forecasts every 5 minutes starting from (dt + 5 minutes).
            steps: number of forecast steps to return, defaults to
                self.load_forecast_steps

        Returns:
            demand: array of shape [N_D] or [N_D, steps], type float32
        """
        net = self.market_operator.network
        if dt is None:
            dt = self.datetime
        if forecast:
            steps = self.load_forecast_steps if steps is None else steps
            dts = pd.date_range(dt + FIVEMIN, periods=steps, freq=FIVEMIN)
        else:
            assert steps is None
            dts = dt
        return net.demand(dts, forecast=forecast).astype(np.float32)

    def _get_aggregate_demand(self, dt: datetime | None = None,
                              forecast: bool = False) -> np.ndarray:
        """Get aggregate demand or forecasted aggregate demand (in MW).

        Args:
            dt: datetime, defaults to current datetime
            forecast: bool, whether to return forecast

        Returns:
            agg_demand: np.ndarray, shape [1] or [self.load_forecast_steps], type
                float32
        """
        net = self.market_operator.network
        if dt is None:
            dt = self.datetime
        if forecast:
            dts = pd.date_range(dt + FIVEMIN, periods=self.load_forecast_steps, freq=FIVEMIN)
        else:
            dts = [dt]
        return net.aggregate_demand(dts, forecast=forecast).astype(np.float32)

    def calculate_dispatch_without_agent(
            self, t: int | None = None, demand: np.ndarray | None = None,
            demand_forecast: np.ndarray | None = None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates market price and dispatch at given time step, without
        agent participation.

        Does not modify any state variables.

        Args:
            t: time step, set to None to current time step
            demand: array of shape [N_D], demand at each load in MW
            demand_forecast: 

        Returns:
            x_gens: array of shape [N_G, h+1], generator dispatch values
            price: array of shape [N], nodal prices
        """
        if t is None:
            dt = self.datetime
        else:
            assert demand is None
            assert demand_forecast is None
            dt = _datetime_from_date(self.date) + t * FIVEMIN

        if demand is None:
            assert demand_forecast is None
            demand = self._get_demand(dt)
            demand_forecast = self._get_demand(dt, forecast=True, steps=self.settlement_interval)
        else:
            assert demand_forecast is not None

        x_gens, x_bat_d, x_bat_c, prices = self.market_operator.get_dispatch(
            demand=demand, demand_forecast=demand_forecast,
            c_bat=None, soc=None, soc_final=None, verbose=False)

        # sanity checks
        assert np.all(0 <= prices), 'prices should be nonnegative'
        assert np.isclose(x_bat_d, 0).all() and np.isclose(x_bat_c, 0).all()

        return x_gens, prices

    def calculate_prices_without_agent(
            self, start: int = 0, steps: int | None = None) -> np.ndarray:
        """Calculates market prices ($/MWh) at each battery as if the agent
        did not participate.

        Does not modify any state variables.

        When ``start = 0``, the returned array includes the initial price at
        t=0, returned by `reset()`.

        Args:
            start: time step to start at, in {0, ..., T}
            steps: num steps to calculate dispatch for, defaults to (T+1-start),
                which computes the prices until the end of the episode

        Returns:
            prices: array of shape [steps, N_B], type float64
        """
        if steps is None:
            steps = self.T + 1 - start

        net = self.market_operator.network
        prices = np.zeros([steps, net.N_B])

        # get prices from market for all time steps
        for t in range(start, start + steps):
            _, _, _, bus_prices = self.calculate_dispatch_without_agent(t)
            prices[t] = bus_prices[net.bat_nodes]

        return prices

    def calculate_price_taking_optimal(
            self, prices: np.ndarray, soc_init: np.ndarray,
            soc_final: np.ndarray, t: int) -> dict[str, np.ndarray]:
        """Calculates offline optimal episode, under price-taking assumption.

        ``prices`` provides the price at each time step for each battery. In this
        offline optimal price-taking setting, the agent gets to observe all
        future prices and then decide on an optimal charging/discharging strategy.

        Args:
            prices: array of shape [steps, N_B], fixed prices ($/MWh) at each
                time step
            soc_init: np.ndarray, shape [N_B], initial energy level (MWh) of each
                battery, bounded in [0, net.soc_max]
            soc_final: np.ndarray, shape [N_B], minimum final energy level (MWh)
                of each battery
            t: int, time step to start from

        Returns:
            results: dict, keys are ['rewards', 'dispatch', 'energy', 'net_prices'],
                values are arrays of shape [steps, N_B].
                'net_prices' is prices + carbon cost

        Example::

            # Under the price-taking assumption, we assume that the agent starts
            # "taking prices" at time step t=1 (instead of t=0). This is to match
            # the regular environment setting where the agent's first reward comes
            # from its participation in the market at t=1.
            prices = self.calculate_prices_without_agent()  # shape [T+1]
            results = self.calculate_price_taking_optimal(
                prices=prices[1:], soc_init=self.soc_init,
                soc_final=self.soc_init, t=1)
        """
        net = self.market_operator.network
        N_B = net.N_B

        steps = prices.shape[0]
        assert t + steps <= self.T
        
        c = cp.Variable((steps, N_B))  # charging (in MWh)
        d = cp.Variable((steps, N_B))  # discharging (in MWh)

        x = d * net.eff_d - c / net.eff_c  # dispatch (in MWh), shape [steps, N_B]
        delta_energy = cp.cumsum(c) - cp.cumsum(d)  # shape [steps, N_B], in MWh

        constraints = [
            0 <= soc_init + delta_energy,
            soc_init + delta_energy <= net.soc_max,

            # rate constraints
            0 <= c,
            c / net.eff_c <= net.bat_c_max * self.TIME_STEP_DURATION,
            0 <= d,
            d * net.eff_d <= net.bat_d_max * self.TIME_STEP_DURATION
        ]
        if soc_final > 0:
            constraints.append(soc_final <= soc_init + delta_energy[-1])

        moers = self.moer_arr[t : t+steps, 0]  # shape [steps]
        net_prices = prices + self.carbon_price * moers  # shape [steps, N_B]

        obj = cp.sum(cp.multiply(net_prices, x))
        prob = cp.Problem(objective=cp.Maximize(obj), constraints=constraints)
        assert prob.is_dcp() and prob.is_dpp()
        solve_mosek(prob)

        rewards = net_prices * x.value  # per-battery
        soc = soc_init + delta_energy.value
        return dict(rewards=rewards, dispatch=x.value, soc=soc, net_prices=net_prices)

    def render(self):
        if self._plot is None:
            self._plot = ElectricityMarketPlot(
                d=self.date, soc_init=self.soc_init, include_returns=True,
                include_bids=True)

        assert self._plot is not None
        net = self.market_operator.network
        self._plot.update(
            demand=self.history['agg demand'],
            demand_forecast=self.history['agg demand forecast'],
            moer=self.history['moer'][:, 0],
            prices=self.history['nodal prices'][:, net.bat_nodes],
            soc=self.history['soc'],
            rewards=self.intermediate_rewards['net'][:self.t],
            bids=self.history['action'][:, :, :, 0])

    def close(self):
        if hasattr(self, '_fig'):
            self._fig.close()
