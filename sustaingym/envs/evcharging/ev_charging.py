"""
The module implements the EVChargingEnv class.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import acnportal.acnsim as acns
import cvxpy as cp
from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec
import numpy as np

from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator
from sustaingym.envs.evcharging.utils import (
    MINS_IN_DAY, site_str_to_site, round)
from sustaingym.envs.utils import solve_mosek

EV_CHARGING_MODULE = 'sustaingym.envs.evcharging'


class EVChargingEnv(Env):
    """EVCharging class.

    This classes simulates the charging schedule of electric vehicles (or EVs)
    connected to an EV charging network. It is based on ACN-Data and ACN-Sim
    developed at Caltech. Each episode is a 24-hour day of charging, and the
    simulation can be done using real data from ACN-Data or a Gaussian mixture
    model (GMM) fitted on the data (see train_artificial_data_model.py). The
    gym supports the Caltech and JPL sites.

    n = number of stations in the EV charging network
    k = number of steps for the MOER CO2 forecast

    Actions:
        Type: Box(n)
        Action                              Shape   Min     Max
        normalized pilot signal             n       0       1

    Observations:
        Type: Dict(Box(n), Box(n), Box(1), Box(k), Box(1))
                                            Shape   Min     Max
        Estimated departures (timesteps)    n       -288    288
        Demands (kWh)                       n       0       Max Allowed Energy Request
        Previous MOER value                 1       0       1
        Forecasted MOER (kg CO2 / kWh)      k       0       1
        Timestep (fraction of day)          1       0       1

    Attributes:
        # attributes required by gym.Env
        action_space: spaces.Box, structure of actions expected by env
        observation_space: spaces.Dict, structure of observations
        reward_range: tuple[float, float], min and max rewards
        spec: EnvSpec, info used to initialize env from gymnasium.make()
        metadata: dict[str, Any], unused
        np_random: np.random.Generator, random number generator for the env

        # attributes specific to EVChargingEnv
        data_generator: AbstractTraceGenerator, generator for sampling EV
            charging events and MOER forecasting
        max_timestep: int, maximum timestep in a day's simulation
        moer_forecast_steps: int, number of steps of MOER forecast to include
        project_action_in_env: bool, whether gym should project action to obey
            network constraints and not overcharge vehicles
        verbose: int, level of verbosity for print out
            0: nothing
            1: print description of current simulation day
            2: print warnings from network constraint violations and convex
                optimization solver
        cn: acns.ChargingNetwork, EV charging network
        num_stations: int, number of stations in EV charging network
        timestep: int, current timestep in episode, from 0 to 288
    """
    # Max charging rate in A for garage EVSEs
    ACTION_SCALE_FACTOR = 32

    # Reward calculation factors
    VOLTAGE = 208  # in volts (V), default value from ACN-Sim
    MARGINAL_REVENUE_PER_KWH = 0.15  # revenue in $ / kWh
    OPERATING_MARGIN = 0.20  # profit / revenue as a %
    MARGINAL_PROFIT_PER_KWH = MARGINAL_REVENUE_PER_KWH * OPERATING_MARGIN  # $ / kWh
    CO2_COST_PER_METRIC_TON = 30.85  # carbon cost in $ / 1000 kg CO2
    A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWh / A * mins)
    VIOLATION_WEIGHT = 0.001  # cost in $ / kWh of violation

    def __init__(self, data_generator: AbstractTraceGenerator,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 verbose: int = 0):
        """
        Args:
            data_generator: generator for sampling EV charging events and MOER
                forecasts
            moer_forecast_steps: number of steps of MOER forecast to include,
                minimum of 1 and maximum of 36. Each step is 5 mins, for a
                maximum of 3 hrs.
            project_action_in_env: whether gym should project action to obey
                network constraints and not overcharge vehicles
            verbose: level of verbosity for print out
                0: nothing
                1: print description of current simulation day
                2: print warnings from network constraint violations and
                    convex optimization solver
        """
        assert 1 <= moer_forecast_steps <= 36

        # Set arguments
        self.data_generator = data_generator
        self.max_timestep = MINS_IN_DAY // self.data_generator.TIME_STEP_DURATION
        self.moer_forecast_steps = moer_forecast_steps
        self.project_action_in_env = project_action_in_env
        self.verbose = verbose
        if self.verbose < 2:
            warnings.filterwarnings("ignore")

        # Reward factor constants: profit, constraint violation, and carbon costs
        self.A_PERS_TO_KWH = self.A_MINS_TO_KWH * self.data_generator.TIME_STEP_DURATION  # (kWh / A * periods)
        self.PROFIT_FACTOR = self.A_PERS_TO_KWH * self.MARGINAL_PROFIT_PER_KWH  # $ / (A * period)
        self.VIOLATION_FACTOR = self.A_PERS_TO_KWH * self.VIOLATION_WEIGHT  # $ / (A * period)
        self.CARBON_COST_FACTOR = self.A_PERS_TO_KWH * (self.CO2_COST_PER_METRIC_TON / 1000)  # ($ * kV * hr) / (kg CO2 * period)

        # Set up infrastructure info with fake parameters
        self.cn = site_str_to_site(self.data_generator.site)
        self.num_stations = len(self.cn.station_ids)
        self._evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Initialize information-tracking arrays once, always gets zeroed out at each step
        self._est_departures = np.zeros(self.num_stations, dtype=np.float32)
        self._demands = np.zeros(self.num_stations, dtype=np.float32)
        self._prev_moer = np.zeros(1, dtype=np.float32)
        self._forecasted_moer = np.zeros(self.moer_forecast_steps, dtype=np.float32)
        self._timestep_obs = np.zeros(1, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'est_departures':  spaces.Box(-288, 288, shape=(self.num_stations,), dtype=np.float32),
            'demands':         spaces.Box(0, self.data_generator.requested_energy_cap,
                                          shape=(self.num_stations,), dtype=np.float32),
            'prev_moer':       spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1.0, shape=(self.moer_forecast_steps,), dtype=np.float32),
            'timestep':        spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
        })

        self._obs = {
            'est_departures': self._est_departures,
            'demands': self._demands,
            'prev_moer': self._prev_moer,
            'forecasted_moer': self._forecasted_moer,
            'timestep': self._timestep_obs,
        }

        # Track cumulative components of reward signal
        self._reward_breakdown = {
            'profit': 0.0,
            'carbon_cost': 0.0,
            'excess_charge': 0.0,
        }

        # Initialize variables for gym resetting
        self.timestep = 0
        self._simulator: acns.Simulator = None

        # Define action space for the pilot signals
        self.action_space = spaces.Box(low=0, high=1.0,
                                       shape=(self.num_stations,), dtype=np.float32)

        # Define reward range
        self.reward_range = (-np.inf, self.PROFIT_FACTOR * 32 * self.num_stations)

        # Define environment spec
        self.spec = EnvSpec(
            id='sustaingym/EVCharging-v0',
            entry_point='sustaingym.envs:EVChargingEnv',
            nondeterministic=False,
            max_episode_steps=288)

        # Set up action projection
        if self.project_action_in_env:
            self._init_action_projection()

    def _init_action_projection(self) -> None:
        """Initializes optimization problem, parameters, and variables."""
        # Projected action to be sent as actual pilot signal
        self.projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Aggregate magnitude (A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.cn._phase_angles))
        A_tilde = self.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.projected_action) * self.ACTION_SCALE_FACTOR  # convert to A

        # Parameters to be set when stepping through environment
        self.agent_action = cp.Parameter((self.num_stations,), nonneg=True)
        self.demands_cvx = cp.Parameter((self.num_stations,), nonneg=True)

        # Action cannot exceed maximum pilot signal or total demand of vehicle
        max_action = cp.minimum(1.,
                                self.demands_cvx / self.A_PERS_TO_KWH / self.ACTION_SCALE_FACTOR)

        objective = cp.Minimize(cp.norm(self.projected_action - self.agent_action, p=2))
        constraints = [
            self.projected_action <= max_action,
            agg_magnitude <= self.cn.magnitudes,
        ]
        self.prob = cp.Problem(objective, constraints)

        assert self.prob.is_dpp() and self.prob.is_dcp()

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """Projects action to satisfy charging network constraints.

        Args:
            action: shape [num_stations], normalized charging rate for each charging station.

        Returns:
            projected action so that network constraints are obeyed and no more charge
                is provided than is demanded. It is the action in the feasible space
                that minimizes the L2 norm between it and the suggested action.
        """
        self.agent_action.value = action
        self.demands_cvx.value = self._demands
        solve_mosek(self.prob, self.verbose)
        action = self.projected_action.value
        return action

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        return (f'EVChargingGym (action projection = {self.project_action_in_env}, moer forecast steps = {self.moer_forecast_steps}) '
                f'using {self.data_generator.__repr__()}')

    def step(self, action: np.ndarray, return_info: bool = False
             ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Calls the step function of the internal simulator and generates the
        observation, reward, done, info tuple specified by OpenAI gym.

        Args:
            action: action: shape [num_stations], normalized charging rate
                for each charging station between 0 and 1.
            return_info: info is always returned, but if return_info is set
                to False, only 'max_profit' and 'reward_breakdown' are
                returned.

        Returns:
            observation: state
                'est_departures': shape [num_stations], the estimated number of
                    periods until departure. If there is no EVSE at the index,
                    the entry is set to zero.
                'demands': shape [num_stations], amount of charge demanded by
                    each EVSE in kWh.
                'prev_moer': shape [1], emissions rate for the current timestep
                    in kg CO2 per kWh. Between 0 and 1.
                'forecasted_moer': shape [moer_forecast_steps], forecasted
                    emissions rate for next timestep(s) in kg CO2 per kWh.
                    Between 0 and 1.
                'timestep': shape [1], fraction of day between 0 and 1.
            reward: scheduler's performance metric per timestep
            terminated: whether episode is terminated
            truncated: whether episode has reached a time limit. Here, truncated
                is always the same as terminated because the episode is always
                across the entire day.
            info: auxiliary useful information
                'num_evs' (int): number of charging sessions in episode.
                'avg_plugin_time' (float): average plugin time in periods
                    (5 mins) across sessions in episode.
                'max_profit' (float): maximum profit if all EVs were charged
                    maximally while they are connected to the network. This does
                    not take into account network constraints or carbon emissions,
                    and it is a good proxy for info['reward_breakdown']['profit'].
                'reward_breakdown' (dict[str, float]): breakdown of evaluation
                    metrics cumulative over the episode.
                    'profit' ($) : profit over charge delivered to all EVs.
                    'carbon_cost'($): cost of marginal emissions.
                    'excess_charge' ($): cost of network violations.
                'evs' (List[acnm.ev.EV]): list of EVs in the event queue
                'active_evs' (List[acnm.ev.EV]): list of active EVs at current
                    timestep
                'moer': shape [289, 37] emissions rate for entire episode.
                'pilot_signals' (DataFrame): pilot signals received by simulator
        """
        self.timestep += 1

        # Step internal simulator
        schedule = self._to_schedule(action)  # transform action to pilot signals
        done = self._simulator.step(schedule)  # NOTE: call reset() for NoneType AttributeError
        self._simulator._resolve = False  # work-around to keep iterating

        # Retrieve environment information
        observation = self._get_observation()
        reward = self._get_reward(schedule)
        info = self._get_info(return_info)

        # terminated, truncated at end of day
        return observation, reward, done, done, info

    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resets the environment.

        Prepares for the next episode by re-creating the charging network,
        generating new events, creating the simulation and interface,
        and resetting information-tracking variables.

        Args:
            seed: seed for resetting the environment. An episode is entirely
                reproducible no matter the generator used.
            options: resetting options
                'verbose': set verbosity level [0-2]

        Returns:
            obs: state. See step().
            info: other information (when return_info = True). See step().
        """
        super().reset(seed=seed)
        self.data_generator.set_seed(seed)

        if options and 'verbose' in options:
            self.verbose = options['verbose']

        # Initialize network, events, MOER data, simulator, interface, and timestep
        self.cn = site_str_to_site(self.data_generator.site)
        self.events, self.evs, num_plugs = self.data_generator.get_event_queue()
        self.moer = self.data_generator.get_moer()
        self._simulator = acns.Simulator(network=self.cn, scheduler=None,
                                         events=self.events, start=self.data_generator.day,
                                         period=self.data_generator.TIME_STEP_DURATION, verbose=False)
        self.interface = acns.Interface(self._simulator)
        self.timestep = 0

        # Restart information tracking for reward component
        for reward_component in self._reward_breakdown:
            self._reward_breakdown[reward_component] = 0.0

        if self.verbose >= 1:
            print(f'Simulating {num_plugs} events using {self.data_generator}')

        return self._get_observation(), self._get_info()

    def _to_schedule(self, action: np.ndarray) -> dict[str, list[float]]:
        """Returns EVSE pilot signals given a numpy action.

        Actions are expected to be in the set [0-1], which are scaled by 32
        to generate pilot signals in [0-32].

        Currently, the gym supports the Caltech and JPL sites with 2 types of
        EVSEs: AV (AeroVironment) and CC (ClipperCreek), each specific with
        the pilot signal required. One type only allows pilot signals in the
        set {0, 8, 16, 24, 32}. The other allows pilot signals in the set
        {0} U {6, 7, 8, ..., 32}. Pilot signals that do not reach the minimum
        pilot signal threshold are set to zero, and they have to be
        appropriately rounded. The gym supports action projection for obeying
        network constraints.

        Args:
            action: shape [num_stations], normalized charging rate for each
                charging station.

        Returns:
            pilot_signals: dict, str => [int]. Maps station ids to a
                single-element list of pilot signals in Amps
        """
        if self.project_action_in_env:
            action = self._project_action(action)

        action = np.round(action * self.ACTION_SCALE_FACTOR)
        pilot_signals = {}
        for i in range(self.num_stations):
            station_id = self.cn.station_ids[i]
            # hacky way to determine allowable rates - allowed rates required to keep simulation running
            # one type of EVSE accepts values in {0, 8, 16, 24, 32}
            # the other type accepts {0} U {6, 7, 8, ..., 32}
            if self.cn.min_pilot_signals[i] == 6:
                # signals less than min pilot signal are set to zero, rest are in action space
                pilot_signals[station_id] = [action[i] if action[i] >= 6 else 0]
            else:
                # set to {0, 8, 16, 24, 32}
                pilot_signals[station_id] = [round(action[i] / 8) * 8]
        return pilot_signals

    def _get_observation(self) -> dict[str, Any]:
        """Returns observations for the current state of simulation."""
        self._est_departures.fill(0)
        self._demands.fill(0)
        for session_info in self.interface.active_sessions():
            station_idx = self._evse_name_to_idx[session_info.station_id]
            self._est_departures[station_idx] = session_info.estimated_departure - self.timestep
            self._demands[station_idx] = session_info.remaining_demand  # kWh

        self._prev_moer[:] = self.moer[self.timestep, 0]
        self._forecasted_moer[:] = self.moer[self.timestep, 1:self.moer_forecast_steps + 1]  # forecasts start from 2nd column
        self._timestep_obs[0] = self.timestep / self.max_timestep

        return self._obs

    def _get_info(self, all: bool = False) -> dict[str, Any]:
        """Returns info. See step().

        Args:
            all: whether all information should be returned. Otherwise, only
                'max_profit' and 'reward_breakdown' are returned.
        """
        if all:
            return {
                'num_evs': len(self.evs),
                'avg_plugin_time': self._calculate_avg_plugin_time(),
                'max_profit': self._calculate_max_profit(),
                'reward_breakdown': self._reward_breakdown,
                'evs': self.evs,
                'active_evs': self._simulator.get_active_evs(),
                'moer': self.moer,
                'pilot_signals': self._simulator.pilot_signals_as_df(),
            }
        else:
            return {
                'max_profit': self._calculate_max_profit(),
                'reward_breakdown': self._reward_breakdown,
            }

    def _calculate_avg_plugin_time(self) -> float:
        """Calculate average plug-in times for evs in periods."""
        return np.mean(np.array([ev.departure - ev.arrival for ev in self.evs]))

    def _calculate_max_profit(self) -> float:
        """Calculate max profits without regards to network constraints."""
        requested_energy = np.array([ev.requested_energy for ev in self.evs])
        duration_in_periods = np.array([ev.departure - ev.arrival for ev in self.evs])
        max_kwh_in_duration = duration_in_periods * 32 * self.A_PERS_TO_KWH
        max_kwh_to_provide = np.minimum(requested_energy, max_kwh_in_duration)
        max_profit = np.sum(max_kwh_to_provide * self.MARGINAL_PROFIT_PER_KWH)
        return max_profit

    def _get_reward(self, schedule: Mapping[str, Sequence[float]]) -> float:
        """Returns reward for scheduler's performance on timestep.

        The reward is a weighted sum of charging rewards, carbon costs,
        and network constraint violation costs.

        Args:
            schedule: maps EVSE charger ID to a single-element
                list of the pilot signal (in Amps) to that charger.

        Returns:
            total_reward: weighted reward awarded to the current timestep
            info: dictionary containing the individual, unweiK
        """
        schedule = np.array([x[0] for x in schedule.values()])  # convert to numpy
        total_charging_rate = np.sum(self._simulator.charging_rates[:, self.timestep-1:self.timestep])

        # profit calculation (Amp * period) -> (Amp * mins) -> (KWH) -> ($)
        profit = self.PROFIT_FACTOR * total_charging_rate

        # Network constraints - amount of charge over maximum allowed rates ($)
        current_sum = np.abs(self._simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(0, current_sum - self._simulator.network.magnitudes))
        excess_charge = excess_current * self.VIOLATION_FACTOR

        # Carbon cost ($)
        carbon_cost = self.CARBON_COST_FACTOR * total_charging_rate * self.moer[self.timestep, 0]

        total_reward = profit - carbon_cost - excess_charge

        # Update reward information-tracking
        self._reward_breakdown['profit'] += profit
        self._reward_breakdown['carbon_cost'] += carbon_cost
        self._reward_breakdown['excess_charge'] += excess_charge

        return total_reward

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        del self._simulator, self.interface, self.events, self.cn


if __name__ == '__main__':
    from sustaingym.envs.evcharging import RealTraceGenerator
    # import cProfile, pstats

    test_ranges = (
        ('2019-05-01', '2019-08-31'),
        ('2019-09-01', '2019-12-31'),
        ('2020-02-01', '2020-05-31'),
        ('2021-05-01', '2021-08-31'),
    )
    env = EVChargingEnv(RealTraceGenerator('caltech', test_ranges[0]))
    # print(env.)

    # with cProfile.Profile() as profile:
    #     env = EVChargingEnv(RealTraceGenerator('caltech', test_ranges[0]))

    #     for i in range(10):
    #         done = False
    #         obs, episode_info = env.reset(seed=i, return_info=True)
    #         print("reset obs length: ", len(obs))
    #         steps = 0
    #         while not done:
    #             action = np.ones((54,))
    #             obs, reward, terminated, truncated, info = env.step(action, return_info=False)
    #             print("step obs length: ", len(obs))
    #             # assert 1 == 0
    #             done = terminated or truncated
    #             steps += 1
    #         print(f"Iteration: {i + 1}, steps {steps}")

    #     print(steps)
    #     print(info.keys())
    #     print(info['reward_breakdown'])

    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.CUMULATIVE)
    # results.print_stats(20)
