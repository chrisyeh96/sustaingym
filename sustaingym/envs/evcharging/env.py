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

from .event_generation import AbstractTraceGenerator
from .utils import MINS_IN_DAY, site_str_to_site
from sustaingym.envs.utils import solve_mosek


class EVChargingEnv(Env):
    """EVCharging class.

    This classes simulates the charging schedule of electric vehicles (or EVs)
    connected to an EV charging network. It is based on ACN-Data and ACN-Sim
    developed at Caltech. Each episode is a 24-hour day of charging, and the
    simulation can be done using real data from ACN-Data or a Gaussian mixture
    model (GMM) fitted on the data (see train_gmm_model.py). The
    gym supports the Caltech and JPL sites.

    This environment's API is known to be compatible with Gymnasium v0.28, v0.29.

    In what follows:

    - ``n`` = number of stations in the EV charging network
    - ``k`` = number of steps for the MOER CO2 forecast

    Actions:

    .. code:: none

        Type: Box(n)
        Action                              Shape   Min     Max
        normalized pilot signal             n       0       1

    Observations:

    .. code:: none

        Type: Dict(Box(n), Box(n), Box(1), Box(k), Box(1))
                                            Shape   Min     Max
        Estimated departures (timesteps)    n       -288    288
        Demands (kWh)                       n       0       Max Allowed Energy Request
        Previous MOER value                 1       0       1
        Forecasted MOER (kg CO2 / kWh)      k       0       1
        Timestep (fraction of day)          1       0       1

    Args:
        data_generator: generator for sampling EV charging events and MOER
            forecasts
        moer_forecast_steps: number of steps of MOER forecast to include,
            minimum of 1 and maximum of 36. Each step is 5 mins, for a
            maximum of 3 hrs.
        project_action_in_env: whether gym should project action to obey
            network constraints and not overcharge vehicles
        verbose: level of verbosity for print out

            - 0: nothing
            - 1: print description of current simulation day
            - 2: print warnings from network constraint violations and
              convex optimization solver

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

            - 0: nothing
            - 1: print description of current simulation day
            - 2: print warnings from network constraint violations and convex
              optimization solver
        cn: acns.ChargingNetwork, EV charging network
        num_stations: int, number of stations in EV charging network
        timestep: int, current timestep in episode, from 0 to 288
    """
    TIMESTEP_DURATION = 5  # in minutes
    ACTION_SCALE_FACTOR = 32  # Max charging rate in A for garage EVSEs

    # Reward calculation factors
    VOLTAGE = 208  # in volts (V), default value from ACN-Sim
    MARGINAL_REVENUE_PER_KWH = 0.15  # revenue in $ / kWh
    OPERATING_MARGIN = 0.20  # profit / revenue as a %
    MARGINAL_PROFIT_PER_KWH = MARGINAL_REVENUE_PER_KWH * OPERATING_MARGIN  # $ / kWh
    CO2_COST_PER_METRIC_TON = 30.85  # carbon cost in $ / 1000 kg CO2
    A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWh / A * mins)
    VIOLATION_WEIGHT = 0.001  # cost in $ / kWh of violation

    A_PERS_TO_KWH = A_MINS_TO_KWH * TIMESTEP_DURATION  # (kWh / A * periods)
    PROFIT_FACTOR = A_PERS_TO_KWH * MARGINAL_PROFIT_PER_KWH  # $ / (A * period)
    VIOLATION_FACTOR = A_PERS_TO_KWH * VIOLATION_WEIGHT  # $ / (A * period)
    CARBON_COST_FACTOR = A_PERS_TO_KWH * (CO2_COST_PER_METRIC_TON / 1000)  # ($ * kV * hr) / (kg CO2 * period)

    def __init__(self, data_generator: AbstractTraceGenerator,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 verbose: int = 0):
        assert 1 <= moer_forecast_steps <= 36

        # Set arguments
        self.data_generator = data_generator
        self.max_timestep = MINS_IN_DAY // self.TIMESTEP_DURATION
        self.moer_forecast_steps = moer_forecast_steps
        self.project_action_in_env = project_action_in_env
        self.verbose = verbose
        if self.verbose < 2:
            warnings.filterwarnings('ignore')

        # Set up infrastructure info with fake parameters
        self.cn = site_str_to_site(self.data_generator.site)
        self.num_stations = len(self.cn.station_ids)
        self._evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Initialize information-tracking arrays once, always gets zeroed out at each step
        self._est_departures = np.zeros(self.num_stations, dtype=np.float32)
        self._demands = np.zeros(self.num_stations, dtype=np.float32)
        self._prev_moer = np.zeros(1, dtype=np.float32)
        self._forecasted_moer = np.zeros(self.moer_forecast_steps, dtype=np.float32)
        self._timestep_obs = np.zeros(1, dtype=np.float32)  # timestep normalized to [0, 1]

        self.observation_space = spaces.Dict({
            'timestep':        spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'est_departures':  spaces.Box(-288, 288, shape=(self.num_stations,), dtype=np.float32),
            'demands':         spaces.Box(0, self.data_generator.requested_energy_cap,
                                          shape=(self.num_stations,), dtype=np.float32),
            'prev_moer':       spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1, shape=(self.moer_forecast_steps,), dtype=np.float32),
        })

        self._obs = {
            'timestep': self._timestep_obs,
            'est_departures': self._est_departures,
            'demands': self._demands,
            'prev_moer': self._prev_moer,
            'forecasted_moer': self._forecasted_moer,
        }

        # Track cumulative components of reward signal
        self._reward_breakdown = {
            'profit': 0.0,
            'carbon_cost': 0.0,
            'excess_charge': 0.0,
        }

        # Initialize variables for gym resetting
        self.t = 0
        self._simulator: acns.Simulator = None

        # Define action space for the pilot signals
        self.action_space = spaces.Box(
            low=0, high=1.0, shape=(self.num_stations,), dtype=np.float32)

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
        # Projected action to be sent as actual pilot signal, normalized to [0, 1]
        self._projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Parameters to be set when stepping through environment
        self._agent_action = cp.Parameter(self.num_stations, nonneg=True)
        self._demands_cvx = cp.Parameter(self.num_stations, nonneg=True)

        # Action cannot exceed maximum pilot signal or total demand of vehicle
        max_action = cp.minimum(
            1., self._demands_cvx / self.A_PERS_TO_KWH / self.ACTION_SCALE_FACTOR)

        objective = cp.Minimize(cp.norm(self._projected_action - self._agent_action, p=2))
        constraints = [
            self._projected_action <= max_action,
            magnitude_constraint(self._projected_action, self.cn)
        ]
        self.prob = cp.Problem(objective, constraints)

        assert self.prob.is_dpp() and self.prob.is_dcp()

    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """Projects action to satisfy charging network constraints.

        The projection ensures that network constraints are obeyed and no more
        charge is provided than is demanded. The projected action is the action
        in the feasible space that minimizes the L2 norm between it and the
        suggested action.

        Args:
            action: shape [num_stations], normalized charging rate in [0, 1]
                for each charging station.

        Returns:
            projected_action: array of shape [num_stations], still a normalized
                charging rate in [0, 1]
        """
        self._projected_action.value = action  # initialize value for faster convergence
        self._agent_action.value = action
        self._demands_cvx.value = self._demands
        solve_mosek(self.prob, self.verbose)
        action = self._projected_action.value
        return action

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        return (f'EVChargingGym (action projection = {self.project_action_in_env}, '
                f'moer forecast steps = {self.moer_forecast_steps}) '
                f'using {self.data_generator.__repr__()}')

    def step(self, action: np.ndarray
             ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Steps the environment.

        Calls the step function of the internal simulator.

        Args:
            action: shape [num_stations], normalized charging rate
                for each charging station between 0 and 1.

        Returns:
            observation: state

                - 'est_departures': shape [num_stations], the estimated number of
                  periods until departure. If there is no EVSE at the index,
                  the entry is set to zero.
                - 'demands': shape [num_stations], amount of charge demanded by
                  each EVSE in kWh.
                - 'prev_moer': shape [1], emissions rate for the current timestep
                  in kg CO2 per kWh. Between 0 and 1.
                - 'forecasted_moer': shape [moer_forecast_steps], forecasted
                  emissions rate for next timestep(s) in kg CO2 per kWh.
                  Between 0 and 1.
                - 'timestep': shape [1], fraction of day between 0 and 1.

            reward: scheduler's performance metric per timestep
            terminated: whether episode is terminated
            truncated: always ``False``, since there is no intermediate stopping condition
            info: auxiliary useful information

                - 'num_evs': int, number of charging sessions in episode.
                - 'avg_plugin_time': float, average plugin time in periods
                  (5 mins) across sessions in episode.
                - 'max_profit': float, maximum profit if all EVs were charged
                  maximally while they are connected to the network. This does
                  not take into account network constraints or carbon emissions,
                  and it is a good proxy for info['reward_breakdown']['profit'].
                - 'reward_breakdown': dict[str, float], breakdown of evaluation
                  metrics cumulative over the episode.

                  - 'profit' ($) : profit over charge delivered to all EVs.
                  - 'carbon_cost'($): cost of marginal emissions.
                  - 'excess_charge' ($): cost of network violations.
                - 'evs': list[acnm.ev.EV], list of EVs in the event queue
                - 'active_evs': list[acnm.ev.EV], list of active EVs at current
                  timestep
                - 'moer': array, shape [289, 37] emissions rate for entire episode.
                - 'pilot_signals': DataFrame, pilot signals received by simulator
        """
        self.t += 1

        # Step internal simulator
        schedule = self._to_schedule(action)  # transform action to pilot signals
        done = self._simulator.step(schedule)  # NOTE: call reset() for NoneType AttributeError
        self._simulator._resolve = False  # work-around to keep iterating

        # Retrieve environment information
        observation = self._get_observation()
        reward = self._get_reward(schedule)
        info = self._get_info()

        return observation, reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Resets the environment.

        Prepares for the next episode by re-creating the charging network,
        generating new events, creating the simulation and interface,
        and resetting information-tracking variables.

        Args:
            seed: seed for resetting the environment. An episode is entirely
                reproducible no matter the generator used.
            options: resetting options

                - 'verbose': set verbosity level [0-2]

        Returns:
            observation: state dict, see `step()`
            info: info dict, see `step()`
        """
        super().reset(seed=seed)
        self.data_generator.set_seed(seed)

        if options is not None and 'verbose' in options:
            self.verbose = options['verbose']

        # Initialize network, events, MOER data, simulator, interface, and timestep
        self.cn = site_str_to_site(self.data_generator.site)
        events, self._evs, num_plugs = self.data_generator.get_event_queue()
        self._max_profit = self._calculate_max_profit()
        self.moer = self.data_generator.get_moer()
        self._simulator = acns.Simulator(
            network=self.cn, scheduler=None, events=events,
            start=self.data_generator.day,
            period=self.data_generator.TIME_STEP_DURATION, verbose=False)
        self._interface = acns.Interface(self._simulator)
        self.t = 0

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
            pilot_signals: mapping of station ids to a single-element list of
                pilot signals in Amps
        """
        if self.project_action_in_env:
            action = self._project_action(action)

        action *= self.ACTION_SCALE_FACTOR  # convert to (A), in [0, 32]
        pilot_signals = {}
        for i in range(self.num_stations):
            station_id = self.cn.station_ids[i]
            # hacky way to determine allowable rates - allowed rates required to keep simulation running
            # one type of EVSE accepts values in {0, 8, 16, 24, 32}
            # the other type accepts {0} U {6, 7, 8, ..., 32}
            if self.cn.min_pilot_signals[i] == 6:
                # signals less than min pilot signal are set to zero, rest are in action space
                pilot_signals[station_id] = [np.round(action[i]) if action[i] >= 6 else 0]
            else:
                # set to {0, 8, 16, 24, 32}
                pilot_signals[station_id] = [np.round(action[i] / 8) * 8]  # TODO: smarter rounding?
        return pilot_signals

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Returns observations for the current state of simulation."""
        self._est_departures.fill(0)
        self._demands.fill(0)
        for session_info in self._interface.active_sessions():
            station_idx = self._evse_name_to_idx[session_info.station_id]
            self._est_departures[station_idx] = session_info.estimated_departure - self.t
            self._demands[station_idx] = session_info.remaining_demand  # kWh

        self._prev_moer[0] = self.moer[self.t, 0]
        self._forecasted_moer[:] = self.moer[self.t, 1:self.moer_forecast_steps + 1]  # forecasts start from 2nd column
        self._timestep_obs[0] = self.t / self.max_timestep

        return self._obs

    def _get_info(self, all: bool = False) -> dict[str, Any]:
        """Returns info. See `step()`.

        Args:
            all: whether all information should be returned. Otherwise, only
                'max_profit' and 'reward_breakdown' are returned.
        """
        info = {
            'max_profit': self._max_profit,
            'reward_breakdown': self._reward_breakdown
        }
        if all:
            info.update({
                'num_evs': len(self._evs),
                'avg_plugin_time': self._calculate_avg_plugin_time(),
                'evs': self._evs,
                'active_evs': self._simulator.get_active_evs(),
                'moer': self.moer,
                'pilot_signals': self._simulator.pilot_signals_as_df()
            })
        return info

    def _calculate_avg_plugin_time(self) -> float:
        """Calculate average plug-in times for evs in periods."""
        return np.mean([ev.departure - ev.arrival for ev in self._evs])

    def _calculate_max_profit(self) -> float:
        """Calculate max profits without regards to network constraints."""
        requested_energy = np.array([ev.requested_energy for ev in self._evs])
        duration_in_periods = np.array([ev.departure - ev.arrival for ev in self._evs])
        max_kwh_in_duration = duration_in_periods * self.ACTION_SCALE_FACTOR * self.A_PERS_TO_KWH
        max_kwh_to_provide = np.minimum(requested_energy, max_kwh_in_duration)
        max_profit = np.sum(max_kwh_to_provide * self.MARGINAL_PROFIT_PER_KWH)
        return max_profit

    def _get_reward(self, schedule: Mapping[str, Sequence[float]]) -> float:
        """Returns total reward for scheduler performance on current timestep.

        The reward is a weighted sum of charging rewards, carbon costs,
        and network constraint violation costs.

        Args:
            schedule: maps EVSE charger ID to a single-element
                list of the pilot signal (in Amps) to that charger.

        Returns:
            total_reward: weighted reward awarded to the current timestep
        """
        # profit calculation (Amp * period) -> ($)
        total_charging_rate = np.sum(self._simulator.charging_rates[:, self.t-1])  # in (A)
        profit = self.PROFIT_FACTOR * total_charging_rate

        # Network constraints - amount of charge over maximum allowed rates ($)
        schedule = np.array([x[0] for x in schedule.values()])  # convert to numpy
        current_sum = np.abs(self._simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(0, current_sum - self._simulator.network.magnitudes))
        excess_charge = excess_current * self.VIOLATION_FACTOR

        # Carbon cost (Amp * period * kg CO2 / kWh) -> ($)
        carbon_cost = self.CARBON_COST_FACTOR * total_charging_rate * self.moer[self.t, 0]

        total_reward = profit - carbon_cost - excess_charge

        # Update reward information-tracking
        self._reward_breakdown['profit'] += profit
        self._reward_breakdown['carbon_cost'] += carbon_cost
        self._reward_breakdown['excess_charge'] += excess_charge

        return total_reward

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        del self._simulator, self.cn


def magnitude_constraint(action: cp.Variable, cn: acns.ChargingNetwork
                         ) -> cp.Constraint:
    """Creates constraint requiring that aggregate magnitude (A) must be less
    than observation magnitude (A).

    Args:
        action: shape [num_stations] or [num_stations, T], charging rates
            normalized to [0, 1]

    Returns:
        constr: constraint on aggregate magnitude
    """
    phase_factor = np.exp(1j * np.deg2rad(cn._phase_angles))  # shape [num_stations]
    A_tilde = cn.constraint_matrix * phase_factor[None, :]  # shape [num_constraints, num_stations]

    # convert to A
    agg_magnitude = cp.abs(A_tilde @ action) * EVChargingEnv.ACTION_SCALE_FACTOR

    if len(action.shape) == 1:
        # agg_magnitude has shape [num_constraints]
        return agg_magnitude <= cn.magnitudes
    elif len(action.shape) == 2:
        # agg_magnitude has shape [num_constraints, T]
        return agg_magnitude <= cn.magnitudes[:, None]
    else:
        raise ValueError(
            'Action should have shape [num_stations] or [num_stations, T], '
            f'but received shape {action.shape} instead.')
