"""This module contains the central class for the EV Charging environment."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import acnportal.acnsim as acns
import cvxpy as cp
from gym import Env, spaces
import numpy as np

from .event_generation import AbstractTraceGenerator
from .utils import MINS_IN_DAY, ActionType, site_str_to_site, round


class EVChargingEnv(Env):
    """Central class for the EV Charging gym.

    This class uses the Simulator class in the acnportal package to
    simulate a day of charging. The simulation can be done using real data
    from Caltech's ACNData or a Gaussian mixture model (GMM) trained on
    that data (see train_artificial_data_model.py). The gym supports the
    Caltech and JPL sites.

    Attributes:
        data_generator: (AbstractTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        site: either 'caltech' or 'jpl' garage to get events from
        period: number of minutes of each time interval in simulation
        requested_energy_cap: largest amount of requested energy allowed (kWh)
        max_timestamp: maximum timestamp in a day's simulation
        action_type: either 'continuous' or 'discrete'
        project_action: flag for whether to project action to the feasible
            action space.
        verbose: level of verbosity for print out.
        cn: charging network in use, either Caltech's or JPL's
        infrastructure_info: info on the charging network's infrastructure
        observation_space: the space of available observations
        action_space: the space of actions for the charging network. Can be
            set to be either continuous or discrete.
    """
    metadata: dict = {"render_modes": []}

    # Action constants
    ACTION_SCALING_FACTOR = 8
    DISCRETE_MULTIPLE = 8
    EPS = 1e-3
    MAX_ACTION = 4

    # Reward calculation factors
    VOLTAGE = 208  # in volts (V), default value from ACN-Sim
    MARGINAL_REVENUE_PER_KWH = 0.15  # revenue in $ / kWh
    OPERATING_MARGIN = 0.20
    MARGINAL_PROFIT_PER_KWH = MARGINAL_REVENUE_PER_KWH * OPERATING_MARGIN
    CO2_COST_PER_METRIC_TON = 30.85  # carbon cost in $ / 1000 kg CO2
    A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWH / A * mins)
    VIOLATION_WEIGHT = 0.001  # cost in $ / kWh of violation

    def __init__(self, data_generator: AbstractTraceGenerator,
                 action_type: ActionType = 'discrete',
                 project_action: bool = False,  # TODO move into RL algorithm
                 moer_forecast_steps: int = 36,
                 verbose: int = 0):
        """
        Args:
            data_generator: a subclass of AbstractTraceGenerator
            action_type: either 'continuous' or 'discrete'. If 'discrete', the
                action space is {0, 1, 2, 3, 4}. If 'continuous', it is [0, 4].
                See to_schedule() for more information.
            project_action: flag for whether to project action to the feasible
                action space. If True, action is projected so that network
                constraints are obeyed and no more charge is provided than is
                demanded. Action projection uses convex optimization to
                minimize the L2 norm between the suggested action and action taken.
                Action projection leads to ~2x slowdown.
            moer_forecast_steps: number of steps of MOER forecast to include,
                maximum of 36. Each step is 5 min, for a maximum of 3 hrs.
            verbose: level of verbosity for print out.
                0: nothing
                1: print description of current simulation day
                2: print current constraint warnings for network constraint
                    violations
        """
        assert 1 <= moer_forecast_steps <= 36

        self.data_generator = data_generator
        self.site = data_generator.site
        self.period = data_generator.period
        self.A_PERS_TO_KWH = self.A_MINS_TO_KWH * self.period  # (kWH / A * periods)
        self.PROFIT_FACTOR = self.A_PERS_TO_KWH * self.MARGINAL_PROFIT_PER_KWH
        self.VIOLATION_FACTOR = self.A_PERS_TO_KWH * self.VIOLATION_WEIGHT
        self.CARBON_COST_FACTOR = self.A_PERS_TO_KWH * (self.CO2_COST_PER_METRIC_TON / 1000)  # ($ * kV * hr) / (kg CO2 * min)

        self.max_timestamp = MINS_IN_DAY // self.period
        self.action_type = action_type
        self.project_action = project_action
        self.moer_forecast_steps = moer_forecast_steps
        self.verbose = verbose
        if verbose < 2:
            warnings.filterwarnings("ignore")

        # Set up infrastructure info with fake parameters
        self.cn = site_str_to_site(self.site)
        self.num_stations = len(self.cn.station_ids)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Define observation space and action space
        self.observation_range = {
            'est_departures':  (0, self.max_timestamp),
            'demands':         (0, data_generator.requested_energy_cap),
            'forecasted_moer': (0, 1.0),
            'timestep':        (0, self.max_timestamp)
        }
        self.observation_space = spaces.Dict({
            'est_departures':  spaces.Box(0, 1.0, shape=(self.num_stations,), dtype=np.float32),
            'demands':         spaces.Box(0, 1.0, shape=(self.num_stations,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1.0, shape=(self.moer_forecast_steps,), dtype=np.float32),
            'timestep':        spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
        })

        # Initialize information-tracking arrays once, always gets zeroed out at each step
        self.est_departures = np.zeros(self.num_stations, dtype=np.float32)
        self.demands = np.zeros(self.num_stations, dtype=np.float32)
        self.forecasted_moer = np.zeros(self.moer_forecast_steps, dtype=np.float32)
        self.timestep_obs = np.zeros(1, dtype=np.float32)
        self.phases = np.zeros(self.num_stations, dtype=np.float32)

        self.obs = {
            'est_departures': self.est_departures,
            'demands': self.demands,
            'forecasted_moer': self.forecasted_moer,
            'timestep': self.timestep_obs,
        }

        # Define action space, which is the charging rate for all EVs
        if action_type == 'discrete':
            self.action_space = spaces.MultiDiscrete([self.MAX_ACTION + 1] * self.num_stations)
        else:
            self.action_space = spaces.Box(
                low=0, high=self.MAX_ACTION, shape=(self.num_stations,),
                dtype=np.float32)

        if project_action:
            self.set_up_action_projection()

    def set_up_action_projection(self) -> None:
        """Creates cvxpy variables and parameters for action projection."""
        self.projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.cn._phase_angles))
        A_tilde = self.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.projected_action) * self.ACTION_SCALING_FACTOR  # convert to A
        magnitude_limit = self.cn.magnitudes

        self.actual_action = cp.Parameter((self.num_stations,), nonneg=True)
        self.demands_cvx = cp.Parameter((self.num_stations,), nonneg=True)

        max_action = cp.minimum(self.MAX_ACTION + self.EPS,
                                self.demands_cvx / self.ACTION_SCALING_FACTOR / self.A_PERS_TO_KWH)

        objective = cp.Minimize(cp.norm(self.projected_action - self.actual_action, p=2))
        constraints = [
            self.projected_action <= max_action,
            agg_magnitude <= magnitude_limit,
        ]
        self.prob = cp.Problem(objective, constraints)

        assert self.prob.is_dpp() and self.prob.is_dcp()

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        site = f'{self.site.capitalize()} site'
        action_type = f'action type {self.action_type}'
        project_action = f'action projection {self.project_action}'
        return f'EVChargingGym for the {site}, {action_type}, {project_action}. '

    def step(self, action: np.ndarray, return_info: bool = False
             ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Steps the environment.

        Calls the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
            action: action: shape [num_stations], charging rate for each charging station.
                If action_type == 'discrete', entries should be in {0, 1, 2, 3, 4}.
                If action_type == 'continuous', entries should be in range [0, 4].
            return_info: info is always returned, but if return_info is set
                to False, less information will be present in the dictionary

        Returns:
            observation: state
                'arrivals': shape [num_stations], arrival timestamp for each EVSE.
                    If the EVSE corresponding to the index is empty, the entry
                    is zero.
                'est_departures': shape [num_stations], estimated departure
                    timestamp for each EVSE. If the EVSE corresponding to the
                    index is empty, the entry is zero.
                'demands': shape [num_stations], amount of charge demanded by
                    each EVSE in Amp * periods.
                'forecasted_moer': shape [moer_forecast_steps], forecasted emissions rate for next
                    timestep in kg CO2 per kWh
                'timestep': shape [1], simulation's current iteration
            reward: float representing scheduler's performance
            done: whether episode is finished
            info: dict with the following key-value pairs TODO may change
                'active_evs': list of acns.EV, all EVs currently charging.
                'active_sessions': list of acns.interface.SessionInfo, active sessions.
                'charging_rates': pd.DataFrame, history of charging rates as
                    provided by the internal simulator. Columns are EVSE id, and
                    the index is the iteration.
                'actual_departures': np.ndarray of shape [num_stations],
                    actual departure timestamp for each EVSE. If the EVSE
                    corresponding to the index is not currently charging an EV,
                    the entry is zero.
                'pilot_signals': pd.DataFrame, entire history of pilot signals
                    throughout simulation. Columns are EVSE id, and the index is
                    the iteration.
                'reward': dict, str => float
                    'revenue': revenue from charge delivered ($)
                    'carbon_cost': marginal CO2 emissions rate ($)
                    'excess_charge': costs for violating network constraints ($)
        """
        # Step internal simulator
        schedule = self.to_schedule(action)  # transform action to pilot signals
        done = self.simulator.step(schedule)
        self.simulator._resolve = False  # work-around to keep iterating

        # Retrieve environment information
        observation, info = self.get_observation(return_info=return_info)
        reward, reward_info = self.get_rewards(schedule)
        self.iteration += 1
        info['reward'] = reward_info
        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None
              ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """Resets the environment.

        Prepares for the next episode by re-creating the charging network,
        generating new events, creating the simulation and interface,
        and resetting the timestamps.

        Args:
            seed: random seed for resetting environment
            return_info: whether information should be returned as well
            options: dictionary containing options for resetting.
                'verbose': set verbosity [0-2]
                'project_action': set action projection boolean

        Returns:
            obs: observations. See step() for more information.
            info: other information. See step().
        """
        self.rng = np.random.default_rng(seed)
        self.data_generator.set_random_seed(seed)

        if options:
            if 'verbose' in options:
                self.verbose = options['verbose']

            if 'project_action' in options:
                self.project_action = options['project_action']
                if self.project_action:
                    self.set_up_action_projection()

        # Initialize charging network
        self.cn = site_str_to_site(self.site)
        # Initialize event queue
        self.events, self.evs, num_plugs = self.data_generator.get_event_queue()
        if self.verbose >= 1:
            print(f'Simulating {num_plugs} events. Using {self.data_generator}')
        # Initialize MOER data
        self.moer = self.data_generator.get_moer()
        # Initialize simulator and interface
        day = self.data_generator.day
        self.simulator = acns.Simulator(network=self.cn, scheduler=None, events=self.events, start=day, period=self.period, verbose=False)
        self.interface = acns.Interface(self.simulator)
        # Initialize time steps
        self.iteration = 0
        # Retrieve environment information
        if return_info:
            return self.get_observation(True)
        else:
            return self.get_observation(False)[0]

    def to_schedule(self, action: np.ndarray) -> dict[str, list[float]]:
        """Returns EVSE pilot signals given a numpy action.

        If the environment uses a discrete action type, actions are expected
        to be in the set {0, 1, 2, 3, 4}, which are scaled by 8 to generate
        pilot signals: {0, 8, 16, 24, 32}. For a continuous action type,
        actions are expected to be in the range [0, 4], which are scaled by
        8 to generate pilot signals. Since some actions may not reach the
        minimum pilot signal threshold, those actions are set to zero.

        Currently, the Caltech and JPL sites have 2 types of EVSEs: AV
        (AeroVironment) and CC (ClipperCreek). They allow a different set of
        currents. One type only allows pilot signals in the set {0, 8, 16,
        24, 32}. The other allows pilot signals in the set {0} U {6, 7, 8,
        ..., 32}. Continuous actions have to be appropriately rounded.

        Args:
            action: shape [num_stations], charging rate for each charging station.
                If action_type == 'discrete', entries should be in {0, 1, 2, 3, 4}.
                If action_type == 'continuous', entries should be in range [0, 4].

        Returns:
            pilot_signals: maps station ids to a single-element list of pilot signals
                in Amps
        """
        # project action if flag is set
        # note that if action is already in the feasible action space,
        # this will just return the action itself
        if self.project_action:
            # TODO(chris): switch to MOSEK
            self.actual_action.value = action
            if self.normalize_observation:
                self.demands_cvx.value = self.demands * self.observation_range['demands'].high
            else:
                self.demands_cvx.value = self.demands
            try:
                self.prob.solve(warm_start=True, solver=cp.MOSEK)
            except cp.SolverError:
                print('Default MOSEK solver failed. Trying ECOS. ')
                self.prob.solve(solver=cp.ECOS)
                if self.prob.status != 'optimal':
                    print(f'prob.status = {self.prob.status}')
                if 'infeasible' in self.prob.status:
                    # your problem should never be infeasible. So now go debug
                    import pdb
                    pdb.set_trace()
            action = self.projected_action.value

        if self.action_type == 'discrete':
            return {e: [self.ACTION_SCALING_FACTOR * a] for a, e in zip(round(action), self.cn.station_ids)}
        else:
            action = np.round(action * self.ACTION_SCALING_FACTOR)  # round to nearest integer rate
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
                    pilot_signals[station_id] = [(np.round(action[i] / self.DISCRETE_MULTIPLE) * self.DISCRETE_MULTIPLE)]
            return pilot_signals

    def get_observation(self, return_info: bool = True
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns observations for the current state of simulation.

        Args:
            return_info: if True, returns info with observation; otherwise,
                info is an empty dictionary.

        Returns:
            obs: observations. See step() for more information.
            info: other information. See step().
        """
        self.est_departures.fill(0)
        self.demands.fill(0)
        for session_info in self.interface.active_sessions():
            station_id = session_info.station_id
            station_idx = self.evse_name_to_idx[station_id]
            self.est_departures[station_idx] = session_info.estimated_departure
            self.demands[station_idx] = self.interface.remaining_amp_periods(session_info) * self.A_PERS_TO_KWH
        self.forecasted_moer[:] = self.moer[self.iteration, 1:self.moer_forecast_steps + 1]  # forecasts start from 2nd column
        self.timestep_obs[0] = self.iteration

        for s in self.obs:
            self.obs[s] /= self.observation_range[s][1]

        if return_info:
            info = {
                'evs': self.evs,
                'active_evs': self.simulator.get_active_evs(),
                'pilot_signals': self.simulator.pilot_signals_as_df(),
            }
            return self.obs, info
        else:
            return self.obs, {}

    def get_rewards(self, schedule: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, float]]:
        """Returns reward for scheduler's performance.

        The reward is a weighted sum of charging rewards, carbon costs,
        and network constraint violation costs.

        Args:
            schedule: maps EVSE charger ID to a single-element
                list of the pilot signal (in Amps) to that charger.

        Returns:
            total_reward: weighted reward awarded to the current timestep
            info: dictionary containing the individual, unweighted
                components making up total reward. See step().
        """
        schedule = np.array([x[0] for x in schedule.values()])  # convert to numpy

        # profit calculation (Amp * period) -> (Amp * mins) -> (KWH) -> ($)
        profit = self.PROFIT_FACTOR * np.sum(self.simulator.charging_rates[:, self.iteration:self.iteration + 1])

        # Network constraints - amount of charge over maximum allowed rates ($)
        current_sum = np.abs(self.simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(0, current_sum - self.simulator.network.magnitudes))
        excess_charge = excess_current * self.VIOLATION_FACTOR

        # Carbon cost ($)
        carbon_cost = self.CARBON_COST_FACTOR * np.sum(schedule) * self.moer[self.iteration, 0]

        total_reward = profit - carbon_cost - excess_charge

        info = {
            'profit': profit,
            'carbon_cost': carbon_cost,
            'excess_charge': excess_charge,
        }
        return total_reward, info

    def render(self) -> None:
        """Render environment."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        del self.simulator, self.interface, self.events, self.cn
