"""
The module implements the EVChargingEnv class.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import warnings

import acnportal.acnsim as acns
import cvxpy as cp
from gym import Env, spaces
import numpy as np

from sustaingym.envs.evcharging.event_generation import AbstractTraceGenerator
from sustaingym.envs.evcharging.utils import MINS_IN_DAY, ActionType, site_str_to_site, round

EV_CHARGING_MODULE = 'sustaingym.envs.evcharging'


class EVChargingEnv(Env):
    """EVCharging class.

    This classes simulates the charging schedule of electric vehicles (or EVs)
    connected to an EV charging network. It is based on ACN-Data and ACN-Sim
    developed at Caltech. Each episode is a 24-hour day of charging, and the
    simulation can be done using real data from ACN-Data or a Gaussian mixture
    model (GMM) fitted on the data (see train_artificial_data_model.py). The
    gym supports the Caltech and JPL sites.

    n = num_stations
    k = moer_forecast_steps

    Actions:
        Type: Box(n)
        Action                              Shape       Min         Max
        normalized pilot signal               n          0           1

    Observations:
        Type: Dict(Box(n), Box(n), Box(k), Box(1))
                                            Shape       Min         Max
        Estimated departures (timesteps)      n        -288         288
        Demands (kWh)                         n          0          Max Allowed Energy Request
        Previous MOER value                   1          0           1
        Forecasted MOER (kg CO2 / kWh)        k          0           1
        Timestep (fraction of day)            1          0           1

    Attributes:
        data_generator: (AbstractTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        site: either 'caltech' or 'jpl' garage to get events from
        max_timestep: maximum timestep in a day's simulation
        action_type: either 'continuous' or 'discrete'
        project_action: flag for whether to project action to the feasible
            action space.
        verbose: level of verbosity for print out.
        cn: charging network in use, either Caltech's or JPL's
        infrastructure_info: info on the charging network's infrastructure
        observation_space: the space of available observations
        action_space: the space of actions for the charging network. Can be
            set to be either continuous or discrete.

            action_type: either 'continuous' or 'discrete'. If 'discrete', the
                action space is {0, 1, 2, 3, 4}. If 'continuous', it is [0, 4].
                See _to_schedule() for more information.
    """
    metadata: dict = {"render_modes": []}

    # Action constants
    ACTION_SCALE_FACTOR = 8 * 4
    EPS = 1e-3
    ROUND_UP_THRESH = 0.7

    # Reward calculation factors
    VOLTAGE = 208  # in volts (V), default value from ACN-Sim
    MARGINAL_REVENUE_PER_KWH = 0.15  # revenue in $ / kWh
    OPERATING_MARGIN = 0.20  # %
    MARGINAL_PROFIT_PER_KWH = MARGINAL_REVENUE_PER_KWH * OPERATING_MARGIN  # $ / kWh
    CO2_COST_PER_METRIC_TON = 30.85  # carbon cost in $ / 1000 kg CO2
    A_MINS_TO_KWH = (1 / 60) * (VOLTAGE / 1000)  # (kWH / A * mins)
    VIOLATION_WEIGHT = 0.001  # cost in $ / kWh of violation

    def __init__(self, data_generator: AbstractTraceGenerator,
                 moer_forecast_steps: int = 36,
                 project_action_in_env: bool = True,
                 verbose: int = 0):
        """
        Args:
            data_generator: a subclass of AbstractTraceGenerator
            moer_forecast_steps: number of steps of MOER forecast to include,
                minimuum of 1 and maximum of 36. Each step is 5 min, for a
                maximum of 3 hrs.
            verbose: level of verbosity for print out.
                0: nothing
                1: print description of current simulation day
                2: print current constraint warnings for network constraint
                    violations
        """
        assert 1 <= moer_forecast_steps <= 36

        # Set arguments
        self.data_generator = data_generator
        self.site = self.data_generator.site
        self.max_timestep = MINS_IN_DAY // self.data_generator.TIME_STEP_DURATION
        self.moer_forecast_steps = moer_forecast_steps
        self.project_action_in_env = project_action_in_env
        self.verbose = verbose
        if self.verbose < 2:
            warnings.filterwarnings("ignore")

        # Reward factor constants: profit, constraint violation, and carbon costs
        self.A_PERS_TO_KWH = self.A_MINS_TO_KWH * self.data_generator.TIME_STEP_DURATION  # (kWH / A * periods)
        self.PROFIT_FACTOR = self.A_PERS_TO_KWH * self.MARGINAL_PROFIT_PER_KWH  # $ / (A * period)
        self.VIOLATION_FACTOR = self.A_PERS_TO_KWH * self.VIOLATION_WEIGHT  # $ / (A * period)
        self.CARBON_COST_FACTOR = self.A_PERS_TO_KWH * (self.CO2_COST_PER_METRIC_TON / 1000)  # ($ * kV * hr) / (kg CO2 * period)

        # Set up infrastructure info with fake parameters
        self.cn = site_str_to_site(self.site)
        self.num_stations = len(self.cn.station_ids)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        self.observation_space = spaces.Dict({
            'est_departures':  spaces.Box(-288, 288, shape=(self.num_stations,), dtype=np.float32),
            'demands':         spaces.Box(0,
                                          self.data_generator.requested_energy_cap,
                                          shape=(self.num_stations,), dtype=np.float32),
            'prev_moer':       spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
            'forecasted_moer': spaces.Box(0, 1.0, shape=(self.moer_forecast_steps,), dtype=np.float32),
            'timestep':        spaces.Box(0, 1.0, shape=(1,), dtype=np.float32),
        })

        # Initialize information-tracking arrays once, always gets zeroed out at each step
        self.est_departures = np.zeros(self.num_stations, dtype=np.float32)
        self.demands = np.zeros(self.num_stations, dtype=np.float32)
        self.prev_moer = np.zeros(1, dtype=np.float32)
        self.forecasted_moer = np.zeros(self.moer_forecast_steps, dtype=np.float32)
        self.timestep_obs = np.zeros(1, dtype=np.float32)
        self.obs = {
            'est_departures': self.est_departures,
            'demands': self.demands,
            'prev_moer': self.prev_moer,
            'forecasted_moer': self.forecasted_moer,
            'timestep': self.timestep_obs,
        }

        # Initialize variables for gym resetting
        self.timestep = 0
        self.simulator: acns.Simulator = None

        # Define action space for the pilot signals
        self.action_space = spaces.Box(low=0, high=1.0,
                                       shape=(self.num_stations,), dtype=np.float32)

        # Set up action projection
        if self.project_action_in_env:
            self._init_action_projection()
    
    def _init_action_projection(self) -> None:
        """Initializes optimization problem, parameters, and variables."""
        self.projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Aggregate magnitude (ACTION_SCALE_FACTOR*A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.cn._phase_angles))
        A_tilde = self.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.projected_action) * self.ACTION_SCALE_FACTOR  # convert to A

        self.agent_action = cp.Parameter((self.num_stations,), nonneg=True)
        self.demands_cvx = cp.Parameter((self.num_stations,), nonneg=True)

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
        self.demands_cvx.value = self.obs['demands']
        try:
            self.prob.solve(warm_start=True, solver=cp.MOSEK)
        except cp.SolverError:
            self.prob.solve(solver=cp.ECOS)
            if self.verbose >= 2:
                print('Default MOSEK solver failed in action projection. Trying ECOS. ')
                if self.prob.status != 'optimal':
                    print(f'prob.status = {self.prob.status}')
            if 'infeasible' in self.prob.status:
                # your problem should never be infeasible. So now go debug
                import pdb
                pdb.set_trace()  # :)
        action = self.projected_action.value
        return action

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        return (f'EVChargingGym (action projection = {self.project_action_in_env}, moer forecast steps = {self.moer_forecast_steps}'
                f'{self.data_generator.__repr__()}')

    def step(self, action: np.ndarray, return_info: bool = False
             ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """Steps the environment.

        Calls the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
            action: action: shape [num_stations], charging rate for each charging station.
            return_info: info is always returned, but if return_info is set
                to False, less information will be present in the dictionary

        Returns:
            observation: state
                'est_departures': shape [num_stations], the estimated time
                    until departure. If there is no EVSE at the index, the
                    entry is set to ``-self.max_timestep``. Normalized between
                    0 and 1.
                'demands': shape [num_stations], amount of charge demanded by
                    each EVSE in kWh, normalized by ``requested_energy_cap``
                    of data generator.
                'forecasted_moer': shape [moer_forecast_steps], forecasted
                    emissions rate for next timestep(s) in kg CO2 per kWh.
                    Between 0 and 1.
                'timestep': shape [1], simulation's current iteration during
                    the day, normalized between 0 and 1.
            reward: scheduler's performance
            done: whether episode is finished
            info: auxiliary information for debugging
                'evs': list of all acns.EV's
                'active_evs': list of all currently charging acns.EV's
                'pilot_signals': pd.DataFrame, entire history of pilot signals
                    throughout simulation. Columns are EVSE id, and the index is
                    the iteration.
                'reward': dict, str => float
                    'revenue': revenue from charge delivered ($)
                    'carbon_cost': marginal CO2 emissions rate ($)
                    'excess_charge': costs for violating network constraints ($)
        """
        self.timestep += 1

        # Step internal simulator
        schedule = self._to_schedule(action)  # transform action to pilot signals
        done = self.simulator.step(schedule)  # should be set by calling reset()
        self.simulator._resolve = False  # work-around to keep iterating

        # Retrieve environment information
        observation = self._get_observation()
        reward, reward_info = self._get_reward(schedule)
        info = self._get_info() if return_info else {}
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
        and resetting the timesteps.

        Args:
            seed: random seed for resetting environment
            return_info: whether information should be returned as well
            options: dictionary containing options for resetting.
                'verbose': set verbosity [0-2]

        Returns:
            obs: state. See step().
            info: auxiliary information. See step().
        """
        self.rng = np.random.default_rng(seed)
        self.data_generator.set_seed(seed)

        if options and 'verbose' in options:
            self.verbose = options['verbose']

        # Initialize network, events, MOER data, simulator, interface, and timestep
        self.cn = site_str_to_site(self.site)
        self.events, self.evs, num_plugs = self.data_generator.get_event_queue()
        self.moer = self.data_generator.get_moer()
        self.simulator = acns.Simulator(network=self.cn, scheduler=None,
                                        events=self.events, start=self.data_generator.day,
                                        period=self.data_generator.TIME_STEP_DURATION, verbose=False)
        self.interface = acns.Interface(self.simulator)
        self.timestep = 0

        if self.verbose >= 1:
            print(f'Simulating {num_plugs} events using {self.data_generator}')

        if return_info:
            return self._get_observation(), self._get_info()
        else:
            return self._get_observation()

    def _to_schedule(self, action: np.ndarray) -> dict[str, list[float]]:
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
            pilot_signals: dict, str => [int]. Maps station ids to a
                single-element list of pilot signals in Amps
        """
        print("Before: ", action)
        if self.project_action_in_env:
            action = self._project_action(action) * self.ACTION_SCALE_FACTOR
        print("After: ", action)
        
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
                pilot_signals[station_id] = [(np.round(action[i] / 8) * 8)]
            print(str(pilot_signals[station_id][0]) + " ", end='')
        print('pilot signals', pilot_signals)
        return pilot_signals

    def _get_observation(self) -> dict[str, Any]:
        """Returns observations for the current state of simulation."""
        self.est_departures.fill(0)
        self.demands.fill(0)
        for session_info in self.interface.active_sessions():
            station_id = session_info.station_id
            station_idx = self.evse_name_to_idx[station_id]
            self.est_departures[station_idx] = session_info.estimated_departure - self.timestep
            self.demands[station_idx] = session_info.remaining_demand  # kWh

        self.prev_moer[:] = self.moer[self.timestep, 0]
        self.forecasted_moer[:] = self.moer[self.timestep, 1:self.moer_forecast_steps + 1]  # forecasts start from 2nd column
        self.timestep_obs[0] = self.timestep / self.max_timestep

        return self.obs
    
    def _get_info(self) -> dict[str, Any]:
        """Returns info."""
        return {
            'evs': self.evs,
            'moer': self.moer,
            'active_evs': self.simulator.get_active_evs(),
            'pilot_signals': self.simulator.pilot_signals_as_df(),
        }

    def _get_reward(self, schedule: Mapping[str, Sequence[float]]) -> tuple[float, dict[str, float]]:
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
        total_charging_rate = np.sum(self.simulator.charging_rates[:, self.timestep-1:self.timestep])

        # profit calculation (Amp * period) -> (Amp * mins) -> (KWH) -> ($)
        profit = self.PROFIT_FACTOR * total_charging_rate

        # Network constraints - amount of charge over maximum allowed rates ($)
        current_sum = np.abs(self.simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(0, current_sum - self.simulator.network.magnitudes))
        excess_charge = excess_current * self.VIOLATION_FACTOR

        # Carbon cost ($)
        carbon_cost = self.CARBON_COST_FACTOR * total_charging_rate * self.moer[self.timestep, 0]

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
