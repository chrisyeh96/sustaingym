"""This module contains the central class for the EV Charging environment."""
from __future__ import annotations

from typing import Any
import warnings

import acnportal.acnsim as acns
import cvxpy as cp
import gym
from gym import spaces
import numpy as np

from .event_generation import AbstractTraceGenerator
from .utils import MINS_IN_DAY, ActionType, site_str_to_site


ACTION_SCALING_FACTOR = 8
DISCRETE_MULTIPLE = 8
EPS = 1e-3
MAX_ACTION = 4
MAX_PILOT_SIGNAL = 32

VOLTAGE = 208
MARGINAL_REVENUE_PER_KWH = 0.10  # revenue in $ / kWh
CO2_COST_PER_METRIC_TON = 30.85
A_MINS_TO_KWH = (1 / 60) * VOLTAGE * (1 / 1000)
VIOLATION_WEIGHT = 0.005  # cost in $ / kWh of violation
REVENUE_FACTOR = A_MINS_TO_KWH * MARGINAL_REVENUE_PER_KWH
VIOLATION_FACTOR = A_MINS_TO_KWH * VIOLATION_WEIGHT
CARBON_COST_FACTOR = A_MINS_TO_KWH * (1 / 1000) * CO2_COST_PER_METRIC_TON


class EVChargingEnv(gym.Env):
    """
    Central class for the EV Charging gym.

    This class uses the Simulator class in the acnportal package to
    simulate a day of charging. The simulation can be done using real data
    from Caltech's ACNData or a Gaussian mixture model (GMM) trained on
    that data (see train_artificial_data_model.py). The gym supports the
    Caltech and JPL sites.

    Attributes:
        data_generator: (AbstractTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        site: either 'caltech' or 'jpl' garage to get events from
        recompute_freq: number of periods for recurring recompute
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

    def __init__(self, data_generator: AbstractTraceGenerator,
                 action_type: ActionType = 'discrete',
                 project_action: bool = False,
                 verbose: int = 0):
        """
        Args:
            data_generator: a subclass of AbstractTraceGenerator
            action_type: either 'continuous' or 'discrete'. If 'discrete', the
                action space is {0, 1, 2, 3, 4}. If 'continuous', it is [0, 4].
                See to_schedule() for more information.
            project_action: flag for whether to project action to the feasible
                action space. If True, network constraints are guaranteed not
                to be violated. Action projection minimizes the L2 norm
                between the suggested action and action taken using convex
                optimization. Note that this is slow and not recommended
                during training.
            verbose: level of verbosity for print out.
                0: nothing
                1: print high-level description of current simulation day
                2: print out current constraint warnings
        """
        self.data_generator = data_generator
        self.site = data_generator.site
        self.recompute_freq = data_generator.recompute_freq
        self.period = data_generator.period
        self.max_timestamp = MINS_IN_DAY // self.period
        self.action_type = action_type
        self.project_action = project_action
        self.verbose = verbose
        if verbose < 2:
            warnings.filterwarnings("ignore")

        # Set up infrastructure info with fake parameters
        self.cn = site_str_to_site(self.site)
        self.num_stations = len(self.cn.station_ids)

        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Define observation space and action space

        # Observations are dictionaries describing the current demand for charge
        self.observation_space =  spaces.Dict({
            'arrivals':        spaces.Box(low=0, high=self.max_timestamp, shape=(self.num_stations,), dtype=np.int32),
            'est_departures':  spaces.Box(low=0, high=self.max_timestamp, shape=(self.num_stations,), dtype=np.int32),
            'demands':         spaces.Box(low=0, high=data_generator.requested_energy_cap, shape=(self.num_stations,), dtype=np.float32),
            'forecasted_moer': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            'timestep':        spaces.Box(low=0, high=self.max_timestamp, shape=(1,), dtype=np.int32),
        })

        # Initialize information-tracking arrays once, always gets zeroed out at each step
        self.arrivals = np.zeros(self.num_stations, dtype=np.int32)
        self.est_departures = np.zeros(self.num_stations, dtype=np.int32)
        self.demands = np.zeros(self.num_stations, dtype=np.float32)
        self.forecasted_moer = np.zeros(1, dtype=np.float32)
        self.timestep_obs = np.zeros(1, dtype=np.int32)
        self.phases = np.zeros(self.num_stations, dtype=np.float32)
        self.actual_departures = np.zeros(self.num_stations, dtype=np.int32)

        self.obs = {
            'arrivals': self.arrivals,
            'est_departures': self.est_departures,
            'demands': self.demands,
            'forecasted_moer': self.forecasted_moer,
            'timestep': self.timestep_obs,
        }

        # Define action space, which is the charging rate for all EVs
        if action_type == 'discrete':
            self.action_space = spaces.MultiDiscrete(
                [MAX_ACTION + 1 for _ in range(len(self.cn.station_ids))]
            )
        else:
            self.action_space = spaces.Box(
                low=0, high=MAX_ACTION, shape=(len(self.cn.station_ids),), dtype=np.float32
            )

        if project_action:
            self.set_up_action_projection()

    def set_up_action_projection(self):
        """Creates cvxpy variables and parameters for action projection."""
        self.projected_action = cp.Variable(self.num_stations, nonneg=True)

        # Aggregate magnitude (ACTION_SCALING_FACTOR*A) must be less than observation magnitude (A)
        phase_factor = np.exp(1j * np.deg2rad(self.cn._phase_angles))
        A_tilde = self.cn.constraint_matrix * phase_factor[None, :]
        agg_magnitude = cp.abs(A_tilde @ self.projected_action) * ACTION_SCALING_FACTOR  # convert to A
        magnitude_limit = self.cn.magnitudes

        self.actual_action = cp.Parameter(self.num_stations)

        objective = cp.Minimize(cp.norm(self.projected_action - self.actual_action, p=2))
        constraints = [self.projected_action <= MAX_ACTION + EPS, agg_magnitude <= magnitude_limit]
        self.prob = cp.Problem(objective, constraints)

        assert self.prob.is_dpp() and self.prob.is_dcp()

    def __repr__(self) -> str:
        """Returns the string representation of charging gym."""
        site = f'{self.site.capitalize()} site'
        action_type = f'action type {self.action_type}'
        project_action = f'action projection {self.project_action}'
        return f'EVChargingGym for the {site}, {action_type}, {project_action}. '

    def step(self, action: np.ndarray, return_info: bool = False) -> tuple[dict[str, Any], float, bool, dict[Any, Any]]:
        """Steps the environment.

        Calls the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
            action: array of shape (number of stations,) with entries in the
                set {0, 1, 2, 3, 4} if action_type == 'discrete'; otherwise,
                entries should fall in the range [0, 4]. See to_schedule()
                for more information.
            return_info: info is always returned, but if return_info is set
                to False, less information will be present in the dictionary

        Returns:
            observation: a dictionary with the following key-values
                'arrivals': arrival timestamp for each EVSE as a numpy array.
                    If the EVSE corresponding to the index is not currently
                    charging an EV, the entry is zero.
                'est_departures': estimated departure timestamp for each EVSE
                    as a numpy array. If the EVSE corresponding to the index
                    is not currently charging an EV, the entry is zero.
                'demands': amount of charge demanded in Amp * periods.
                'forecasted_moer': next timestep's forecasted emissions rate in
                    kg * CO2 per kWh
                'timestep': simulation's current iteration.
            reward: float representing scheduler's performance.
            done: bool indicating whether episode is finished
            info: dict with the following key-values
                'active_evs': List of all EVs currenting charging.
                'active_sessions': List of active sessions.
                'charging_rates': pd.DataFrame. History of charging rates as
                    provided by the internal simulator.
                'actual_departures': actual departure timestamp for each EVSE
                    as a numpy array. If the EVSE corresponding to the index
                    is not currently charging an EV, the entry is zero.
                'pilot_signals': entire history of pilot signals throughout
                    simulation.
                'reward': dictionary with the following key-values
                    'charging_cost': cost of charge in current timestep in
                        kA * hrs
                    'charging_reward': reward for charge delivered to vehicles
                        in previous timestep in kA * hrs
                    'constraint_punishment': punishment for charge delivered
                        over constraint limits in kA * hrs
                    'remaining_charge_punishment': punishment for charge left
                        to deliver to vehicles in kA * hrs
        """
        self.prev_timestamp = self.simulator._iteration
        # Step internal simulator
        schedule = self.to_schedule(action)  # transform action to pilot signals
        done = self.simulator.step(schedule)
        self.simulator._resolve = False  # work-around to keep iterating
        self.timestamp = self.simulator._iteration

        # Retrieve environment information
        observation, info = self.get_observation(return_info=return_info)
        reward, reward_info = self.get_rewards(schedule, done)

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
                'verbose': set verbose factor
                'project_action': set action projection flag

        Returns:
            obs: observations on arrivals, estimated departures, demands,
                and timesteps.
            info: observations on active EVs, active sessions, actual
                departures (which an online algorithm should not know),
                history of charging rates, and pilot signals.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.data_generator.set_random_seed(seed)

        if options and 'verbose' in options:
            if 'verbose' in options:
                self.verbose = options['verbose']

            if 'project_action' in options:
                self.project_action = options['project_action']
                if self.project_action:
                    self.set_up_action_projection()

        # Initialize charging network
        self.cn = site_str_to_site(self.site)
        # Initialize event queue - always generates at least one RecomputeEvent
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
        self.prev_timestamp, self.timestamp = self.simulator._iteration, self.simulator._iteration

        # Retrieve environment information
        if return_info:
            return self.get_observation(True)
        else:
            return self.get_observation(False)[0]

    def to_schedule(self, action: np.ndarray) -> dict[str, list[float]]:
        """
        Returns pilot signals for the EVSEs given a numpy action.

        If the environment uses a discrete action type, actions are expected
        to be in the set {0, 1, 2, 3, 4}, which are used to generate pilot
        signals that are scaled up by a factor of 8: {0, 8, 16, 24, 32}.
        For a continuous action type, actions are expected to be in the range
        [0, 4] and generate pilot signals that are also scaled up by 8. Since
        some actions may not reach the minimum pilot signal threshold, those
        actions are set to zero.

        Currently, the Caltech and JPL sites have 2 types of EVSEs: AV
        (AeroVironment) and CC (ClipperCreek). They allow a different set of
        currents. One type only allows pilot signals in the set {0, 8, 16,
        24, 32}. The other allows pilot signals in the set {0} U {6, 7, 8,
        ..., 32}. Continuous actions have to be appropriately rounded.

        Args:
            action: charging rate to take at each charging station.
                If action_type is 'discrete', expects actions in {0, 1, 2, 3, 4}.
                If action_type is 'continuous', expects actions in [0, 4].

        Returns:
            pilot_signals: a dictionary mapping station ids to pilot signals.
        """
        # project action if flag is set
        # note that if action is already in the feasible action space,
        # this will just return the action itself
        if self.project_action:
            # TODO(chris): switch to MOSEK
            self.actual_action.value = action
            self.prob.solve(solver='ECOS', warm_start=True)
            action = self.projected_action.value

        if self.action_type == 'discrete':
            return {e: [ACTION_SCALING_FACTOR * a] for a, e in zip(np.round(action), self.cn.station_ids)}
        else:
            action = np.round(action * ACTION_SCALING_FACTOR)  # round to nearest integer rate
            pilot_signals = {}
            for i in range(len(self.cn.station_ids)):
                station_id = self.cn.station_ids[i]
                # hacky way to determine allowable rates - allowed rates required to keep simulation running
                # one type of EVSE accepts values in {0, 8, 16, 24, 32}
                # the other type accepts {0} U {6, 7, 8, ..., 32}
                if self.cn.min_pilot_signals[i] == 6:
                    # signals less than min pilot signal are set to zero, rest are in action space
                    pilot_signals[station_id] = [action[i] if action[i] >= 6 else 0]
                else:
                    # set to {0, 8, 16, 24, 32}
                    pilot_signals[station_id] = [(np.round(action[i] / DISCRETE_MULTIPLE) * DISCRETE_MULTIPLE)]
            return pilot_signals

    def get_observation(self, return_info: bool = True
                        ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns observations for the current state of simulation.

        Args:
            return_info: if True, returns info with observation; otherwise,
                info is an empty dictionary.

        Returns:
            obs: observations on arrivals, estimated departures, demands,
                and timesteps.
            info: observations on active EVs, active sessions, actual
                departures (which an online algorithm should not know),
                history of charging rates, and pilot signals.
        """
        self.arrivals.fill(0)
        self.est_departures.fill(0)
        self.demands.fill(0)
        for session_info in self.interface.active_sessions():
            station_id = session_info.station_id
            station_idx = self.evse_name_to_idx[station_id]
            self.arrivals[station_idx] = session_info.arrival
            self.est_departures[station_idx] = session_info.estimated_departure
            self.demands[station_idx] = self.interface.remaining_amp_periods(session_info)
        self.forecasted_moer[0] = self.moer[len(self.moer)-1-self.timestamp, 1]  # array goes back in time, choose 2nd col
        self.timestep_obs[0] = self.timestamp

        if return_info:
            # Fill other information
            self.actual_departures.fill(0)
            for session_info in self.interface.active_sessions():
                station_id = session_info.station_id
                station_idx = self.evse_name_to_idx[station_id]
                self.actual_departures[station_idx] = session_info.departure

            info = {
                'active_evs': self.simulator.get_active_evs(),
                'active_sessions': self.interface.active_sessions(),
                'charging_rates': self.simulator.charging_rates_as_df(),
                'actual_departures': self.actual_departures,
                'pilot_signals': self.simulator.pilot_signals_as_df(),
            }
            return self.obs, info
        else:
            return self.obs, {}

    def get_rewards(self, schedule: dict, done: bool
                    ) -> tuple[float, dict[str, Any]]:
        """Returns reward for scheduler's performance.

        Reward is a weighted sum of charging rewards from the previous
        timestep, costs of delivering charge on the current timestep, costs
        of constraint violations, and punishment for not fulfilling a
        customer's request by the time they leave.

        Args:
            schedule: dictionary mapping EVSE charger to a single-element list of
                the pilot signal to that charger.
            done: whether simulation is done and customer satisfaction reward
                should be added in.

        Returns:
            total_reward: weighted reward awarded to the current timestep
            info: dictionary containing the individual, unweighted
                components that make up the total reward
        """
        # schedule for the interval between self.prev_timestamp and self.timestamp
        schedule = np.array(list(map(lambda x: x[0], schedule.values())))  # convert to numpy

        interval_mins = (self.timestamp - self.prev_timestamp) * self.period

        # revenue calculation (Amp * period) -> (Amp * mins) -> (KWH) -> ($)
        revenue = np.sum(self.simulator.charging_rates[:, self.prev_timestamp: self.timestamp])
        revenue *= self.period * A_MINS_TO_KWH * MARGINAL_REVENUE_PER_KWH

        # Network constraints - amount of charge over maximum allowed rates ($)
        current_sum = np.abs(self.simulator.network.constraint_current(schedule))
        excess_current = np.sum(np.maximum(current_sum - self.simulator.network.magnitudes, 0))
        excess_charge = excess_current * interval_mins * VIOLATION_FACTOR

        # Carbon cost ($)
        carbon_cost = np.sum(schedule) * interval_mins * self.moer[len(self.moer)-1-self.prev_timestamp, 0]
        carbon_cost *= CARBON_COST_FACTOR

        total_reward = revenue - carbon_cost - excess_charge

        info = {
            'revenue': revenue,
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


if __name__ == "__main__":

    from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator
    from sustaingym.algorithms.evcharging.base_algorithm import PPOAlgorithm, SelectiveChargingAlgorithm, GreedyAlgorithm

    from .event_generation import GMMsTraceGenerator, RealTraceGenerator
    from acnportal.acnsim.events import PluginEvent
    import time
    from collections import defaultdict

    from ...algorithms.evcharging.base_algorithm import GreedyAlgorithm

    np.random.seed(42)
    import random
    random.seed(42)

    rtg1 = RealTraceGenerator(site='caltech', date_period=('2019-05-01', '2019-08-31'), sequential=True, period=5)
    # atg = GMMsTraceGenerator(site='caltech', date_period=('2019-05-01', '2019-08-31'), n_components=50, random_seed=106)

    for generator in [rtg1]:  # [rtg1, rtg2, atg]:
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print(generator.site)
        env1 = EVChargingEnv(generator, action_type='discrete', project_action=False)
        # env2 = EVChargingEnv(generator, action_type='discrete', project_action=True)
        # env3 = EVChargingEnv(generator, action_type='continuous', project_action=False)
        # env4 = EVChargingEnv(generator, action_type='continuous', project_action=True)
        print("Finished building environments... ")
        start = time.time()
        for env in [env1]:#[env1, env2, env3, env4]:
            all_rewards = 0.
            for _ in range(3):
                observation = env.reset()

                greedy_alg = GreedyAlgorithm(env)
                rewards = 0.
                done = False
                i = 0
                d = defaultdict(list)
                d2 = defaultdict(list)
                while not done:
                    # print(env, " stepping")
                    # print(env.__repr__())
                    # action = np.ones((54,)) * 4
                    action = np.random.randint(size=(54,), low=4, high=5)
                    # action = greedy_alg.get_action(observation)
                    observation, reward, done, info = env.step(action)

                    for k in info['reward']:
                        d[k].append(info['reward'][k])
                    rewards += reward
                    i += 1
                for k, v in d.items():
                    d[k] = np.array(v)
                    print(k, len(d[k]), d[k].min(), d[k].max(), d[k].mean(), d[k].sum())
                    d2[k] = d[k].mean()
                print("total iterations: ", i)
                print("total reward: ", rewards)
                for k, v in d2.items():
                    print("about total: ", k, v * 200)
                all_rewards += rewards
                print("\n")

        print("Total time: ", time.time() - start)
        print("Average rewards: ", all_rewards / 3)
    env.close()
    print(env.max_timestamp)
