"""This module contains the class for the central EV Charging environment."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import warnings

import acnportal.acnsim as acns
import gym
import numpy as np

from .actions import get_action_space, to_schedule
from .event_generation import AbstractTraceGenerator, RealTraceGenerator
from .observations import get_observation_space, get_observation
from .rewards import get_rewards
from .utils import MINS_IN_DAY, DATE_FORMAT, ActionType, site_str_to_site


class EVChargingEnv(gym.Env):
    """
    Central class for the EV Charging gym.

    This class uses the Simulator class in the acnportal package to
    simulate a day of charging. The simulation can be done using real data
    from Caltech's ACNData or a Gaussian mixture model (GMM) trained on
    that data (see train_artificial_data_model.py). The gym supports the
    Caltech and JPL sites.

    Args:
        data_generator: a subclass of AbstractTraceGenerator
        action_type: either 'continuous' or 'discrete'. If 'discrete', the
            action space is {0, 1, 2, 3, 4}. If 'continuous', it is [0, 4].
            These are then scaled to charging rates from 0 A to 32 A. If a
            charging rate does not achieve the minimum pilot signal, it is set
            to zero.
        verbose: whether to print out warnings when constraints are being violated

    Attributes:
        max_timestamp: maximum timestamp in a day's simulation
        constraint_matrix: constraint matrix of charging garage
        num_constraints: number of constraints in constraint_matrix
        num_stations: number of EVSEs in the garage
        day: only present when self.real_traces is True. The actual day that
            the simulator is simulating.
        generator: (AbstractTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        observation_space: the space of available observations
        action_space: the space of actions. Note that not all actions in the
            action space may be feasible.
        events: the current EventQueue for the simulation
        evs: the entire list of EVs during simulation day
        cn: charging network in use
        simulator: internal simulator from acnportal package
        interface: interface wrapper for simulator
    """
    metadata: dict = {"render_modes": []}

    def __init__(self, data_generator: AbstractTraceGenerator,
                 action_type: ActionType = 'discrete', verbose: int = 0):
        self.data_generator = data_generator
        self.site = data_generator.site
        self.period = data_generator.period
        self.max_timestamp = MINS_IN_DAY // self.period
        self.action_type = action_type
        self.verbose = verbose
        if verbose < 2:
            warnings.filterwarnings("ignore")

        # Set up charging network parameters
        self.cn = site_str_to_site(self.site)
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Define observation space and action space
        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.observation_space = get_observation_space(self.cn)
        # Define action space, which is the charging rate for all EVs
        self.action_space = get_action_space(self.cn, action_type)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[Any, Any]]:
        """
        Step the environment.

        Call the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
            action: array of shape (number of stations,) with entries in the
                set {0, 1, 2, 3, 4} if action_type == 'discrete'; otherwise,
                entries should fall in the range [0, 4].

        Returns: TODO fix depending on rewards
            observation: dict
            - "arrivals": array of shape (num_stations,). If the EVSE
                corresponding to the index is active, the entry is the number
                of periods that have elapsed since arrival (always negative).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
            - "est_departures": array of shape (num_stations,). If the
                EVSE corresponding to the index is active, the entry is the
                estimated number of periods before departure (always positive).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
            - "constraint_matrix": array of shape (num_constraints,
                num_stations). Each row indicates a constraint on the aggregate
                current of EVSEs with non-zero entries. Entry (i, j) indicates
                the fractional amount that station j contributes to constraint
                i.
            - "magnitudes": array of shape (num_constraints,). The absolute
                value of the maximum amount of allowable current row i of the
                constraint_matrix can handle.
            - "demands": amount of charge demanded in units A*period
            - "phases": the phase of a station id
            - "timestep": an integer between 0 and MINS_IN_DAY // period
                indicating the timestep
            reward: float
            - objective function to maximize
            done: bool
            - indicator to whether the day's simulation has finished
            info: dict
            - "charging_rates": pd.DataFrame. History of charging rates as
                provided by the internal simulator.
            - "active_evs": List of all EVs currenting charging.
            - "pilot_signals": entire history of pilot signals throughout simulation.
            - "active_sessions": List of active sessions.
            - "departures": array of shape (num_stations,). If the
                EVSE corresponding to the index is active, the entry is the
                actual number of periods before departure (always positive).
                Otherwise, for indices with corresponding non-active EVSEs,
                the entry is zero.
        """
        # TODO: make action fit constraints somehow? or nah
        # Step internal simulator
        schedule = to_schedule(action, self.cn, self.action_type)  # transform action to pilot signals
        done = self.simulator.step(schedule)
        self.simulator._resolve = False  # work-around to keep iterating
        # Next timestamp
        next_timestamp = self.max_timestamp if done else self.simulator.event_queue.queue[0][0]
        # Retrieve environment information
        observation, info = get_observation(self.interface, self.evse_name_to_idx)
        reward, reward_info = get_rewards(self.interface, schedule, self.prev_timestamp, self.timestamp, next_timestamp, self.period, done, self.evs)
        # Update timestamps
        self.prev_timestamp, self.timestamp = self.timestamp, next_timestamp
        info['reward_calculation'] = reward_info

        return observation, reward, done, info

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None
              ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """
        Reset the environment.

        Prepare for next episode by re-creating charging network,
        generating new events, creating simulation and interface,
        and resetting the timestamps.

        Args:
            seed: random seed to reset environment
            return_info: whether information should be returned as well
            options: dictionary containing options for resetting.
            - "verbose": reset verbose factor
            - "p": probability distribution for choosing GMM for day's
                simulation, has 1 probability of choosing first GMM by
                default. By default, 3 GMMs can be chosen, so "p" should be a
                sequence of floats of length 3 that sum up to 1. Ignored when
                self.real_traces is True. TODO
        """
        # super().reset(seed=seed) TODO
        if options:
            if 'verbose' in options:
                self.verbose = options['verbose']

        # Initialize charging network
        self.cn = site_str_to_site(self.site)
        # Initialize event queue
        self.events, self.evs, num_plugs = self.data_generator.get_event_queue()

        if self.verbose >= 1:
            if type(self.data_generator) is RealTraceGenerator:
                print(f'Simulating day {self.data_generator.day.strftime(DATE_FORMAT)} with {num_plugs} plug in events. ')
            else:
                print(f'Simulating {num_plugs} plug in events. ')

        # Initialize simulator and interface
        if type(self.data_generator) is RealTraceGenerator:
            day = self.data_generator.day
        else:
            day = datetime.now()
        self.simulator = acns.Simulator(network=self.cn, scheduler=None, events=self.events, start=day, period=self.period, verbose=False)
        self.interface = acns.Interface(self.simulator)
        # Initialize time steps
        self.prev_timestamp, self.timestamp = 0, self.simulator.event_queue.queue[0][0]
        # Retrieve environment information
        return get_observation(self.interface, self.evse_name_to_idx, get_info=return_info)

    def render(self) -> None:
        """Render environment."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment. Delete internal variables."""
        del self.simulator, self.interface, self.events, self.cn


if __name__ == "__main__":
    from .event_generation import GMMsTraceGenerator
    from acnportal.acnsim.events import PluginEvent

    np.random.seed(42)
    import random
    random.seed(42)

    rtg1 = RealTraceGenerator(site='caltech', date_range=('2018-11-05', '2018-11-11'), sequential=True, period=5)
    rtg2 = RealTraceGenerator(site='caltech', date_range=('2018-11-05', '2018-11-11'), sequential=True, period=10)
    # rtg2 = RealTraceGenerator(site='caltech', date_range=['2018-11-05', '2018-11-13'], sequential=False)
    # atg = GMMsTraceGenerator(site='caltech', date_range=['2018-11-05', '2018-11-15'], n_components=50)

    for generator in [rtg1, rtg2]:  # [rtg1, rtg2, atg]:
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print(generator.site)
        env = EVChargingEnv(generator, action_type='discrete')
        for _ in range(2):
            observation = env.reset()
            # all_timestamps = sorted([event[0] for event in env.events._queue])

            offline_reward_calc = 0
            for event in env.events.queue:
                if isinstance(event[1], PluginEvent):
                    amt_amp_mins = min(env.interface._convert_to_amp_periods(event[1].ev.requested_energy, event[1].ev.station_id) * env.period,
                                       (event[1].ev.departure - event[1].ev.arrival) * env.period * 16)
                    # offline_reward_calc += amt_amp_mins# (event[1].ev.departure - event[1].ev.arrival) * env.period * 16

            rewards = 0.
            done = False
            i = 0
            action = np.ones(shape=(54,), ) * 2
            # d = defaultdict(list)
            while not done:
                observation, reward, done, info = env.step(action)
                # for x in ["charging_cost", "charging_reward", "constraint_punishment", "remaining_amp_periods_punishment"]:
                # d[x].append(info[x])
                # d["reward"].append(reward)
                # print(observation['demands'][3])
                rewards += reward
                i += 1
            # for k, v in d.items():
            #     print(k)
            #     d[k] = np.array(v)
            #     print(d[k].min(), d[k].max(), d[k].mean(), d[k].sum())

            print()
            print()
            print("total iterations: ", i)
            print("total reward: ", rewards)
    # 1 -> d
    #  charging cost -864
    #  charging reward 16 - 48
    #  constraint punishment 0
    #  remaining amp periods punishment -15 to -38
    # 2
    #  charging cost -1728
    #  charging reward 32 - 96
    #  constraint punishment -411
    #  remaining amp periods -10 to -60
    # 3
    #  charging cost -2592
    #  charging reward 48 - 144
    #  constraint punishment -1374
    #  remaining amp periods -5 to -35
    # charging cost: 0.05
    # charging reward: 1
    # constraint punishment: 5
    # remaining amp periods: 10

    env.close()
    print(env.max_timestamp)
