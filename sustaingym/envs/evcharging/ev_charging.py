"""
This module contains the class for the central EV Charging environment.
"""
from __future__ import annotations

from typing import Any
import warnings

from acnportal.acnsim.interface import Interface
from acnportal.acnsim.network.sites import caltech_acn, jpl_acn
from acnportal.acnsim.simulator import Simulator
import gym
import numpy as np

from .actions import get_action_space, to_schedule
from .event_generation import AbstractTraceGenerator
from .observations import get_observation_space, get_observation
from .rewards import get_rewards
from .utils import MINS_IN_DAY, DATE_FORMAT


class EVChargingEnv(gym.Env):
    """
    Central class for the EV Charging gym.

    This class uses the Simulator class in the acnportal.acnsim package to
    simulate a day of charging. The simulation can be done using real data
    from Caltech's ACNData or a Gaussian mixture model (GMM) trained on
    that data (script is located in train_artificial_data_model.py). The
    gym has support for the Caltech and JPL site.

    Args:
        site: the charging site, currently supports either 'caltech' or 'jpl'
        date_range: a sequence of length 2 that gives the start and end date of
            a period. Must be a string and have format YYYY-MM-DD.
        period: the number of minutes for the timestep interval. Should divide
            evenly into the number of minutes in a day. Default: 5
        recompute_freq: number of periods elapsed to make the scheduler
            recompute pilot signals, which is in addition to the compute events
            when there is a plug in or unplug. Default: 2
        real_traces: if True, uses real traces for the simulation;
            otherwise, uses Gaussian mixture model (GMM) samples. Default: False
        sequential: if True, simulates days sequentially from the start
            and end date of the data; otherwise, randomly samples a day at the
            beginning of each episode. Ignored when real_traces is False. Default:
            True
        n_components: number of components for GMM
        action_type: either 'continuous' or 'discrete'. If 'discrete' the action
            space is {0, 1, 2, 3, 4} to signify charging rates of 0 A, 8 A, 16 A,
            24 A, and 32 A. If 'continuous', the action space is [0, 1] to signify
            charging rates from 0 A to 32 A. If a charging rate does not achieve
            the minimum pilot signal, it is set to zero.
        verbose: whether to print out warnings when constraints are being violated

    Attributes: TODO??????????? - how to comment attribute type
        max_timestamp: maximum timestamp in a day's simulation
        constraint_matrix: constraint matrix of charging garage
        num_constraints: number of constraints in constraint_matrix
        num_stations: number of EVSEs in the garage
        day: only present when self.real_traces is True. The actual day that
            the simulator is simulating.
        generator: (ArtificialTraceGenerator) a class whose instances can
            sample events that populate the event queue.
        observation_space: the space of available observations
        action_space: the space of actions. Note that not all actions in the
            action space may be feasible.
        events: the current EventQueue for the simulation
        cn: charging network in use
        simulator: internal simulator from acnportal package
        interface: interface wrapper for simulator
    """
    metadata: dict = {"render_modes": []}

    def __init__(self, data_generator: AbstractTraceGenerator,
                 action_type: str = 'discrete',
                 verbose: int = 0):
        self.data_generator = data_generator
        self.day = self.data_generator.day
        self.site = data_generator.site
        self.period = data_generator.period
        self.max_timestamp = MINS_IN_DAY // self.period
        self.action_type = action_type
        self.verbose = verbose
        if verbose < 2:
            warnings.filterwarnings("ignore")

        # Set up charging network parameters
        self.cn = caltech_acn() if self.site == 'caltech' else jpl_acn()
        self.evse_name_to_idx = {evse: i for i, evse in enumerate(self.cn.station_ids)}

        # Define observation space and action space
        # Observations are dictionaries describing arrivals and departures of EVs,
        # constraint matrix, current magnitude bounds, demands, phases, and timesteps
        self.observation_space = get_observation_space(self.cn)
        # Define action space, which is the charging rate for all EVs
        self.action_space = get_action_space(self.cn, action_type)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """
        Step the environment.

        Call the step function of the internal simulator and generate the
        observation, reward, done, info tuple required by OpenAI gym.

        Args:
        action: array of shape (number of stations,) with entries in the set
            {0, 1, 2, 3, 4} if action_type == 'discrete'; otherwise, entries
            should fall in the range [0, 1]

        Returns:
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
        schedule = to_schedule(action, self.cn, self.action_type)
        if self.simulator.event_queue.empty():
            done = True
            cur_event = None
        else:
            cur_event = self.simulator.event_queue.queue[0][1]
            done = self.simulator.step(schedule)
            self.simulator._resolve = False  # work-around to keep iterating
        # Next timestamp
        next_timestamp = self.max_timestamp if done else self.simulator.event_queue.queue[0][0]
        # Retrieve environment information
        observation, info = get_observation(self.interface, self.evse_name_to_idx)
        reward, reward_info = get_rewards(self.interface, schedule, self.prev_timestamp, self.timestamp, next_timestamp, cur_event)
        # Update timestamps
        self.prev_timestamp, self.timestamp = self.timestamp, next_timestamp
        info.update(reward_info)
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
        if options:
            if "verbose" in options:
                self.verbose = options["verbose"]
        # super().reset(seed=seed)

        # Initialize charging network
        self.cn = caltech_acn() if self.site == 'caltech' else jpl_acn()
        # Initialize event queue
        self.events, num_plugs = self.data_generator.get_event_queue()
        self.day = self.data_generator.day
        if self.verbose >= 1:
            print(f"Simulating day {self.day.strftime(DATE_FORMAT)} with {num_plugs} plug in events. ")
        # Initialize simulator and interface
        self.simulator = Simulator(network=self.cn, scheduler=None, events=self.events, start=self.day, period=self.period, verbose=False)
        self.interface = Interface(self.simulator)
        # Initialize time steps
        self.prev_timestamp, self.timestamp = 0, self.simulator.event_queue.queue[0][0]
        # Retrieve environment information
        return get_observation(self.interface, self.evse_name_to_idx, get_info=return_info)

    def render(self) -> None:
        """Render environment."""
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the environment.

        Delete simulator, interface, events, and charging network.
        """
        del self.simulator, self.interface, self.events, self.cn


if __name__ == "__main__":
    from collections import defaultdict
    from .event_generation import RealTraceGenerator, ArtificialTraceGenerator

    np.random.seed(42)
    import random
    random.seed(42)

    rtg1 = RealTraceGenerator(site='caltech', date_range=['2018-11-05', '2018-11-11'], sequential=True)
    rtg2 = RealTraceGenerator(site='caltech', date_range=['2018-11-05', '2018-11-13'], sequential=False)
    atg = ArtificialTraceGenerator(site='caltech', date_range=['2018-11-05', '2018-11-15'], n_components=50)

    for generator in [rtg1, rtg2, atg]:
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print("----------------------------")
        print(generator.site)
        env = EVChargingEnv(generator, action_type='discrete')
        for _ in range(10):
            observation = env.reset()
            all_timestamps = sorted([event[0] for event in env.events._queue])

            rewards = 0
            done = False
            i = 0
            action = np.ones(shape=(54,), ) * 2
            d = defaultdict(list)
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
