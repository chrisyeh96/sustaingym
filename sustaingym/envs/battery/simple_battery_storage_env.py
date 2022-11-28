"""
The module implements the BatteryStorageEnv class.
"""
from __future__ import annotations

import pkgutil
from io import StringIO
import os
import sys
sys.path.append('../')
from typing import Union

import numpy as np
import gym
from gym import spaces
import pandas as pd


class BatteryStorageEnv(gym.Env):
    """
    Actions:
        Type: Box(1)
        Action                              Min                     Max
        Charging/Discharing Power           -max discharge rate     max charge rate
    Observation:
        Type: Box(2)
                                            Min                     Max
        Energy storage level (MWh)          -Inf                    Inf
        Current Electricity Price ($/MWh)   -Inf                    Inf
        Time (fraction of day)               0                       1
    """
    # Minimum storage level (MWh)
    ENERGY_MIN = 0.0
    # Maximum storage level (MWh)
    ENERGY_MAX = 60.0
    # Initial storage level (MWh)
    ENERGY_INIT = 30.0
    # Max charge rate (MW)
    MAX_CHARGE_RATE = 4.0
    # Max discharge rate (MW)
    MAX_DISCHARGE_RATE = 2.0
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
    # file_path to dataset
    FILE_PATH = None
    # indicates whether local file will be used instead
    LOCAL_FILE = False

    def __init__(self, render_mode: str | None = None, env_config: dict | None = None, LOCAL_FILE_PATH: str | None = None):
        """
        Constructs instance of BatteryStorageEnv class.

        Args:
            env_config: optional argument which allows user to input dictionary of
            alternate values for class' attributes
        Returns:
            N/A
        """
        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        if env_config:
            # replace instance attributes if value specified in env_config
            for key, val in env_config.items():
                try:
                    self.__dict__[key] = val
                except KeyError:
                    raise KeyError(key)

        # store custom file path to data
        self.LOCAL_PATH = LOCAL_FILE_PATH
        # price dataset
        self.df_price_data = self._get_pricing_data()
        # number of prices in dataset
        self.price_data_len = self.df_price_data.shape[0]
        # action space is a single value ranging between the max charge rates
        self.action_space = spaces.Box(low=-self.MAX_DISCHARGE_RATE,
                                       high=self.MAX_CHARGE_RATE, shape=(1, ),
                                       dtype=np.float32)
        # observation space is current energy level and current price
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3, ), dtype=np.float32
        )
        # labels if environment starts with a reset to ensure standardization
        self.init = False

    def _get_pricing_data(self) -> pd.DataFrame:
        """
        Generates temporal pricing data.

        Args:
            N/A
        Returns:
            pandas dataframe containing price data
        """
        if self.LOCAL_PATH is not None:
            return pd.read_csv(self.LOCAL_PATH)
        else:
            csv_path = os.path.join('data', 'electricity_price_data', 'prices_2022.csv.gz')
            bytes_data = pkgutil.get_data('sustaingym', csv_path)
            assert bytes_data is not None
            s = bytes_data.decode('utf-8')
            data = StringIO(s)
            df_prices = pd.read_csv(data)
            return df_prices

    def reset(self, *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None) -> Union[np.ndarray, tuple]:  # changed from dictionary type
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
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        # initial running average price
        self.avg_price = 0.0
        # initial energy storage level
        self.energy_lvl = self.ENERGY_INIT
        self.reward_type = 0  # default reward type without moving price average
        if options and 'reward' in options.keys():
            if options.get('reward') == 1:
                self.reward_type = 1
        # get random initial starting price position
        self.idx = self.rng.integers(
            low=0, high=self.price_data_len - self.MAX_STEPS_PER_EPISODE, size=1
        )
        self.count = 1  # counter for the step in current episode
        self.init = True
        if self.LOCAL_PATH is not None:
            self.curr_price = self.df_price_data.iloc[self.idx, 1]
        else:
            self.curr_price = float(self.df_price_data.iloc[self.idx, 2])

        if self.LOCAL_PATH is not None:
            time = self.df_price_data.iloc[self.idx, 0]
        else:
            date = self.df_price_data.iloc[self.idx, 0].to_string()
            idx = date.find('T')
            time_day = date[idx+1:idx+6]
            hours = int(time_day[0:2])
            minutes = int(time_day[3:5])
            time = (hours + minutes/60) / 24
        observation = np.array([self.energy_lvl, self.curr_price, time], dtype=np.float32)
        info = self._get_info(self.curr_price)
        return (observation, info) if return_info else observation

    def step(self, action: float) -> tuple:
        """
        Executes a single time step in the environments current trajectory.

        Args:
            action: float value representing the amount of charging/discharging power to
            exert during this time step
        Returns:
            tuple representing the resulting state from that action
        """
        assert self.init
        if self.LOCAL_PATH is not None:
            self.curr_price = self.df_price_data.iloc[self.idx + self.count, 0]
        else:
            self.curr_price = float(self.df_price_data.iloc[self.idx, 2])
        if self.LOCAL_PATH is not None:
            time = self.df_price_data.iloc[self.idx, 0]
        else:
            date = self.df_price_data.iloc[self.idx, 0].to_string()
            idx = date.find('T')
            time_day = date[idx+1:idx+6]
            hours = int(time_day[0:2])
            minutes = int(time_day[3:5])
            time = (hours + minutes/60) / 24
        info = self._get_info(self.curr_price)
        # profit reward function
        if self.reward_type == 0:
            if action < 0:
                pwr = min(
                    abs(action), (self.energy_lvl - self.ENERGY_MIN) / self.TIME_STEP_DURATION
                )
                reward = self.curr_price*self.DISCHARGE_EFFICIENCY*pwr
                self.energy_lvl -= self.DISCHARGE_EFFICIENCY*pwr*self.TIME_STEP_DURATION
            elif action > 0:
                pwr = min(
                    action, (self.ENERGY_MAX - self.energy_lvl) / self.TIME_STEP_DURATION
                )
                reward = -self.curr_price*(1 / self.CHARGE_EFFICIENCY)*pwr
                self.energy_lvl += (1 / self.CHARGE_EFFICIENCY)*pwr*self.TIME_STEP_DURATION
            else:
                reward = 0
        # running average over prices reward function
        else:
            if action < 0:
                pwr = min(
                    abs(action), (self.energy_lvl - self.ENERGY_MIN) / self.TIME_STEP_DURATION
                )
                reward = (self.curr_price - info['running_avg'])*self.DISCHARGE_EFFICIENCY*pwr
                self.energy_lvl -= self.DISCHARGE_EFFICIENCY*pwr*self.TIME_STEP_DURATION
            elif action > 0:
                pwr = min(
                    action, (self.ENERGY_MAX - self.energy_lvl) / self.TIME_STEP_DURATION
                )
                reward = (info['running_avg'] - self.curr_price)*(1 / self.CHARGE_EFFICIENCY)*pwr
                self.energy_lvl += (1 / self.CHARGE_EFFICIENCY)*pwr*self.TIME_STEP_DURATION
            else:
                reward = 0
        self.count += 1
        if self.count >= self.MAX_STEPS_PER_EPISODE:
            done = True
        else:
            done = False
        state = np.array([self.energy_lvl, self.curr_price, time], dtype=np.float32)
        return (state, float(reward), done, info)

    def _get_info(self, curr_price: float) -> dict[str, float]:
        self.avg_price = (1.0 - self.eta)*self.avg_price + self.eta*curr_price
        return {"running_avg": self.avg_price}

    def render(self):
        raise NotImplementedError

    def close(self):
        return
