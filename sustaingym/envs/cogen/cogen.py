"""
This module implements the CogenEnv class
"""
from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
import onnxruntime as rt
import pandas as pd

from sustaingym.data.cogen import load_ambients


class CogenEnv(gym.Env):
    """
    Actions:
        Type: Dict(Box(1), Discrete(2), Discrete(2), Box(1),
                   Box(1), Discrete(2), Discrete(2), Box(1),
                   Box(1), Discrete(2), Discrete(2), Box(1),
                   Box(1), Box(1), Discrete(12, start=1))
        Action                        Min                 Max
        GT1_PWR (MW)                  41.64               168.27
        GT1_PAC_FFU (binary)          0                   1
        GT1_EVC_FFU (binary)          0                   1
        HR1_HPIP_M_PROC (klb/hr)      403.16              819.57
        GT2_PWR (MW)                  41.49               168.41
        GT2_PAC_FFU (binary)          0                   1
        GT2_EVC_FFU (binary)          0                   1
        HR2_HPIP_M_PROC (klb/hr)      396.67              817.35
        GT3_PWR (MW)                  46.46               172.44
        GT3_PAC_FFU (binary)          0                   1
        GT3_EVC_FFU (binary)          0                   1
        HR3_HPIP_M_PROC (klb/hr)      439.00              870.27
        ST_PWR (MW)                   25.65               83.54
        IPPROC_M (klb/hr)             -1218.23            -318.05
        CT_NrBays (int)               1                   12

    Observation:
        Type: Dict(Box(1), Action_Dict,
                   Box(forecast_horizon + 1), Box(forecast_horizon + 1),
                   Box(forecast_horizon + 1), Box(forecast_horizon + 1),
                   Box(forecast_horizon + 1), Box(forecast_horizon + 1),
                   Box(forecast_horizon + 1))
        Observation                   Min                 Max
        Time (fraction of day)        0                   1
        Previous action (dict)        see above           see above
        Temperature forecast (F)      32                  115
        Pressure forecast (psia)      14                  15
        Humidity forecast (fraction)  0                   1
        Target net power (MW)         0                   700
        Target process steam (klb/hr) 0                   1300
        Electricity price ($/MWh)     0                   1500
        Natural gas price ($/MMBtu)   0                   7
    """
    def __init__(self,
                 renewables_magnitude: float = None,  # TODO: implement renewables
                 ramp_penalty: float = 2.0,
                 supply_imbalance_penalty: float = 1000,
                 constraint_violation_penalty: float = 1000,
                 forecast_horizon: int = 3,
                 forecast_noise_std: float = 0.1,
                 ):
        """
        Constructs the CogenEnv object
        """
        self.ramp_penalty = ramp_penalty
        self.supply_imbalance_penalty = supply_imbalance_penalty
        self.constraint_violation_penalty = constraint_violation_penalty
        self.forecast_horizon = forecast_horizon
        # load the ambient conditions dataframes
        self.ambients_dfs = load_ambients.construct_df(renewables_magnitude=renewables_magnitude)
        self.n_days = len(self.ambients_dfs)
        self.timesteps_per_day = len(self.ambients_dfs[0])
        assert (self.forecast_horizon >= 0 and self.forecast_horizon < self.timesteps_per_day - 1), "forecast_horizon must be between 0 and timesteps_per_day - 1"

        # load the onnx model
        self._model = rt.InferenceSession('sustaingym/data/cogen/onnx_model/model.onnx')  # TODO(Chris): pkgutil

        # load model parameters from JSON file into DataFrame
        #     id  (index)      str
        #     min          float64
        #     max          float64
        #     unit             str
        #     data_type        str
        with open('sustaingym/data/cogen/onnx_model/model.json', 'r') as f:  # TODO(Chris): pkgutil
            json_data = json.load(f)
        inputs_table = pd.DataFrame(json_data['inputs'])
        inputs_table.drop(columns=['index'], inplace=True)
        inputs_table.set_index('id', inplace=True)

        # output_labels = [json_data['outputs'][i]['id'] for i in range(len(json_data['outputs']))]

        # action space is power output, evaporative cooler switch, power augmentation switch, and equivalent
        # process steam flow for generators 1, 2, and 3, as well as steam turbine power output, steam flow
        # through condenser, and number of cooling bays employed
        self.action_space = gym.spaces.Dict({
            'GT1_PWR': gym.spaces.Box(low=inputs_table.loc['GT1_PWR', 'min'], high=inputs_table.loc['GT1_PWR', 'max'], shape=(1,), dtype=np.float32),
            'GT1_PAC_FFU': gym.spaces.Discrete(2),
            'GT1_EVC_FFU': gym.spaces.Discrete(2),
            'HR1_HPIP_M_PROC': gym.spaces.Box(low=inputs_table.loc['HR1_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR1_HPIP_M_PROC', 'max'], shape=(1,), dtype=np.float32),
            'GT2_PWR': gym.spaces.Box(low=inputs_table.loc['GT2_PWR', 'min'], high=inputs_table.loc['GT2_PWR', 'max'], shape=(1,), dtype=np.float32),
            'GT2_PAC_FFU': gym.spaces.Discrete(2),
            'GT2_EVC_FFU': gym.spaces.Discrete(2),
            'HR2_HPIP_M_PROC': gym.spaces.Box(low=inputs_table.loc['HR2_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR2_HPIP_M_PROC', 'max'], shape=(1,), dtype=np.float32),
            'GT3_PWR': gym.spaces.Box(low=inputs_table.loc['GT3_PWR', 'min'], high=inputs_table.loc['GT3_PWR', 'max'], shape=(1,), dtype=np.float32),
            'GT3_PAC_FFU': gym.spaces.Discrete(2),
            'GT3_EVC_FFU': gym.spaces.Discrete(2),
            'HR3_HPIP_M_PROC': gym.spaces.Box(low=inputs_table.loc['HR3_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR3_HPIP_M_PROC', 'max'], shape=(1,), dtype=np.float32),
            'ST_PWR': gym.spaces.Box(low=inputs_table.loc['ST_PWR', 'min'], high=inputs_table.loc['ST_PWR', 'max'], shape=(1,), dtype=np.float32),
            'IPPROC_M': gym.spaces.Box(low=inputs_table.loc['IPPROC_M', 'min'], high=inputs_table.loc['IPPROC_M', 'max'], shape=(1,), dtype=np.float32),
            'CT_NrBays': gym.spaces.Discrete(12, start=1)
        })

        # define the observation space
        self.observation_space = gym.spaces.Dict({
            'Time': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'Prev_Action': self.action_space,
            'TAMB': gym.spaces.Box(low=inputs_table.loc['TAMB', 'min'], high=inputs_table.loc['TAMB', 'max'], shape=(forecast_horizon+1,), dtype=np.float32),
            'PAMB': gym.spaces.Box(low=inputs_table.loc['PAMB', 'min'], high=inputs_table.loc['PAMB', 'max'], shape=(forecast_horizon+1,), dtype=np.float32),
            'RHAMB': gym.spaces.Box(low=inputs_table.loc['RHAMB', 'min'], high=inputs_table.loc['RHAMB', 'max'], shape=(forecast_horizon+1,), dtype=np.float32),
            'Target_Power': gym.spaces.Box(low=0, high=700, shape=(forecast_horizon+1,), dtype=np.float32),
            'Target_Steam': gym.spaces.Box(low=0, high=1300, shape=(forecast_horizon+1,), dtype=np.float32),
            'Energy_Price': gym.spaces.Box(low=0, high=1500, shape=(forecast_horizon+1,), dtype=np.float32),
            'Gas_Price': gym.spaces.Box(low=0, high=7, shape=(forecast_horizon+1,), dtype=np.float32)
        })

        # define the current info
        self.current_info = None

    def _forecast_values_from_time(self, day: int, time_step: int) -> tuple[list[float], list[float], list[float],
                                                                            list[float], list[float], list[float],
                                                                            list[float]]:
        """Returns the forecast values starting at the given day and time step
        for the following self.forecast_horizon + 1 time steps."""
        slice = self.ambients_dfs[day].iloc[time_step:min(time_step+self.forecast_horizon+1, self.timesteps_per_day)]
        # fix so that if the slice is not long enough, it will take the first values of the next day
        # TODO: figure out what to do if we're on the last day and there is no next day
        if len(slice) < self.forecast_horizon + 1:
            slice = pd.concat([slice, self.ambients_dfs[day+1].iloc[:self.forecast_horizon + 1 - len(slice)]])
        return (slice['Ambient Temperature'].to_numpy(), slice['Ambient Pressure'].to_numpy(), slice['Ambient rel. Humidity'].to_numpy(),
                slice['Target Net Power'].to_numpy(), slice['Target Process Steam'].to_numpy(), slice['Energy Price'].to_numpy(),
                slice['Gas Price'].to_numpy())

    def reset(self, seed: int | None = None, options: dict | None = None) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an instance of an episode for the Cogen environment.

        Args:
            seed: optional seed vaue for controlling seed of np.random attributes
            return_info: determines if returned observation includes additional
                info or not (not implemented)
            options: includes optional settings like reward type (not implemented)

        Returns:
            tuple containing the initial observation for env's episode
        """
        super().reset(seed=seed)

        # randomly pick a day for the episode
        # subtract 1 as temporary fix to make sure we don't go over the number of days with lookahead window
        if seed is None:
            self.current_day = self.np_random.integers(low=0, high=self.n_days-1)
        else:
            self.current_day = seed % self.n_days

        self.current_timestep = 0  # keeps track of which timestep we are on
        self.current_terminated = False
        self.current_reward = None

        # initial action is drawn randomly from the action space
        # not sure if this is reasonable, TODO: check this
        self.current_action = self.action_space.sample()

        forecast_values = self._forecast_values_from_time(self.current_day, self.current_timestep)

        # set up initial observation
        self.obs = {
            'Time': np.array([self.current_timestep / self.timesteps_per_day], dtype=np.float32),
            'Prev_Action': self.current_action,
            'TAMB': forecast_values[0],
            'PAMB': forecast_values[1],
            'RHAMB': forecast_values[2],
            'Target_Power': forecast_values[3],
            'Target_Steam': forecast_values[4],
            'Energy_Price': forecast_values[5],
            'Gas_Price': forecast_values[6]
        }

        info = {
            'Operating constraint violation': None,
            'Demand constraint violation': None
        }
        return self.obs, info

    def _dyn_constraint_volation(self, input_data: np.ndarray, output_data: np.ndarray) -> np.ndarray:
        """
        Computes the dynamic operating constraint violation for one
        timestep of plant operation.

        Args:
            input_data: shape [18], plant model inputs for the current timestep
            output_data: shape [29], plant model outputs for the current timestep

        Returns:
            cv: shape [16], constraint violations for the current timestep
        """
        cv = np.zeros(16)

        # GT1_PWR
        cv[0] = max(0, output_data[9] - input_data[5])  # min violation
        cv[1] = max(0, input_data[5] - output_data[10])  # max violation
        # GT1_HR
        cv[2] = max(0, output_data[15] - input_data[12])  # min violation
        cv[3] = max(0, input_data[12] - output_data[16])  # max violation

        # GT2_PWR
        cv[4] = max(0, output_data[11] - input_data[8])  # min violation
        cv[5] = max(0, input_data[8] - output_data[12])  # max violation
        # GT2_HR
        cv[6] = max(0, output_data[17] - input_data[13])  # min violation
        cv[7] = max(0, input_data[13] - output_data[18])  # max violation

        # GT3_PWR
        cv[8] = max(0, output_data[13] - input_data[11])  # min violation
        cv[9] = max(0, input_data[11] - output_data[14])  # max violation
        # GT3_HR
        cv[10] = max(0, output_data[19] - input_data[14])  # min violation
        cv[11] = max(0, input_data[14] - output_data[20])  # max violation

        # ST_PWR
        cv[12] = max(0, output_data[24] - input_data[15])  # min violation
        cv[13] = max(0, input_data[15] - output_data[25])  # max violation
        # IPPROC
        cv[14] = max(0, input_data[16] - output_data[22])  # max violation
        cv[15] = max(0, input_data[16] - output_data[23])  # max violation

        return cv

    def _compute_reward(self, obs: dict[str, Any], action: dict[str, Any]) -> float:
        """Computes the reward for the current timestep.

        Reward is the negative of the sum of the four following components:
        - total generation fuel consumption
        - total ramp cost
        - penalty for steam/energy non-delivery
        - penalty for dynamic operating constraint violation

        Args:
            obs: the current state observation
            action: the current action

        Returns:
            reward: the reward for the current timestep
        """
        # run the cc model on the action
        model_input = np.array([
            obs['TAMB'][0], obs['PAMB'][0], obs['RHAMB'][0],
            action['GT1_PAC_FFU'], action['GT1_EVC_FFU'], action['GT1_PWR'][0],
            action['GT2_PAC_FFU'], action['GT2_EVC_FFU'], action['GT2_PWR'][0],
            action['GT3_PAC_FFU'], action['GT3_EVC_FFU'], action['GT3_PWR'][0],
            action['HR1_HPIP_M_PROC'][0], action['HR2_HPIP_M_PROC'][0],
            action['HR3_HPIP_M_PROC'][0], action['ST_PWR'][0],
            action['IPPROC_M'][0], action['CT_NrBays']
        ], dtype=np.float32)

        model_output = self._model.run(None, {self._model.get_inputs()[0].name: [model_input]})[0][0]
        # print(model_output)
        # extract the fuel consumption (klb/hr)
        total_fuel = model_output[-8]

        # compute the ramp cost
        ramp = (np.abs(action['GT1_PWR'][0] - obs['Prev_Action']['GT1_PWR'][0])
                + np.abs(action['GT2_PWR'][0] - obs['Prev_Action']['GT2_PWR'][0])
                + np.abs(action['GT3_PWR'][0] - obs['Prev_Action']['GT3_PWR'][0])
                + np.abs(action['ST_PWR'][0] - obs['Prev_Action']['ST_PWR'][0]))
        ramp_cost = self.ramp_penalty * ramp

        # compute the penalty for steam/energy non-delivery
        # this will be a relu penalty, so if the target is not met, the penalty is the difference
        # between the target and the actual value
        steam_penalty = np.maximum(0, obs['Target_Steam'][0] - model_output[-1])
        energy_penalty = np.maximum(0, obs['Target_Power'][0] - model_output[-2])
        non_delivery_penalty = self.supply_imbalance_penalty * (steam_penalty + energy_penalty)

        # compute the penalty for dynamic operating constraint violation
        dyn_cv = self._dyn_constraint_volation(model_input, model_output).sum()
        dyn_cv_penalty = self.constraint_violation_penalty * dyn_cv

        # compute the total reward
        reward = -(total_fuel + ramp_cost + non_delivery_penalty + dyn_cv_penalty)
        return reward

    def _terminated(self) -> bool:
        """Determines if the episode is terminated or not.

        Returns:
            terminated: True if the episode is terminated, False otherwise
        """
        return self.current_timestep > self.timesteps_per_day - 1

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action: an action provided by the environment

        Returns:
            tuple containing the next observation, reward, terminated flag, truncated flag, and info dict
        """
        # compute the loss of taking the action
        self.current_reward = self._compute_reward(self.obs, action)

        # update the current action
        self.current_action = action

        # update the current timestep
        self.current_timestep += 1

        # update the current observation
        forecast_values = self._forecast_values_from_time(self.current_day, self.current_timestep)

        self.obs = {
            'Time': np.array([self.current_timestep / self.timesteps_per_day], dtype=np.float32),
            'Prev_Action': self.current_action,
            'TAMB': forecast_values[0],
            'PAMB': forecast_values[1],
            'RHAMB': forecast_values[2],
            'Target_Power': forecast_values[3],
            'Target_Steam': forecast_values[4],
            'Energy_Price': forecast_values[5],
            'Gas_Price': forecast_values[6]
        }

        # update the current done
        self.current_terminated = self._terminated()

        # update the current info
        self.current_info = {
            'Operating constraint violation': None,
            'Demand constraint violation': None
        }

        # always False due to no intermediate stopping conditions
        truncated = False

        return self.obs, self.current_reward, self.current_terminated, truncated, self.current_info

    def close(self):
        return
