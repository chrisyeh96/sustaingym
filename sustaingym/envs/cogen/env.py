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
from sustaingym.data.utils import read_bytes, read_to_bytesio


class CogenEnv(gym.Env):
    """
    This environment's API is known to be compatible with Gymnasium v0.28, v0.29.

    Actions:

    .. code:: none

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

    .. code:: none

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

    Args:
        renewables_magnitude: TODO
        ramp_penalty: TODO
        supply_imbalance_penalty: TODO
        constraint_violation_penalty: TODO
        forecast_horizon: TODO
        forecast_noise_std: TODO
    """
    def __init__(self,
                 renewables_magnitude: float = 0.,
                 ramp_penalty: float = 2.,
                 supply_imbalance_penalty: float = 1000,
                 constraint_violation_penalty: float = 1000,
                 forecast_horizon: int = 3,
                 forecast_noise_std: float = 0.1,
                 ):
        self.ramp_penalty = ramp_penalty
        self.supply_imbalance_penalty = supply_imbalance_penalty
        self.constraint_violation_penalty = constraint_violation_penalty
        self.forecast_horizon = forecast_horizon
        # load the ambient conditions dataframes
        self.ambients_dfs = load_ambients.construct_df(renewables_magnitude=renewables_magnitude)
        self.n_days = len(self.ambients_dfs)
        self.timesteps_per_day = len(self.ambients_dfs[0])
        assert (0 <= self.forecast_horizon < self.timesteps_per_day - 1), 'forecast_horizon must be in [0, timesteps_per_day - 1)'

        self.spec = gym.envs.registration.EnvSpec(
            id='sustaingym/CogenEnv-v0',
            entry_point='sustaingym.envs:CogenEnv',
            nondeterministic=False,
            max_episode_steps=self.timesteps_per_day)

        # actual ONNX model is loaded in reset()
        self._model: rt.InferenceSession | None = None

        # load model parameters from JSON file into DataFrame
        #     id  (index)      str
        #     min          float64
        #     max          float64
        #     unit             str
        #     data_type        str
        bytesio = read_to_bytesio('data/cogen/onnx_model/model.json')
        json_data = json.load(bytesio)
        bytesio.close()

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

    def _forecast_from_time(self, day: int, time_step: int) -> pd.DataFrame:
        """Returns the forecast values starting at the given day and time step
        for the following self.forecast_horizon + 1 time steps."""
        slice_df = self.ambients_dfs[day].iloc[time_step:min(time_step+self.forecast_horizon+1, self.timesteps_per_day)]
        # fix so that if the slice_df is not long enough, it will take the first values of the next day
        # TODO: figure out what to do if we're on the last day and there is no next day
        if len(slice_df) < self.forecast_horizon + 1:
            slice_df = pd.concat([slice_df, self.ambients_dfs[day+1].iloc[:self.forecast_horizon + 1 - len(slice_df)]])
        cols = ['Ambient Temperature', 'Ambient Pressure',
                'Ambient rel. Humidity', 'Target Net Power',
                'Target Process Steam', 'Energy Price', 'Gas Price']
        return slice_df[cols].astype(np.float32)

    def _get_obs(self) -> dict[str, Any]:
        """Get the current observation.

        The following values must be updated before calling `self._get_obs()`:
        - self.t
        - self.current_day
        - self.current_action
        """
        forecast_df = self._forecast_from_time(self.current_day, self.t)
        obs = {
            'Time': np.array([self.t / self.timesteps_per_day], dtype=np.float32),
            'Prev_Action': self.current_action,
            'TAMB': forecast_df['Ambient Temperature'].values,
            'PAMB': forecast_df['Ambient Pressure'].values,
            'RHAMB': forecast_df['Ambient rel. Humidity'].values,
            'Target_Power': forecast_df['Target Net Power'].values,
            'Target_Steam': forecast_df['Target Process Steam'].values,
            'Energy_Price': forecast_df['Energy Price'].values,
            'Gas_Price': forecast_df['Gas Price'].values,
        }
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an episode.

        Args:
            seed: optional seed vaue for controlling seed of np.random attributes
            return_info: determines if returned observation includes additional
                info or not (not implemented)
            options: includes optional settings like reward type (not implemented)

        Returns:
            obs: initial state
            info: initial info dict
        """
        super().reset(seed=seed)

        # We initialize ONNX model in reset() instead of __init__() because
        # RLLib inits the environment first, then forks worker processes.
        # However, an ONNX InferenceSession cannot be "pickled" and therefore
        # cannot be forked across RLLib worker processes.
        if self._model is None:
            b = read_bytes('data/cogen/onnx_model/model.onnx')
            self._model = rt.InferenceSession(
                b, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # randomly pick a day for the episode
        # subtract 1 as temporary fix to make sure we don't go over the number of days with lookahead window
        if seed is None:
            self.current_day = self.np_random.integers(low=0, high=self.n_days-1)
        else:
            self.current_day = seed % self.n_days

        self.t = 0  # keeps track of which timestep we are on

        # initial action is drawn randomly from the action space
        # not sure if this is reasonable, TODO: check this
        self.action_space.seed(seed)
        self.current_action = self.action_space.sample()

        self.obs = self._get_obs()
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

    def _compute_reward(self, obs: dict[str, Any], action: dict[str, Any]
                        ) -> tuple[float, dict[str, Any]]:
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

        assert self._model is not None
        model_output = self._model.run(None, {self._model.get_inputs()[0].name: [model_input]})[0][0]
        # print(model_output)
        # extract the fuel consumption (klb/hr)

        # fuel costs
        fuel_costs = {
            'GT1': model_output[6],
            'GT2': model_output[7],
            'GT3': model_output[8],
            'ST' : 0,
        }
        total_fuel_cost = model_output[-8]

        # ramp costs
        prev_action = obs['Prev_Action']
        ramp_costs = {
            'GT1': self.ramp_penalty * np.abs(action['GT1_PWR'][0] - prev_action['GT1_PWR'][0]),
            'GT2': self.ramp_penalty * np.abs(action['GT2_PWR'][0] - prev_action['GT2_PWR'][0]),
            'GT3': self.ramp_penalty * np.abs(action['GT3_PWR'][0] - prev_action['GT3_PWR'][0]),
            'ST' : self.ramp_penalty * np.abs(action['ST_PWR'][0] - prev_action['ST_PWR'][0]),
        }
        total_ramp_cost = sum(ramp_costs.values())

        # dynamic operating constraint violation
        dyn_cv = self._dyn_constraint_volation(model_input, model_output)
        dyn_cv_costs = {
            'GT1': self.constraint_violation_penalty * dyn_cv[:4].sum(),
            'GT2': self.constraint_violation_penalty * dyn_cv[4:8].sum(),
            'GT3': self.constraint_violation_penalty * dyn_cv[8:12].sum(),
            'ST' : self.constraint_violation_penalty * dyn_cv[12:].sum(),
        }
        total_dyn_cv_cost = sum(dyn_cv_costs.values())

        # compute the penalty for steam/energy non-delivery
        # this will be a relu penalty, so if the target is not met, the penalty is the difference
        # between the target and the actual value
        steam_penalty = np.maximum(0, obs['Target_Steam'][0] - model_output[-1])
        energy_penalty = np.maximum(0, obs['Target_Power'][0] - model_output[-2])
        non_delivery_penalty = self.supply_imbalance_penalty * (steam_penalty + energy_penalty)

        # compute the total reward
        total_reward = -(total_fuel_cost + total_ramp_cost + non_delivery_penalty + total_dyn_cv_cost)
        reward_breakdown = {
            'fuel_costs': fuel_costs,
            'ramp_costs': ramp_costs,
            'dyn_cv_costs': dyn_cv_costs,
            'non_delivery_cost': non_delivery_penalty
        }
        return total_reward, reward_breakdown

    def step(self, action: dict[str, Any]
             ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action: an action provided by the environment

        Returns:
            obs: new state
            reward: reward
            terminated: termination flag
            truncated: always ``False``, since there is no intermediate stopping condition
            info: info dict
        """
        # compute the loss of taking the action
        self.current_reward, self.current_info = self._compute_reward(self.obs, action)

        # update the current action
        self.current_action = action

        # update the current timestep
        self.t += 1

        # update the current observation
        self.obs = self._get_obs()

        # update the current done
        terminated = (self.t >= self.timesteps_per_day)

        # always False due to no intermediate stopping conditions
        truncated = False

        return self.obs, self.current_reward, terminated, truncated, self.current_info
