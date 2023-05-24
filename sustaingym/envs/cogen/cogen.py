"""
This module implements the CogenEnv class
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import onnxruntime as rt

from gym import Env, spaces
from typing import Literal, Any
from sustaingym.data.cogen import load_ambients

class CogenEnv(Env):
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
                   Box(forecast_length + 1), Box(forecast_length + 1),
                   Box(forecast_length + 1), Box(forecast_length + 1),
                   Box(forecast_length + 1), Box(forecast_length + 1),
                   Box(forecast_length + 1))
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
                renewables_magnitude: float = None, # TODO: implement renewables
                constraint_violation_penalty: float = 1000,
                forecast_length: int = 12,
                forecast_noise_std: float = 0.1,
                seed: int | None = None,
                LOCAL_FILE_PATH: str | None = None
                ):
        """
        Constructs the CogenEnv object
        """
        self.forecast_length = forecast_length
        # load the ambient conditions dataframes
        self.ambients_dfs = load_ambients.construct_df(renewables_magnitude=renewables_magnitude)
        self.n_days = len(self.ambients_dfs)
        self.timesteps_per_day = float(len(self.ambients_dfs[0]))
        assert (self.forecast_length >= 0 and self.forecast_length < self.timesteps_per_day - 1), "forecast_length must be between 0 and timesteps_per_day - 1"

        # load the onnx model and parameters
        self.model = rt.InferenceSession('sustaingym/sustaingym/data/cogen/onnx_model/model.onnx')
        # Load the JSON file
        with open('sustaingym/sustaingym/data/cogen/onnx_model/model.json', 'r') as f:
            json_data = json.load(f)
        # I/O labels
        input_labels = [json_data['inputs'][i]['id'] for i in range(len(json_data['inputs']))]
        output_labels = [json_data['outputs'][i]['id'] for i in range(len(json_data['outputs']))]

        # Upper and lower bounds on inputs
        lower_bound = [json_data['inputs'][i]['min'] for i in range(len(json_data['inputs']))]
        upper_bound = [json_data['inputs'][i]['max'] for i in range(len(json_data['inputs']))]

        # Other spec
        unit = [json_data['inputs'][i]['unit'] for i in range(len(json_data['inputs']))]
        data_type = [json_data['inputs'][i]['data_type'] for i in range(len(json_data['inputs']))]

        inputs_table = pd.DataFrame({'Label': input_labels, 'unit': unit, 'data type': data_type, 'min': lower_bound, 'max': upper_bound})
        inputs_table = inputs_table.set_index('Label')

        self.rng = np.random.default_rng(seed)

        # action space is power output, evaporative cooler switch, power augmentation switch, and equivalent
        # process steam flow for generators 1, 2, and 3, as well as steam turbine power output, steam flow
        # through condenser, and number of cooling bays employed
        self.action_space = spaces.Dict({
            'GT1_PWR': spaces.Box(low=inputs_table.loc['GT1_PWR', 'min'], high=inputs_table.loc['GT1_PWR', 'max'], shape=(1,), dtype=float),
            'GT1_PAC_FFU': spaces.Discrete(2),
            'GT1_EVC_FFU': spaces.Discrete(2),
            'HR1_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR1_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR1_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
            'GT2_PWR': spaces.Box(low=inputs_table.loc['GT2_PWR', 'min'], high=inputs_table.loc['GT2_PWR', 'max'], shape=(1,), dtype=float),
            'GT2_PAC_FFU': spaces.Discrete(2),
            'GT2_EVC_FFU': spaces.Discrete(2),
            'HR2_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR2_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR2_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
            'GT3_PWR': spaces.Box(low=inputs_table.loc['GT3_PWR', 'min'], high=inputs_table.loc['GT3_PWR', 'max'], shape=(1,), dtype=float),
            'GT3_PAC_FFU': spaces.Discrete(2),
            'GT3_EVC_FFU': spaces.Discrete(2),
            'HR3_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR3_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR3_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
            'ST_PWR': spaces.Box(low=inputs_table.loc['ST_PWR', 'min'], high=inputs_table.loc['ST_PWR', 'max'], shape=(1,), dtype=float),
            'IPPROC_M': spaces.Box(low=inputs_table.loc['IPPROC_M', 'min'], high=inputs_table.loc['IPPROC_M', 'max'], shape=(1,), dtype=float),
            'CT_NrBays': spaces.Discrete(12, start=1)
        })

        # define the observation space
        self.observation_space = spaces.Dict({
            'Time': spaces.Box(low=0, high=1, shape=(1,), dtype=float),
            'Prev_Action': spaces.Dict({
                'GT1_PWR': spaces.Box(low=inputs_table.loc['GT1_PWR', 'min'], high=inputs_table.loc['GT1_PWR', 'max'], shape=(1,), dtype=float),
                'GT1_PAC_FFU': spaces.Discrete(2),
                'GT1_EVC_FFU': spaces.Discrete(2),
                'HR1_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR1_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR1_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
                'GT2_PWR': spaces.Box(low=inputs_table.loc['GT2_PWR', 'min'], high=inputs_table.loc['GT2_PWR', 'max'], shape=(1,), dtype=float),
                'GT2_PAC_FFU': spaces.Discrete(2),
                'GT2_EVC_FFU': spaces.Discrete(2),
                'HR2_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR2_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR2_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
                'GT3_PWR': spaces.Box(low=inputs_table.loc['GT3_PWR', 'min'], high=inputs_table.loc['GT3_PWR', 'max'], shape=(1,), dtype=float),
                'GT3_PAC_FFU': spaces.Discrete(2),
                'GT3_EVC_FFU': spaces.Discrete(2),
                'HR3_HPIP_M_PROC': spaces.Box(low=inputs_table.loc['HR3_HPIP_M_PROC', 'min'], high=inputs_table.loc['HR3_HPIP_M_PROC', 'max'], shape=(1,), dtype=float),
                'ST_PWR': spaces.Box(low=inputs_table.loc['ST_PWR', 'min'], high=inputs_table.loc['ST_PWR', 'max'], shape=(1,), dtype=float),
                'IPPROC_M': spaces.Box(low=inputs_table.loc['IPPROC_M', 'min'], high=inputs_table.loc['IPPROC_M', 'max'], shape=(1,), dtype=float),
                'CT_NrBays': spaces.Discrete(12, start=1)
            }),
            'TAMB': spaces.Box(low=inputs_table.loc['TAMB', 'min'], high=inputs_table.loc['TAMB', 'max'], shape=(forecast_length+1,), dtype=float),
            'PAMB': spaces.Box(low=inputs_table.loc['PAMB', 'min'], high=inputs_table.loc['PAMB', 'max'], shape=(forecast_length+1,), dtype=float),
            'RHAMB': spaces.Box(low=inputs_table.loc['RHAMB', 'min'], high=inputs_table.loc['RHAMB', 'max'], shape=(forecast_length+1,), dtype=float),
            'Target_Power': spaces.Box(low=0., high=700., shape=(forecast_length+1,), dtype=float),
            'Target_Steam': spaces.Box(low=0., high=1300., shape=(forecast_length+1,), dtype=float),
            'Energy_Price': spaces.Box(low=0., high=1500., shape=(forecast_length+1,), dtype=float),
            'Gas_Price': spaces.Box(low=0., high=7., shape=(forecast_length+1,), dtype=float)
        })

        # define the timestep
        self.timestep = timestep

        # define the current timestep
        self.current_timestep = 0

        # define the current state
        self.current_state = None

        # define the current action
        self.current_action = None

        # define the current reward
        self.current_reward = None

        # define the current done
        self.current_done = None

        # define the current info
        self.current_info = None

    def _forecast_values_from_time(self, day: int, time_step: int) -> tuple[list[float], list[float], list[float],
                                                                            list[float], list[float], list[float],
                                                                            list[float]]:
        """Returns the forecast values starting at the given day and time step
        for the following self.forecast_length + 1 time steps."""
        slice = self.ambients_dfs[day].iloc[time_step:min(time_step+self.forecast_length+1, self.timesteps_per_day)]
        # fix so that if the slice is not long enough, it will take the first values of the next day
        if len(slice) < self.forecast_length + 1:
            slice = slice.append(self.ambients_dfs[day+1].iloc[:self.forecast_length + 1 - len(slice)])
        return (slice['Ambient Temperature'].to_list(), slice['Ambient Pressure'].to_list(), slice['Ambient rel. Humidity'].to_list(),
                slice['Target Net Power'].to_list(), slice['Target Process Steam'].to_list(), slice['Energy Price'].to_list(),
                slice['Gas Price'].to_list())
    
    def reset(self, seed: int | None = None, return_info: bool = False,
              options: dict | None = None) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
        """Initialize or restart an instance of an episode for the Cogen environment.

        Args:
            seed: optional seed vaue for controlling seed of np.random attributes
            return_info: determines if returned observation includes additional
                info or not (not implemented)
            options: includes optional settings like reward type (not implemented)
        
        Returns:
            tuple containing the initial observation for env's episode
        """
        self.rng = np.random.default_rng(seed=seed)

        # randomly pick a day for the episode
        self.current_day = self.rng.integers(low=0, high=self.n_days)

        self.init = True # no clue what this is for
        self.step = 0 # keeps track of which timestep we are on

        # initial action is drawn randomly from the action space
        # not sure if this is reasonable, TODO: check this
        self.current_action = self.action_space.sample()

        forecast_values = self._forecast_values_from_time(self.current_day, self.step)

        # set up initial observation
        self.obs = {
            'Time': self.step / self.timesteps_per_day,
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
        return self.obs if not return_info else (self.obs, info)
    
    def _compute_reward(self) -> float:
        """Computes the reward for the current timestep."""
        raise NotImplementedError
    
    def step(self, action: dict[str, float | int]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Run one timestep of the Cogen environment's dynamics.

        Args:
            action: an action provided by the environment
        
        Returns:
            tuple containing the next observation, reward, done flag, and info dict
        """
        # update the current action
        self.current_action = action

        # update the current timestep
        self.step += 1

        # update the current state
        self.current_state = self.obs

        # update the current reward
        self.current_reward = self._compute_reward()

        # update the current done
        self.current_done = self._done()

        # update the current info
        self.current_info = self._info()

        # update the current observation
        self.obs = self._next_obs()

        return self.obs, self.current_reward, self.current_done, self.current_info

    def render(self):
        raise NotImplementedError

    def close(self):
        return