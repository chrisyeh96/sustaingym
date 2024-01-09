from env import BuildingEnv
from utils import ParameterGenerator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

block_size = 120 # always in hours
num_periods = 365 # number of episodes in one year of data
time_res = 300 # always in seconds
summer_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson",
                                   stochastic_seasonal_ambient_features="summer",
                                   stochasic_generator_block_size=block_size,
                                   num_periods=num_periods,
                                   time_res=time_res)
winter_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson",
                                   stochastic_seasonal_ambient_features="winter",
                                   stochasic_generator_block_size=block_size,
                                   num_periods=num_periods,
                                   time_res=time_res)
neutral_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson",
                                   num_periods=num_periods,
                                   time_res=time_res)

summer_env = BuildingEnv(summer_params)
winter_env = BuildingEnv(winter_params)
neutral_env = BuildingEnv(neutral_params)

summer_obs = []
winter_obs = []
neutral_obs = []

num_running_periods = 1 # num_periods
seed_start = 180

print("Collecting weather data...")
for i in tqdm(range(num_running_periods)):
    done = False
    this_summer_obs, _ = summer_env.reset(seed=seed_start)
    this_winter_obs, _ = winter_env.reset(seed=seed_start)
    this_neutral_obs, _ = neutral_env.reset(seed=seed_start)
    if i == 0:
        summer_obs = this_summer_obs.copy()
        winter_obs = this_winter_obs.copy()
        neutral_obs = this_neutral_obs.copy()
    while not done:
        this_summer_action = summer_env.action_space.sample()
        this_winter_action = winter_env.action_space.sample()
        this_neutral_action = neutral_env.action_space.sample()

        this_summer_obs, _, done, _, _ = summer_env.step(this_summer_action)
        this_winter_obs, _, done, _, _ = winter_env.step(this_winter_action)
        this_neutral_obs, _, done, _, _ = neutral_env.step(this_neutral_action)

        summer_obs = np.vstack((summer_obs, this_summer_obs))
        winter_obs = np.vstack((winter_obs, this_winter_obs))
        neutral_obs = np.vstack((neutral_obs, this_neutral_obs))

ewm_periods = num_periods
feature_mappings = {-2: "heat gain from irradiance",
                    -3: "ground temperature",
                    -4: "outdoor temperature"}
ambient_feature_indices = feature_mappings.keys()

for ft_idx in ambient_feature_indices:
    summer_series = pd.Series(summer_obs[:, ft_idx])
    winter_series = pd.Series(winter_obs[:, ft_idx])
    neutral_series = pd.Series(neutral_obs[:, ft_idx])

    summer_ewma = summer_series.ewm(ewm_periods).mean()[10:]
    winter_ewma = winter_series.ewm(ewm_periods).mean()[10:]
    neutral_ewma = neutral_series.ewm(ewm_periods).mean()[10:]

    plt.figure()
    plt.title(f"Summer vs. winter for {feature_mappings[ft_idx]}")
    plt.plot(summer_ewma, label=f"Summer EWMA({ewm_periods})", color="red", linewidth=0.75)
    plt.plot(winter_ewma, label=f"Winter EWMA({ewm_periods})", color="blue", linewidth=0.75)
    plt.plot(neutral_ewma, label=f"Normal-year EWMA({ewm_periods})", color="gray", linewidth=0.75)
    plt.xlabel("Time")
    plt.ylabel("Feature value")
    plt.legend()
    plt.show()