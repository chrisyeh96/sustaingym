from env import BuildingEnv
from utils import ParameterGenerator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

block_size = 288
summer_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson",
                                   stochastic_seasonal_ambient_features="summer",
                                   stochasic_generator_block_size=block_size)
winter_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson",
                                   stochastic_seasonal_ambient_features="winter",
                                   stochasic_generator_block_size=block_size)
neutral_params = ParameterGenerator(building="OfficeSmall", 
                                   weather="Hot_Dry",
                                   location="Tucson")

summer_env = BuildingEnv(summer_params)
winter_env = BuildingEnv(winter_params)
neutral_env = BuildingEnv(neutral_params)

this_summer_obs, _ = summer_env.reset()
this_winter_obs, _ = winter_env.reset()
this_neutral_obs, _ = neutral_env.reset()

summer_obs = [this_summer_obs]
winter_obs = [this_winter_obs]
neutral_obs = [this_neutral_obs]

terminated = False
while not terminated:
    action = summer_env.action_space.sample()
    this_obs, _, terminated, _, _ = summer_env.step(action)
    summer_obs = np.vstack((summer_obs, this_obs))

terminated = False
while not terminated:
    action = winter_env.action_space.sample()
    this_obs, _, terminated, _, _ = winter_env.step(action)
    winter_obs = np.vstack((winter_obs, this_obs))

terminated = False
while not terminated:
    action = neutral_env.action_space.sample()
    this_obs, _, terminated, _, _ = neutral_env.step(action)
    neutral_obs = np.vstack((neutral_obs, this_obs))

ewm_periods = 100
feature_mappings = {-2: "heat gain from irradiance",
                    -3: "ground temperature",
                    -4: "outdoor temperature"}
ambient_feature_indices = feature_mappings.keys()

for ft_idx in ambient_feature_indices:
    summer_series = pd.Series(summer_obs[:, ft_idx])
    winter_series = pd.Series(winter_obs[:, ft_idx])
    neutral_series = pd.Series(neutral_obs[:, ft_idx])

    summer_ewma = summer_series.ewm(ewm_periods).mean()
    winter_ewma = winter_series.ewm(ewm_periods).mean()
    neutral_ewma = neutral_series.ewm(ewm_periods).mean()

    plt.figure()
    plt.title(f"Summer vs. winter for {feature_mappings[ft_idx]}")
    plt.plot(summer_ewma, label=f"Summer EWMA", color="red", linewidth=0.75)
    plt.plot(winter_ewma, label=f"Winter EWMA", color="blue", linewidth=0.75)
    plt.plot(neutral_ewma, label=f"Normal-year EWMA", color="gray", linewidth=0.75)
    plt.xlabel("Time")
    plt.ylabel("Feature value")
    plt.legend()
    plt.show()