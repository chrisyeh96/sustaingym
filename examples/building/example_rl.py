from __future__ import annotations
import sys
sys.path.append("../..")
from sustaingym.envs.building.building import BuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import set_random_seed


Parameter = ParameterGenerator(
    "OfficeSmall", "Hot_Dry", "Tucson"
)  # Description of ParameterGenerator in bldg_utils.py
# Create environment
env = BuildingEnv(Parameter)
numofhours = 24
seed = 25
env.reset()
set_random_seed(seed=seed)
model = PPO(MlpPolicy, env, verbose=1)
rewardlist = []

for i in range(300):
    model.learn(total_timesteps=1000)
    rw = 0
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(24):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        rw += rewards
    print(rw / 24)
    rewardlist.append(rw / 24)
print("################TRAINING is Done############")
model.save("PPO_quick")

model = PPO(MlpPolicy, env, verbose=1)
vec_env = model.get_env()
model = PPO.load("PPO_quick")
obs = vec_env.reset()
print("Initial observation", obs)

for i in range(24):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
plt.plot(np.array(env.statelist)[:, :7])
plt.title("Office Small Zonal Temperature")

plt.xlabel("hours")
plt.ylabel("Celsius")
plt.legend(
    ["South", "East", "North", "West", "Core", "Plenum", "Outside"], loc="lower right"
)
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist)), 1))
plt.title("Office Small Power Consumption")
plt.xlabel("hours")
plt.ylabel("Watts")
plt.show()

plt.title("Quick PPO training")
plt.plot(rewardlist)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
