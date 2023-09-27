from __future__ import annotations
import sys
sys.path.append("../..")
from sustaingym.envs.building.building import BuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator
from sustaingym.envs.building.mpc_controller import MPCAgent
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO,A2C,SAC

Parameter=ParameterGenerator(
    'OfficeSmall','Hot_Dry','Tucson',time_reso=300)  #Description of ParameterGenerator in bldg_utils.py
#Create environment
env = BuildingEnv(Parameter)
numofhours=24
#Initialize
env.reset()

############  SAC  ############

rwlistsac=[]
sacstr=[]
model = SAC("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = SAC.load("plot/SAC_winter0")
obs = vec_env.reset()

#Winter Test
#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)
#Then loop a month's testing data
for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  sacstr.append('Winter')
  rwlistsac.append(rw/288)

#Summer Test
env.reset()
model = SAC("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = SAC.load("plot/SAC_summer0")
obs = vec_env.reset()

#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  sacstr.append('Summer')
  rwlistsac.append(rw/288)

############  PPO  ############
rwlistppo=[]
model = PPO("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = PPO.load("plot/PPO_winter0")
obs = vec_env.reset()

#Winter Test
#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)
#Then loop a month's testing data
for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlistppo.append(rw/288)

#Summer Test
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = PPO.load("plot/PPO_summer0")
obs = vec_env.reset()

#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlistppo.append(rw/288)

############  A2C  ############
rwlista2c=[]
model = A2C("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = A2C.load("plot/A2C_winter0")
obs = vec_env.reset()

#Winter Test
#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)
#Then loop a month's testing data
for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlista2c.append(rw/288)

#Summer Test
env.reset()
model = A2C("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = A2C.load("plot/A2C_summer0")
obs = vec_env.reset()

#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlista2c.append(rw/288)

############  MPC  ############
env.reset()
rwlistmpc=[]
mpcstr=[]
agent = MPCAgent(env,
                gamma=env.gamma,
                safety_margin=0.96, planning_steps=10)

for k in range(2048):
  action, _states = agent.predict(env)
  obs, rewards, terminated, truncated, info = env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = agent.predict(env)
    obs, rewards, terminated, truncated, info = env.step(action)
    rw+=rewards
  rwlistmpc.append(rw/288)
  mpcstr.append('Winter')

############  RANDOM  ############
env.reset()
rwlistrandom=[]
randomstr=[]

for k in range(2048):
  a = env.action_space.sample()#Randomly select an action
  obs, r, terminated, truncated, _ = env.step(a)#Return observation and reward

for i in range(30):
  rw=0
  for j in range(288):
    a = env.action_space.sample()#Randomly select an action
    obs, r, terminated, truncated, _ = env.step(a)#Return observation and reward
    rw+=r
  rwlistrandom.append(rw/288)
  randomstr.append('Winter')

############# DC ##############
Parameter=ParameterGenerator('OfficeSmall','Hot_Dry','Tucson',time_reso=300,spacetype='discrete')  #Description of ParameterGenerator in bldg_utils.py
#Create environment
env = BuildingEnv(Parameter)
numofhours=24
#Initialize
env.reset()
for i in range(numofhours):
    a = env.action_space.sample()#Randomly select an action
    obs, r, terminated, truncated, _ = env.step(a)#Return observation and reward

############# PPOdc ##############
rwlistppo_dc=[]
model = PPO("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = PPO.load("plot/PPOdc_winter0")
obs = vec_env.reset()

#Winter Test
#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)
#Then loop a month's testing data
for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlistppo_dc.append(rw/288)

#Summer Test
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = PPO.load("plot/PPOdc_summer0")
obs = vec_env.reset()

#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlistppo_dc.append(rw/288)

################# A2Cdc ################
rwlista2c_dc=[]
model = A2C("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
# model = PPO.load("Model/PPO5")
model = A2C.load("plot/A2Cdc_winter0")
obs = vec_env.reset()

#Winter Test
#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)
#Then loop a month's testing data
for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlista2c_dc.append(rw/288)

#Summer Test
env.reset()
model = A2C("MlpPolicy", env, verbose=1)
vec_env = model.get_env()
model = A2C.load("plot/A2Cdc_summer0")
obs = vec_env.reset()

#First loop through the training data
for k in range(2048):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = vec_env.step(action)

for i in range(30):
  rw=0
  for j in range(288):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    rw+=rewards
  rwlista2c_dc.append(rw/288)

############### Ploting #############
sns.set(style="darkgrid")
a0 = np.array(rwlistsac).flatten()

a1 = np.array(rwlistppo).flatten()

a2 = np.array(rwlistppo_dc).flatten()

a3 = np.array(rwlista2c).flatten()

a4 = np.array(rwlista2c_dc).flatten()

a5 = np.array(rwlistmpc+rwlistmpc).flatten()

a6 = np.array(rwlistrandom+rwlistrandom).flatten()
df = pd.melt( pd.DataFrame( {"SAC":a0, "PPO":a1, "PPO discrete":a2,"A2C":a3,"A2C discrete":a4, "MPC":a5,"Random":a6}), var_name = 'Algorithm', value_name = 'Daily Return')

df=df.assign(Trained_On=(sacstr+sacstr+sacstr+sacstr+sacstr+mpcstr+mpcstr+randomstr+randomstr))
# fig = sns.violinplot(data = df, x = 'Algorithm', y = 'Daily Return', hue = 'Trained_On').get_figure()

# Set the figure size to be wider
plt.figure(figsize=(12, 6))

# Get the default seaborn color palette
default_palette = sns.color_palette()

# Manually specify colors for "Summer" and "Winter"
custom_palette = {
    'Summer': default_palette[0],  # Originally the color for Winter
    'Winter': default_palette[1],  # Originally the color for Summer
}

# Specify the desired order for 'Algorithm'
algorithm_order = ['MPC', 'Random', 'SAC', 'PPO', 'PPO discrete', 'A2C', 'A2C discrete']

# Create violinplot with custom hue and algorithm order
ax = sns.violinplot(data=df, x='Algorithm', y='Daily Return', hue='Trained_On',
                    palette=custom_palette, hue_order=['Summer', 'Winter'],
                    order=algorithm_order)

# Set the labels and title with larger fonts
plt.xlabel('Algorithm', fontsize=22)
plt.ylabel('Daily Return', fontsize=22)

# Increase the fontsize for tick labels on x and y axes
plt.xticks(fontsize=18)  # Increase the fontsize to 20 for x-axis tick labels
plt.yticks(fontsize=18)  # Increase the fontsize to 20 for y-axis tick labels

# Get the handles and labels from the current legend
handles, labels = ax.get_legend_handles_labels()

# Create a mapping between labels and handles (useful for reordering)
label_handle_map = {label: handle for label, handle in zip(labels, handles)}

# Reorder the labels and handles in the desired order
ordered_labels = ['Summer', 'Winter']
ordered_handles = [label_handle_map[label] for label in ordered_labels]

# Set the updated legend
plt.legend(ordered_handles, ordered_labels, title='trained on', fontsize='18', title_fontsize='20', loc='lower left')

# Add a vertical line at x=1.5 to separate 'Random' and 'SAC' using large dotted markers
x_value = 1.5
y_range = ax.get_ylim()  # Get current y-axis limits for the specific Axes object
plt.plot([x_value, x_value], y_range, linestyle=':', color='black', linewidth=3)
ax.set_ylim(y_range)

# Save and show the figure
plt.savefig('plot/BuildingViolin.png')
plt.show()