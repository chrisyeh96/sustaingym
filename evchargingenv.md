---
layout: page
title: EVChargingEnv
permalink: /evchargingenv/
nav_order: 2
use_math: true
---

# EVChargingEnv

EVChargingEnv uses [ACNSim](https://ev.caltech.edu/simulator) to simulate the charging of electric vehicles (EVs) based on actual data gathered from EV charging networks between fall 2018 and summer 2021. ACNSim is a "digital twin" of actual EV charging networks at Caltech and JPL, which have $n=54$ and $52$ charging stations (abbrv. EVSEs, Electric Vehicle Supply Equipment), respectively. ACNSim accounts for nonlinear EV battery charging dynamics and unbalanced 3-phase AC power flows, and is thus very realistic. ACNSim (and therefore EVChargingEnv) can be extended to model other charging networks as well. When drivers charge their EVs, they provide an estimated time of departure and amount of energy requested. Because of network and power constraints, not all EVSEs can simultaneously provide their maximum charging rates (a.k.a. "pilot signals").

Each episode starts at midnight and runs at 5-minute time steps for 24 hours ($T = 288$, $\tau = 5/60$ hours). At each time step, the agent simultaneously decides all $n$ EVSE pilot signals to be executed for the duration of that time step. Its objective is to maximize charge delivery while minimizing carbon costs and obeying the network and power constraints. At every time step, the system must decide the pilot signals for each EVSE to be executed for the duration of that time step. In the single-agent setting, the single agent simultaneously controls all $n$ EVSEs. In the multi-agent setting, each agent decides the charging rate for a single EVSE.

EVChargingEnv supports real historical data as well as data sampled from a 30-component Gaussian Mixture Model (GMM) fit to historical data.

## Observation Space
An observation at time $t$ is $s(t) = (t, d, e, m_{t-1}, \hat{m}_{t:t+k-1|t})$. $t \in \Z_+$ is the fraction of day between 0 and 1, inclusive. $d \in \Z^n$ is estimated remaining duration of each EV (in \# of time steps). $e \in \R_+^n$ is remaining energy demand of each EV (in kWh). If no EV is charging at EVSE $i$, then $d_i = 0$ and $e_i = 0$. If an EV charging at EVSE $i$ has exceeded the user-specified estimated departure time, then $d_i$ becomes negative, while $e_i$ may still be nonzero.

## Action Space
EVChargingEnv exposes a choice of discrete actions $a(t) \in \{0,1,2,3,4\}^n$, representing pilot signals scaled down by a factor of 8, or continuous actions $a(t) \in [0, 1]^n$ representing the pilot signal normalized by the maximum signal allowed $M$ (in amps) for each EVSE. Physical infrastructure in a charging network constrains the set $\mathcal{A}_t$ of feasible actions at each time step $t$. Furthermore, the EVSEs only support discrete pilot signals, so $\mathcal{A}_t$ is nonconvex. To satisfy these physical constraints, EVChargingEnv can project an agent's action $a(t)$ into the convex hull of $\mathcal{A}_t$ and round it to the nearest allowed pilot signal, resulting in final normalized pilot signals $\tilde{a}(t)$. ACNSim processes $\tilde{a}(t)$ and returns the actual charging rate $M \bar{a} \in \R_+^n$ (in amps) delivered at each EVSE, as well as the remaining demand $e_i(t+1)$.

## Reward Function
The reward function is a sum of three components: $r(t) = p(t) - c_V(t) - c_C(t)$. The profit term $p(t)$ aims to maximize energy delivered to the EVs. The constraint violation cost $c_V(t)$ penalizes network and power constraint violations. Finally, the CO<sub>2</sub> emissions cost $c_C(t)$, which is a function of the MOER $m_t$ and charging action, aims to reduce emissions by encouraging the agent to charge EVs when the MOER is low.

## Getting Started

To train a single-agent PPO (RLLib) with a learning rate of 5e-4 using the GMM trained on Summer 2021 data from Caltech:

```bash
python -m examples.evcharging.train_rllib -a ppo -t "Summer 2021" -s caltech -r 123 --lr 5e-4
```

More generally, our training script takes the following arguments:

```
usage: train_rllib.py [-h] [-a {ppo,sac}] [-t {Summer 2019,Fall 2019,Spring 2020,Summer 2021}]
                      [-s {caltech,jpl}] [-d] [-m] [-p PERIODS_DELAY] [-r SEED] [-l LR]

train RLLib models on EVChargingEnv

optional arguments:
  -h, --help            show this help message and exit
  -a {ppo,sac}, --algo {ppo,sac}
                        RL algorithm (default: ppo)
  -t {Summer 2019,Fall 2019,Spring 2020,Summer 2021}, --train_date_period {Summer 2019,Fall 2019,Spring 2020,Summer 2021}
                        Season. (default: Summer 2021)
  -s {caltech,jpl}, --site {caltech,jpl}
                        site of garage. caltech or jpl (default: caltech)
  -d, --discrete
  -m, --multiagent
  -p PERIODS_DELAY, --periods_delay PERIODS_DELAY
                        communication delay in multiagent setting. Ignored for single agent. (default: 0)
  -r SEED, --seed SEED  Random seed (default: 123)
  -l LR, --lr LR        Learning rate (default: 5e-05)
```



### Custom RL Loop

```python
from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator

# Create events generator which samples events from a GMM trained on Caltech
# data. The 'jpl' site is also supported, along with the periods
# 'Fall 2019', 'Spring 2020' , and 'Summer 2021'.
gmmg = GMMsTraceGenerator('caltech', 'Summer 2019')

# Create environment
env = EVChargingEnv(gmmg)

obs, episode_info = env.reset(seed=123, return_info=True)
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```