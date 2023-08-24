---
layout: page
title: ElectricityMarketEnv
permalink: /electricitymarketenv/
nav_order: 3
use_math: true
---

# ElectricityMarketEnv

ElectricityMarketEnv simulates a realtime electricity market with 5-minute settlements. The default environment consists of 33 dispatchable generators and 1 80 MWh battery storage system connected on a 24-bus congested transmission network based on the IEEE Reliability Test System (IEEE RTS-24) [[1]](#references), with load data from the IEEE Reliability Test System of the Grid Modernization Laboratory Consortium (IEEE RTS-GMLC) [[2]](#references). Every time step (representing a 5 minute lapse of time), all participants submit bids to the market operator (MO). Based on the bids, the MO solves the multi-timestep security-constrained economic dispatch (SCED) problem which determines the price and amount of electricity purchased from (or sold by) each generator and battery to meet realtime electricity demand. Each episode runs for 1 day, with 5-minute time intervals ($$T = 288$$, $$\tau = 5/60$$ hours). The agent controls the battery system and is rewarded for submitting bids that result in charging (buy) when prices are low, and discharging (sell) when prices and CO<sub>2</sub> emissions are high, thus performing price arbitrage.

While ElectricityMarketEnv is a simplification of real-world electricity markets, the parameters of the environment's generators and battery systems and the historical electricity demand, marginal operating emissions rate (MOER) values, and associated forecasts are obtained from actual systems within California. In this regard, ElectricityMarketEnv serves as a minimum test-bed for RL algorithms, as a real-world electricity market is significantly more complex (_e.g._, featuring complex generator cost models).

## Observation Space

An observation is $$s(t) = (t, e, a(t-1), x_{t-1}, p_{t-1}, l_{t-1}, \hat{l}_{t:t+k-1}, m_{t-1}, \hat{m}_{t:t+k-1|t})$$. $$t \in \{0, 1, \dotsc, 288\}$$ is the current time step. $$e \in \R_+$$ is the agent's battery level (in MWh). $$a(t-1) \in \R_+^{2 \times k}$$ is the previous action. $$x_{t-1} \in \R$$ is the previous dispatch (in MWh) asked of the agent, and $$p_{t-1} \in \R$$ is market clearing price from the previous step (in $/MWh). $$l_{t-1} \in \R$$  is the previous demand experienced by the agent (in MWh), while $$\hat{l}_{t:t+k|t} \in \R^k$$ is the forecasted demand for the next $$k$$ steps. Likewise, $$m_{t-1} \in \R$$ is the previous MOER experienced by the agent (in kg CO<sub>2</sub> / MWh), while $$\hat{l}_{t:t+k|t} \in \R^k$$ is the forecasted MOER values for the next $$k$$ steps.

## Action Space
Each agent action is a bid $$a(t) = (a^c, a^d) \in \R_+^k \times \R_+^k$$, representing prices (in $/MWh) that the agent is willing to pay (or receive) for charging (or discharging) per MWh of energy, for the next $$k+1$$ time steps starting at time step $$t$$. The generators are assumed to always bid their fixed true cost of generation. The environment solves the optimal dispatch problem to determine the electricity price $$p_t$$ (in $/MWh) and the agent's dispatch $$x_t \in \R$$, which is the amount of energy (in MWh) that the agent is obligated to sell into or buy from the grid within the next time step. The dispatch in turn determines the storage system's next energy level. We also provide a wrapper that discretizes the action space into 3 actions only: charge, do nothing, or discharge.

## Reward Function
The reward function encourages the agent to maximize profit from charging decisions while minimizing associated carbon emission. The reward function is a sum of three components:

$$
r(t) = r_R(t) + r_C(t) - c_T(t).
$$

The revenue term $$r_R(t) = p_t x_t$$ is the immediate revenue from the dispatch. The CO<sub>2</sub> emissions reward term $$r_C(t) = P_\text{CO2} m_t x_t$$ represents the price of CO<sub>2</sub> emissions displaced or incurred by the battery dispatch. The terminal cost $$c_T(t)$$, which is nonzero only when $$t=T$$, encourages the battery to have the same energy level at the end of the day as when it started. We also provide an option to delay all reward signals until the terminal time step (intermediate rewards are set to 0).

## Distribution Shift
ElectricityMarketEnv considers temporal distribution shifts, specifically in the time of year demand and MOER values are drawn from.

## Getting Started

### Installation

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda-other-installer-links.html).
2. (Optional) Set the conda solver to libmamba for faster dependency solving.
    ```bash
    conda config --set solver libmamba
    ```
3. Download SustainGym from GitHub. (In the future, we plan on releasing SustainGym via [pypi](https://pypi.org/).)
    ```bash
    git clone --depth 1 https://github.com/chrisyeh96/sustaingym.git
    ```
4. Install the libraries necessary for runnning the ElectricityMarketEnv environment.
    ```bash
    conda env update --file env_em.yml --prune
    ```

### Using our training script

To train a central, single agent Soft Actor-Critic (SAC) with a learning rate of 0.0001 and demand and MOER data from July 2020:

```bash
python -m examples.electricitymarket.train_rllib -m 7 -a sac -l 0.0001 -o examples/train_logs
```

The resulting trained model and its metrics will be stored in the subdirectory "examples/train_logs" according to the flag `-o`.

More generally, our training script takes the following arguments:

```
usage: python -m examples.electricitymarket.train_rllib -m MONTH [-v EVAL_MONTH] [-d] [-i] -a ALGORITHM -l LR [-g GAMMA] [-e EVAL_FREQ] [-n EVAL_EPISODES] [-o LOG_DIR]

train RLLib models on ElectricityMarketEnv

options:
  -m MONTH, --month MONTH  month of environment data for training (default: None)
  -v EVAL_MONTH, --eval-month EVAL_MONTH month of environment data for out of
              distribution evaluation (default: None)
  -d, --discrete        whether to use discretized actions (default: False)
  -i, --intermediate-rewards
                        whether to use intermediate rewards (default: False)
  -a ALGORITHM, --algo ALGORITHM
                        type of model. dqn, sac, ppo, a2c, or ddpg (default: None)
  -l LR, --lr LR        learning rate (default: None)
  -g GAMMA, --gamma GAMMA
                        discount factor, between 0 and 1 (default: 0.9999)
  -e EVAL_FREQ, --eval-freq EVAL_FREQ
                        # of episodes between eval/saving model during training (default: 20)
  -n EVAL_EPISODES, --eval-episodes EVAL_EPISODES
                        # of episodes algorithm evaluated on during training (default: 5)
  -o LOG_DIR, --log-dir LOG_DIR
                        directory for saving logs and models (default: .)
```


### Custom RL Loop

```python
from sustaingym.envs.electricitymarket import ElectricityMarketEnv

# utilize July 2020 load and MOER values and only report cumulative terminal rewards
env = ElectricityMarketEnv(month='2021-05', use_intermediate_rewards=False)

obs = env.reset(seed=123)
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```


## References

[1] P. M. Subcommittee, "IEEE Reliability Test System," in _IEEE Transactions on Power Apparatus and Systems_, vol. PAS-98, no. 6, pp. 2047-2054, Nov. 1979, doi: 10.1109/TPAS.1979.319398.

[2] C. Barrows _et al._, "The IEEE Reliability Test System: A Proposed 2019 Update," in _IEEE Transactions on Power Systems_, vol. 35, no. 1, pp. 119-127, Jan. 2020, doi: 10.1109/TPWRS.2019.2925557.
