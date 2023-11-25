# CogenEnv

CogenEnv simulates the operation of a combined cycle gas power plant tasked with meeting local steam and energy demand. Conventional dispatchable generators suffer decreased efficiency as a result of frequent ramping, posing a particular challenge as increasing penetrations of variable renewables necessitate larger and more frequent ramps to ensure supply-demand balance. Thus, optimal operation of cogeneration resources requires balancing the competing objectives of minimizing fuel use, anticipating future ramp needs, and ensuring delivery of sufficient energy and steam to the grid.

CogenEnv is modeled after a specific combined cycle gas power plant whose location and details are anonymized for security reasons. The model is provided by [Enexsa](https://www.enexsa.com/), in collaboration with [Beyond Limits](https://www.beyond.ai/). CogenEnv should be representative of the general task of controlling a thermal power plant to meet energy and steam production targets while minimizing fuel and ramping costs.


## Observation Space
An observation takes the form

$$
s(t) = (\tau, a(t-1), T_{t:t+k}, P_{t:t+k}, H_{t:t+k}, d^p_{t:t+k}, d^q_{t:t+k}, \pi^p_{t:t+k}, \pi^f_{t:t+k}),
$$

where $\tau = t/96$ is the time (normalized by number of 15 minute intervals in a day), $a(t-1)$ is the agent's previous action, $T_{t:t+k}, P_{t:t+k},$ and $H_{t:t+k}$ are current and $k$ forecast steps of temperature, pressure, and relative humidity, respectively, $d^p_{t:t+k}$ and $d^q_{t:t+k}$ are current and $k$ forecast steps of electricity and steam demand, respectively, and $\pi^p_{t:t+k}$ and $\pi^f_{t:t+k}$ are current and $k$ forecast steps of electricity and fuel price, respectively. 

## Action Space
The action space is a vector $a(t) \in \R^{15}$ specifying dispatch setpoints and other auxiliary variables for all turbines in the plant. Specifically, for each of three gas turbines, the agent specifies (a) a scalar turbine electricity output, (b) a scalar heat recovery steam flow, (c) a binary evaporative cooler switch setting, and (d) a binary power augmentation switch setting. In addition, for the steam turbine, the agent specifies (a) a scalar turbine electricity output, (b) a scalar steam flow through the plant condenser, and (c) an integer number of cooling tower bays employed.

## Reward Function
The reward function is comprised of three components:

$$
r(t) = -\left(r_f(a(t); T_t, P_t, H_t) + r_r(a(t); a(t-1)) + r_c(a(t); d^p_t, d^q_t)\right).
$$

The term $r_f(a(t); T_t, P_t, H_t)$ is the generator fuel consumption in response to dispatch $a(t)$. The term $r_r(a(t); a(t-1))$ is the ramp cost, captured via an $\ell_1$ norm penalty for any change in generator electricity dispatch between consecutive actions. The third term, $r_c(a(t); d^p_t, d^q_t)$, is a constraint violation penalty, penalizing any unmet electricity and steam demand, as well as any violation of the plant's dynamic operating constraints. The sum of these three components is negated to convert costs to rewards.

## Distribution Shift
CogenEnv considers distribution shifts in the renewable generation profiles, and specifically, increasing penetration of wind energy. This increased variable renewable energy on the grid necessitates more frequent ramping in order to meet electricity demand, and may pose a challenge for RL algorithms trained on electricity demand traces without such variability.

## Multiagent Setting
The multiagent setting treats each turbine unit (each of the three gas turbines and the steam turbine) as an individual agent whose action is the turbine's electricity dispatch decision and auxiliary variable settings. The negative reward of each agent is the sum of the corresponding turbine unit's fuel consumption, ramp cost, and dynamic operating constraint penalty, as well as a shared penalty for unmet electricity and steam demand that is split evenly across agents. All agents observe the global observation.

## Getting Started

### Installation

SustainGym is hosted on [PyPI](https://pypi.org/project/sustaingym/) and can be installed with `pip`:

```bash
pip install sustaingym[cogen]
```

### Custom RL Loop

```python
from sustaingym.envs.cogen import CogenEnv

rm = 300  # 300 MW renewables penetration
env = CogenEnv(renewables_magnitude=rm)

obs = env.reset(seed=123)
terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```


### Using our training script

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda-other-installer-links.html).
2. (Optional, but recommended) If you are using a conda version `<=23.9.0`, set the conda solver to libmamba for faster dependency solving. Starting from conda version [`23.10.0`](https://github.com/conda/conda/releases/tag/23.10.0), libmamba is the default solver.
    ```bash
    conda config --set solver libmamba
    ```
3. Install the libraries necessary for runnning the CogenEnv environment.
    ```bash
    conda env update --file env_cogen.yml --prune
    ```

To train a multi-agent PPO (RLLib) with a learning rate of 5e-4 with 300 MW of renewable wind penetration:

```bash
python -m examples.cogen.train_rllib -a ppo -m --seed 123 --lr 5e-4 --rm 300
```

More generally, our training script takes the following arguments:

```
usage: python -m examples.cogen.train_rllib [-h] [-a {ppo,rand}] [-m] [-r SEED] [-l LR] [-n RM]

train RLLib models on CogenEnv

options:
  -h, --help            show this help message and exit
  -a {ppo,rand}, --algo {ppo,rand}
                        Only "ppo" supports the mixed discrete/continuous action spacein CogenEnv
                        (default: ppo)
  -m, --multiagent
  -r SEED, --seed SEED  Random seed (default: 123)
  -l LR, --lr LR        Learning rate (default: 5e-05)
  -n RM, --rm RM        Renewables magnitude in MW (default: 0)
```
