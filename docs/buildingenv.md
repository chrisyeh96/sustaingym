# BuildingEnv

BuildingEnv considers the control of the heat flow in a multi-zone building so as to maintain a desired temperature setpoint.  Building temperature simulation uses first-principled physics models. Users can either choose from a pre-defined list of buildings (Office small, School primary, Apartment midrise, and Office large) and three climate types and cities (San Diego, Tucson, New York) provided by the Building Energy Codes Program or define a customized BuildingEnv environment by importing any self-defined EnergyPlus building models. Each episode runs for 1 day, with 5-minute time intervals ($H = 288$, $\tau = 5/60$ hours).

## Observation Space
For a building with $M$ indoor zones, the state $s(t) \in \R^{M+4}$ contains observable properties of the building environment at timestep $t$:

$$
    s(t) = (T_1(t), \dotsc, T_M(t), T_\mathrm{E}(t), T_\mathrm{G}(t), Q^\mathrm{GHI}(t), \bar{Q}^\mathrm{p}(t)),
$$

where $T_i(t)$ is zone $i$'s temperature at time step $t$, $\bar{Q}^\mathrm{p}(t)$ is the heat acquisition from occupant's activities, $Q^\mathrm{GHI}(t)$ is the heat gain from the solar irradiance, and $T_\mathrm{G}(t)$ and $T_\mathrm{E}(t)$ denote the ground and outdoor environment temperature. In practice, the agent may have access to all or part of the state variables for decision-making depending on the sensor setup. Note that the outdoor/ground temperature, room occupancy, and heat gain from solar radiance are time-varying uncontrolled variables from the environment.

## Action Space
The action $a(t) \in [-1, 1]^M$ sets the controlled heating supplied to each of the $M$ zones, scaled to $[-1, 1]$.

## Reward Function
The objective is to reduce energy consumption while keeping the temperature within a given comfort range. The default reward function is a weighted $\ell_2$ reward, defined as

$$
    r(t) = - (1-\beta) \|a(t)\|_2 - \beta \|T^\mathrm{target}(t)-T(t)\|_2
$$

where $T^\mathrm{target}(t)=[T^\mathrm{target}_{1}(t),\cdots,T^\mathrm{target}_{M}(t)]$ are the target temperatures and $T(t)=[T_1(t),\cdots,T_M(t)]$ are the actual zonal temperature. BuildingEnv also allows users to customize reward functions by changing the weight term $\beta$ or the parameter $p$ defining the $\ell_p$ norm. Users can also customize the reward function to take CO<sub>2</sub> emissions into consideration.

## Distribution Shift
BuildingEnv features distribution shifts in the ambient outdoor temperature profile $T_\mathrm{E}$ which varies with different seasons.

## Multiagent Setting
In the multiagent setting for BuildingEnv, we treat each building as an independent agent whose action is the building's heat control decisions. It must coordinate with other building agents to maximize overall reward, which is the summation of each agent's reward. Each agent obtains either the global observation or individual building states.

## Getting Started

### Installation

SustainGym is designed for Linux machines. SustainGym is hosted on [PyPI](https://pypi.org/project/sustaingym/) and can be installed with `pip`:

```bash
pip install sustaingym[building]
```

### Using our training script

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda-other-installer-links.html).
2. (Optional, but recommended) If you are using a conda version `<=23.9.0`, set the conda solver to libmamba for faster dependency solving. Starting from conda version [`23.10.0`](https://github.com/conda/conda/releases/tag/23.10.0), libmamba is the default solver.
    ```bash
    conda config --set solver libmamba
    ```
3. Install the libraries necessary for runnning the BuildingEnv environment.
    ```bash
    conda env update --file env_building.yml --prune
    ```

More instructions coming soon!

### Custom RL Loop

Coming soon!
