---
layout: page
title: BuildingEnv
permalink: /buildingenv/
nav_order: 6
use_math: true
---

BuildingEnv considers the control of the heat flow in a multi-zone building so as to maintain a desired temperature setpoint.  Building temperature simulation uses first-principled physics models. Users can either choose from a pre-defined list of buildings (Office small, School primary, Apartment midrise, and Office large) and three climate types and cities (San Diego, Tucson, New York) provided by the Building Energy Codes Program or define a customized BuildingEnv environment by importing any self-defined EnergyPlus building models. Each episode runs for 1 day, with 5-minute time intervals ($$H = 288$$, $$\tau = 5/60$$ hours).

## Observation Space
For a building with $$M$$ indoor zones, the state $$s(t) \in \R^{3M+2}$$ contains observable properties of the building environment at timestep $$t$$:

$$
s(t) = (T_1(t), ...,T_{M}(t), N_1(t), ..., N_{M}(t), Q_{1}^{GHI}(t), ..., Q_{M}^{GHI}(t), T_G(t), T_{E}(t)),
$$

where $$T_i(t)$$ is zone $$i$$'s temperature at time step $$t$$, $$N_i(t)$$ is the number of occupants, $$Q_{i}^{GHI}(t)$$ is the heat gain from the solar irradiance, and $$T_G(t)$$ and $$T_E(t)$$ denote the ground and outdoor environment temperature. In practice, the agent may have access to all or part of the state variables for decision-making depending on the sensor setup. Note that the outdoor/ground temperature, room occupancy, and heat gain from solar radiance are time-varying uncontrolled variables from the environment.

## Action Space
The action $$a(t) \in [-1, 1]^M$$ sets the controlled heating supplied to each of the $$M$$ zones, scaled to $$[-1, 1]$$.

## Reward Function
The objective is to reduce energy consumption while keeping the temperature within a given comfort range. The default reward function is a weighted $$\ell_2$$ reward, defined as

$$
    r(t) = - (1-\beta) \|a(t)\|_2 - \beta \|T^{obj}(t)-T(t)\|_2
$$

where $$T^{obj}(t)=[T^{obj}_{1}(t),...,T^{obj}_{M}(t)]$$ are the target temperatures and $$T(t)=[T_{1}(t),...,T_{M}(t)]$$ are the actual zonal temperature. BuildingEnv also allows users to customize reward functions using environment states $$s(t)$$, actions $$a(t)$$, target values $$T^{obj}(t)$$, and a weight term $$\beta$$. Users can also customize the reward function to take CO<sub>2</sub> emissions into consideration.

## Distribution Shift
BuildingEnv features distribution shifts in the ambient outdoor temperature profile $$T_E$$ which varies with different seasons.

## Multiagent Setting
In the multiagent setting for BuildingEnv, we treat each building as an independent agent whose action is the building's heat control decisions. It must coordinate with other building agents to maximize overall reward, which is the summation of each agent's reward. Each agent obtains either the global observation or individual building states.

## Getting Started

### Installation

Coming soon!

### Using our training script

Coming soon!

### Custom RL Loop

Coming soon!
