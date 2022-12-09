---
layout: page
title: EletricityMarketEnv
permalink: /electricitymarketenv/
nav_order: 3
use_math: true
---

ElectricityMarketEnv simulates a realtime electricity market with 5-minute settlements. The default environment consists of 10 dispatchable generators and 5 battery storage systems, all of which submit bids to the market operator (MO) at every time step. Based on the bids, the MO solves the economic dispatch problem which determines the price and amount of electricity purchased from (or sold by) each generator and battery to meet realtime electricity demand. Each episode runs for 1 day, with 5-minute time intervals ($$T = 288$$, $$\tau = 5/60$$ hours). The agent controls one of the battery systems and is rewarded for submitting bids that result in charging (buy) when prices are low, and discharging (sell) when prices and CO2 emissions are high, thus performing price arbitrage.

While ElectricityMarketEnv is a simplification of real-world electricity markets, the parameters of the environment's generators and battery systems and the historical electricity demand and forecasts are obtained from actual systems within California. In this regard, ElectricityMarketEnv serves as a minimum test-bed for RL algorithms, as a real-world electricity market is significantly more complex (_e.g._, featuring congestion constraints).

## Observation Space
An observation is $$s(t) = (t, e, a(t-1), x_{t-1}, p_{t-1}, l_{t-1}, \hat{l}_t, m_{t-1}, \hat{m}_{t:t+k|t})$$. $$t \in \Z_+$$ is the time step in $$[0,288]$$. $$e \in \R_+$$ is the agent's battery energy level (in MWh). $$a(t-1) \in \R_+^2$$ is the previous action. $$x_{t-1} \in \R$$ is the previous dispatch (in MWh) asked of the agent, and $$p_{t-1} \in \R$$ is market clearing price from the previous step (in $/MWh). $$l_{t-1} \in \R$$ is energy demand from the previous step (in MWh). $$\hat{l}_t \in \R$$ is estimated net demand for this step (in MWh).

## Action Space
Each agent action is a bid $$a(t) = (a^c, a^d) \in \R_+^2$$, representing prices ($/MWh) that the agent is willing to pay (or receive) for charging (or discharging) per MWh of energy, at time step $t$. Bids for other generators and battery systems are sampled randomly by the environment. The environment solves the optimal dispatch problem to determine the electricity price $$p_t$$ (in $/MWh) and the agent's dispatch $$x_t \in \R$$, which is the amount of energy (in MWh) that the agent is obligated to sell into or buy from the grid within the next time step. The dispatch in turn determines the storage system's next energy level. We also provide a wrapper that discretizes the action space into 3 actions only: charge, do nothing, or discharge.

## Reward Function
The reward function is a sum of three components: $$r(t) = r_R(t) + r_C(t) - c_T(t)$$. The revenue term $$r_R(t) = p_t x_t$$ is the immediate revenue from the dispatch. The CO2 emissions reward term $$r_C(t) = P_\text{CO$_2$} m_t x_t$$ represents the price of CO2 emissions displaced or incurred by the battery dispatch. The terminal cost $$c_T(t)$$, which is nonzero only when $$t=T$$, encourages the battery to have the same energy level at the end of the day as when it started. We also provide an option to delay all reward signals until the terminal time step (intermediate rewards are set to 0).