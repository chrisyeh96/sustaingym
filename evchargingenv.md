---
layout: page
title: EVChargingEnv
permalink: /evchargingenv/
nav_order: 2
use_math: true
---

EVChargingEnv uses ACNSim to simulate the charging of electric vehicles (EVs) based on actual data gathered from EV charging networks between fall 2018 and summer 2021. ACNSim is a digital twin of actual charging networks at Caltech and JPL, taking into account nonlinear battery charging dynamics and unbalanced three-phase AC power flows, and is thus very realistic. The Caltech and JPL networks have $$n=54$$ and 52 charging stations (abbrv. EVSEs, Electric Vehicle Supply Equipment), respectively. When drivers charged their EVs, they provided an estimated time of departure and amount of energy requested. Thus, each charging session includes time of EV arrival, estimated departure, actual departure, energy delivered, and EVSE ID.

Each episode starts at midnight and runs for 24 hours, with 5-minute time step intervals. EV arrival and departure events are discretized to these 5-minute intervals ($$T = 288$$, $$\tau = 5/60$$ hours). At every time step, an agent decides the charging rates (a.k.a. ``pilot signals'') for each EVSE to be executed for the duration of that time step. That is, a single agent simultaneously controls all $$n$$ EVSEs.

EVChargingEnv supports real historical data as well as data sampled from a 30-component Gaussian Mixture Model (GMM) fit to historical data.

## Observation Space
An observation at time $$t$$ is $$s(t) = (t, d, e, m_{t-1}, \hat{m}_{t:t+k-1|t})$$. $$t \in \Z_+$$ is the fraction of day between 0 and 1, inclusive. $$d \in \Z^n$$ is estimated remaining duration of each EV (in \# of time steps). $$e \in \R_+^n$$ is remaining energy demand of each EV (in kWh). If no EV is charging at EVSE $$i$$, then $$d_i = 0$$ and $$e_i = 0$$. If an EV charging at EVSE $$i$$ has exceeded the user-specified estimated departure time, then $$d_i$$ becomes negative, while $$e_i$$ may still be nonzero.

## Action Space
The action space is continuous $$a(t) \in [0, 1]^n$$, representing the pilot signal normalized by the maximum signal allowed $$M$$ (in amps) for each EVSE. Physical infrastructure in a charging network constrain the set $$\mathcal{A}_t$$ of feasible actions at each time step $$t$$. Furthermore, the EVSEs only support discrete pilot signals, so $$\mathcal{A}_t$$ is nonconvex. To satisfy these physical constraints, EVChargingEnv can project an agent's action $$a(t)$$ into the convex hull of $$\mathcal{A}_t$$ and round it to the nearest allowed pilot signal, resulting in final normalized pilot signals $$\tilde{a}(t)$$. ACNSim processes $$\tilde{a}(t)$$ and returns the actual charging rate $$M \bar{a} \in \R_+^n$$ (in amps) delivered at each EVSE, as well as the remaining demand $$e_i(t+1)$$.

## Reward Function
The reward function is a sum of three components: $$r(t) = p(t) - c_V(t) - c_C(t)$$. The profit term $$p(t)$$ aims to maximize energy delivered to the EVs. The constraint violation cost $$c_V(t)$$ aims to reduce physical constraint violations and encourage the agent's action $$a(t)$$ to be in $$\mathcal{A}_t$$. Finally, the CO<sub>2</sub> emissions cost $$c_C(t)$$, which is a function of the MOER $$m_t$$ and charging action, aims to reduce emissions by encouraging the agent to charge EVs when the MOER is low.

## Getting Started

TODO

```bash
python run_script.py  # TODO
```