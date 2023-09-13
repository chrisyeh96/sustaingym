# DatacenterEnv

DatacenterEnv is a simulator for carbon-aware job scheduling in datacenters, which aims to reduce the carbon emissions associated with electrcity usage in a datacenter. Carbon-aware job scheduling is premised upon two facts: (i) a significant fraction of a datacenter's workload (_e.g._, up to 50% in some of Google datacenters) is comprised of low priority jobs whose execution can be delayed, and (ii) the carbon intensity of the electric grid fluctuates predictably over time. Therefore, if the execution of low priority workload is delayed to a time of day with "greener" energy, the datacenter's carbon emissions can be minimized.

We assume that jobs are scheduled according to a priority queue, with jobs spread evenly across the available machines. Following [Radovanovic et al. (2022)](https://arxiv.org/abs/2106.11750), we implement workload execution delay by artificially limiting the total datacenter capacity with a _virtual capacity curve_ (VCC) at each time step. If more jobs are enqueued than the VCC permits, then the jobs must wait until the VCC is raised high enough to allow the jobs to run. Simulation is carried out by replaying a sample of [real job traces from a Google cluster from May 2019](https://github.com/google/cluster-data/blob/master/ClusterData2019.md). One timestep in the environment corresponds to one hour, and each episode lasts the whole month.

## Observation Space
An observation $s(t) = (a(t-1), d_t, n, \hat{m}_{t:t+23|t}) \in \R^{27}$ contains the active VCC $a(t-1)$ set from the previous time step, currently running compute load $d_t$, number of jobs waiting to be scheduled $n$, as well as the forecasted MOER for the next 24h $\hat{m}_{t:t+23|t}$.

## Action Space
An agent interacts with the environment by setting the VCC for the next time step. In particular, an agents takes a scalar action $a(t) \in[0,1]$, denoting the _fraction_ of the datacenter's maximum capacity allowed to be allocated by the scheduler.

## Reward Function
The reward consists of two components that encourage the agents to trade-off between scheduling more jobs and reducing associated carbon emissions. The first component penalizes the agent when jobs are scheduled more than 24h after they were originally submitted. The second component is a carbon emissions cost. Formally, the reward is specified as

$$
r(t) = d_t \cdot m_t + \mathbf{1}_{ [t \% 24 = 0 ]} \max \left(0, 0.97 w_t - C \sum_{h=0}^{23} a(t-h)\right)
$$

where $d_t \cdot m_t$ is the carbon emissions, $C$ is the datacenter's maximum capacity, and $w_t$ is the total job-hours of enqueued jobs on that day.

## Distribution Shift
The distribution shift in DatacenterEnv comes from changes in the MOER between 2019 and 2021.

## Getting Started

### Installation

Coming soon!

### Using our training script

Coming soon!

### Custom RL Loop

Coming soon!
