# DataCenterEnv

DataCenterEnv is a simulator for carbon-aware job scheduling in datacenters, which aims to reduce the carbon emissions associated with electricity usage in a datacenter. Carbon-aware job scheduling is premised upon two facts: (i) a significant fraction of a datacenter's workload (_e.g._, up to 50% in some of Google datacenters) is comprised of low priority jobs whose execution can be delayed, and (ii) the carbon intensity of the electric grid fluctuates predictably over time. Therefore, if the execution of low priority workload is delayed to a time of day with "greener" energy, the datacenter's carbon emissions can be minimized.

The environment models a cluster of identical GPUs. Every 5 minutes, the agent decides which queued batch jobs to start and which to hold for cleaner power. A started job keeps its GPUs until it finishes. The reward charges the carbon of the cluster's energy, priced at the marginal operating emissions rate (MOER) of a California balancing authority, plus a penalty for each missed deadline.

Each episode starts at local midnight (US/Pacific) and runs in 5-minute steps for `horizon_days` days (default 3, so 864 steps). Jobs arrive only during the first day, so every policy is charged for the cluster's energy over the same window, idle GPUs included. Every job has a deadline inside the episode. Jobs come from `SyntheticJobGenerator` by default, or from `TraceJobGenerator`, which replays prepared Helios GPU traces.

An earlier plan for this environment used an hourly virtual capacity curve over Google cluster traces, following [Radovanovic et al. (2022)](https://arxiv.org/abs/2106.11750). This version instead admits individual GPU jobs every 5 minutes.

## Observation Space

The agent acts on the `queue_window` most urgent queued jobs (default 10), ordered by latest feasible start, then arrival and job ID. For a window of $k$ jobs, the observation is a float32 `Box` of shape $(42 + 5k,)$, so 92 values with the default window. The first 42 values describe the cluster and the carbon signal:

| Feature | Values | Range |
|---|---:|---|
| Sine and cosine of the time of day | 2 | -1 to 1 |
| Current step divided by the episode length | 1 | 0 to 1 |
| Fraction of GPUs free | 1 | 0 to 1 |
| Queued jobs outside the window, divided by 100 and clipped | 1 | 0 to 1 |
| Current MOER, kg CO2/kWh | 1 | 0 to 2 |
| MOER forecast for the next 36 steps (3 hours) | 36 | 0 to 2 |

Each window slot adds five values, all zero for an empty slot:

| Per-job feature | Range |
|---|---|
| 1 if the slot holds a job | 0 or 1 |
| GPUs requested divided by the cluster's GPUs | 0 to 1 |
| Duration divided by the job generator's longest duration | 0 to 1 |
| Steps until the latest feasible start, divided by the generator's largest slack and clipped | -1 to 1 |
| Power per GPU for the job's workload divided by the GPU's power limit, clipped | 0 to 1 |

If the job generator has no fixed duration limit, durations are divided by the longest job of the episode instead. MOER values are clipped to 0–2 kg CO2/kWh.

## Action Space

The action is `MultiBinary(k)`. A 1 asks to start the job in that window slot, and a 0 leaves it queued. Slots are taken in order, and a job starts only if enough GPUs are free. Running jobs are never preempted.

## Reward Function

At each step, the reward is

$$
r(t) = -\left(c_t + p \cdot n_t\right)
$$

where $c_t$ is the step's energy in kWh times the current MOER, $n_t$ is the number of jobs whose deadline passed unfinished during the step, and $p$ is `miss_penalty_kg` (default 10 kg CO2). The energy includes busy GPUs, idle GPUs and job-start energy, as the power model defines them. Each missed deadline is penalized once. The `info` dict reports `carbon_kg`, `energy_kwh`, `moer`, `busy_gpus`, `num_started`, `queue_len` and `new_misses` for the step. At the end of the horizon, `truncated` is true. `terminated` is always false.

## Distribution Shift

The bundled MOER data cover CAISO PG&E and SCE from May 2019 to August 2021, so an agent can be trained on one region or period and evaluated on another. The jobs can also shift, from synthetic jobs to replayed Helios traces.

## Power Models

| Name | Busy GPU | Idle GPU | Job start |
|---|---|---|---|
| `tdp` | power limit (TDP) | 0 W | 0 |
| `tdp_idle40` | power limit (TDP) | 40% of TDP | 0 |
| `measured` | measured power for the job's workload | measured idle power | 0 |
| `measured_released` | as `measured` | 0 W (GPU handed back) | measured cold-start energy |

Measured values are NVML board power for seven GPU types (T4 to B200). They exclude host CPU, memory, networking, cooling and facility overhead. A GPU must have an entry for every workload the job generator can assign. The T4 has no W6 entry, so on a T4 use `tdp` or a workload mix without W6. `sustaingym/data/data_center/README.md` describes the measurements, their source and their licence.

## Baselines

`sustaingym.algorithms.data_center` provides four policies:

- `ASAP` starts every job as soon as GPUs are free.
- `CarbonThreshold` starts a job when the current MOER is in the lowest 30% of the current value and the 3-hour forecast.
- `ForecastWindow` starts a job now unless the forecast shows a lower-carbon start within the next 3 hours and before the job's latest start.
- `OracleWindow` does the same with the actual future MOER. It is a reference only.

Each one takes `deadline_safe=True` to plan held jobs against GPU capacity, so that a held job does not find the cluster full at its latest start. On busy traces, a larger `queue_window` (for example 32) lets carbon-aware policies act on more jobs per step.

## Getting Started

### Installation

From a clone of the repository, install SustainGym with the `datacenter` extra:

```bash
pip install -e ".[datacenter]"
```

### Custom RL Loop

The environment is registered as `sustaingym/DataCenter-v0`. Its defaults are 64 H100s, the `measured` power model, the `SGIP_CAISO_PGE` MOER signal, and episodes starting between May 1 and September 30, 2020.

```python
import gymnasium as gym
import sustaingym

env = gym.make('sustaingym/DataCenter-v0')
obs, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

To fix the episode's day, pass `options={'day': '2020-06-02'}` to `reset`.

### Replaying a Job Trace

The Helios traces are not bundled. This downloads them (36 MB, checked by SHA-256) and writes one cluster's GPU jobs:

```bash
python -m sustaingym.data.data_center.prepare_helios --cluster Venus --out helios_venus.csv.gz
```

```python
from sustaingym.envs.data_center import DataCenterEnv, TraceJobGenerator

jobs = TraceJobGenerator('helios_venus.csv.gz', offered_load=0.6, workload_rule='shape')
env = DataCenterEnv(jobs, num_gpus=64, gpu='H100', power_model='measured')
obs, info = env.reset(seed=0, options={'day': '2020-06-02'})
```

### Running a Baseline

```python
from sustaingym.algorithms.data_center import ForecastWindow
from sustaingym.envs.data_center import DataCenterEnv

env = DataCenterEnv(queue_window=32)
policy = ForecastWindow(env, deadline_safe=True, reserve_frac=0.05)
summary = policy.run_episode(seed=0, day='2020-06-02')
print(summary['carbon_kg'], summary['misses'])
```

## Limitations

- Each cluster has one GPU type. There are no placement, preemption or power-capping actions.
- The Helios traces do not record workloads, so the trace generator assigns them by a rule (`workload_rule`).
- The measured power comes from one cloud provider on one day, with a few runs per GPU and workload.
- Steps ignore daylight saving, so the time of day is an hour off after a switch. The default date range has none.
