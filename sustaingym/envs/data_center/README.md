# DataCenterEnv

Carbon-aware admission control for a GPU cluster. Every 5 minutes the agent picks which queued batch jobs to start and which to hold for a lower-carbon time, without missing deadlines. Carbon is priced with historical CAISO marginal emissions (MOER).

Install with `pip install sustaingym[datacenter]`, or `conda env update --file env_datacenter.yml --prune`.

```python
import gymnasium as gym
import sustaingym

env = gym.make('sustaingym/DataCenter-v0')          # synthetic jobs, 64 H100s
obs, info = env.reset(seed=0)
```

With a real trace (see `sustaingym/data/data_center/README.md` to prepare one):

```python
from sustaingym.envs.data_center import DataCenterEnv, TraceJobGenerator
from sustaingym.algorithms.data_center import ForecastWindow

gen = TraceJobGenerator('helios_venus.csv.gz', offered_load=0.6, workload_rule='shape')
env = DataCenterEnv(gen, num_gpus=64, gpu='H100', power_model='measured')
summary = ForecastWindow(env).run_episode(seed=0, day='2020-06-02')
```

The observation, action and reward are documented in the `DataCenterEnv` class docstring.

## Power models

| Name | Busy GPU | Idle GPU | Job start |
|---|---|---|---|
| `tdp` | TDP | 0 W | 0 |
| `tdp_idle40` | TDP | 40% of TDP | 0 |
| `measured` | measured power of the job's workload | measured idle power | 0 |
| `measured_released` | as `measured` | 0 W (GPU handed back) | measured cold-start energy |

Measured values are NVML board power on seven GPU types (T4 to B200); see the data README.

## Baselines

In `sustaingym.algorithms.data_center`:

- `ASAP`: start every job as soon as GPUs are free.
- `CarbonThreshold`: start when the current MOER is in the lowest 30% of the 3-hour forecast.
- `ForecastWindow`: start at the lowest-carbon time in the forecast.
- `OracleWindow`: the same with the actual future MOER (reference only).

Each takes `deadline_safe=True` to plan held jobs against GPU capacity, so that a held job does not find the cluster full at its latest start:

```python
policy = ForecastWindow(env, deadline_safe=True, reserve_frac=0.05)
```

Only the `queue_window` most urgent jobs can start each step. On busy traces, raise it (e.g. `queue_window=32`) for carbon-aware policies.

## Limitations

- One GPU type per cluster; no preemption, placement or power capping.
- Trace jobs get workloads by rule, since the traces do not record them.
- The bundled MOER covers two CAISO regions, May 2019 to August 2021.
- Steps ignore daylight saving, so time of day is an hour off after a switch. The default date range has none.
