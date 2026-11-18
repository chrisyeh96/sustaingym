"""Jobs and job generators for DataCenterEnv.

All times are counted in 5-minute steps from the start of an episode, which is
local midnight of the episode's day.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

STEP_MINUTES = 5
STEPS_PER_HOUR = 60 // STEP_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR

# Default share of each measured workload among batch jobs. Workload ids refer
# to rows of data/data_center/gpu_power.csv (see the README next to it):
#   W3 LoRA fine-tuning, W4 ResNet-50 training, W5 GPT-2-small training,
#   W6 offline batch LLM inference (vLLM).
DEFAULT_MIX: dict[str, float] = {'W3': 0.15, 'W4': 0.40, 'W5': 0.30, 'W6': 0.15}

WorkloadRule = Literal['mix', 'shape']


@dataclass
class Job:
    """A non-preemptive GPU batch job. Times are in steps from the start of
    the episode."""
    job_id: int
    """index of the job within its episode"""
    arrival: int
    """step at which the job joins the queue"""
    duration: int
    """number of steps the job runs once started"""
    num_gpus: int
    """number of GPUs the job holds while it runs"""
    workload: str
    """workload id used by the power model, e.g. ``'W4'``"""
    deadline: int
    """step by which the job should have finished"""
    start: int = -1
    """step at which the job started, or -1"""
    finish: int = -1
    """step at which the job finished, or -1"""
    missed: bool = False
    """whether the job has passed its deadline unfinished"""

    @property
    def latest_start(self) -> int:
        """Last step at which the job can start and still meet its deadline."""
        return self.deadline - self.duration


def assign_workloads(num_gpus: np.ndarray, duration: np.ndarray,
                     rule: WorkloadRule, mix: Mapping[str, float],
                     rng: np.random.Generator) -> list[str]:
    """Assigns a workload id to each job.

    Args:
        num_gpus: GPUs per job
        duration: duration per job, in steps
        rule: ``'mix'`` draws each job's workload independently from ``mix``.
            ``'shape'`` uses the job's shape: 8 or more GPUs is W5 (large
            model training), 2 to 7 GPUs is W4 (training), and single-GPU
            jobs are W6 (batch inference) when they run at most one hour
            and W3 (fine-tuning) otherwise.
        mix: workload probabilities for the ``'mix'`` rule
        rng: random generator

    Returns:
        list of workload ids, one per job
    """
    if rule == 'mix':
        names = list(mix.keys())
        probs = np.array([mix[k] for k in names], dtype=float)
        probs /= probs.sum()
        idx = rng.choice(len(names), size=len(num_gpus), p=probs)
        return [names[i] for i in idx]
    if rule == 'shape':
        out = []
        for g, d in zip(num_gpus, duration):
            if g >= 8:
                out.append('W5')
            elif g >= 2:
                out.append('W4')
            elif d <= STEPS_PER_HOUR:
                out.append('W6')
            else:
                out.append('W3')
        return out
    raise ValueError(f'unknown workload rule {rule!r}')


class JobGenerator:
    """Base class. Subclasses return the jobs that arrive on one day.

    Args:
        max_slack_hours: each job's deadline is its arrival plus its duration
            plus a slack drawn uniformly from [0, max_slack_hours]
        workload_rule: see `assign_workloads`
        mix: see `assign_workloads`

    Attributes:
        max_slack_steps: int, largest slack a job can get, in steps
        max_duration_steps: int | None, longest job the generator can return,
            in steps, or None if it has no fixed limit
        workload_rule: str, ``'mix'`` or ``'shape'``
        mix: dict[str, float], workload probabilities for the ``'mix'`` rule
    """
    def __init__(self, max_slack_hours: float = 12.0,
                 workload_rule: WorkloadRule = 'mix',
                 mix: Mapping[str, float] | None = None):
        self.max_slack_steps = int(round(max_slack_hours * STEPS_PER_HOUR))
        self.max_duration_steps: int | None = None
        self.workload_rule = workload_rule
        self.mix = dict(DEFAULT_MIX if mix is None else mix)

    def available_days(self) -> list[pd.Timestamp] | None:
        """Days this generator can produce jobs for, or None for any day."""
        return None

    def _sample(self, day: pd.Timestamp, num_gpus: int,
                rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns arrival steps, durations (steps) and GPU counts."""
        raise NotImplementedError

    def generate(self, day: pd.Timestamp, num_gpus: int,
                 rng: np.random.Generator) -> list[Job]:
        """Returns the day's jobs sorted by arrival.

        Args:
            day: local calendar day of the episode
            num_gpus: cluster size. Jobs that need more GPUs are dropped.
            rng: random generator (the environment's ``np_random``)
        """
        arrival, duration, gpus = self._sample(day, num_gpus, rng)
        keep = (gpus >= 1) & (gpus <= num_gpus) & (duration >= 1)
        arrival, duration, gpus = arrival[keep], duration[keep], gpus[keep]
        order = np.argsort(arrival, kind='stable')
        arrival, duration, gpus = arrival[order], duration[order], gpus[order]
        slack = rng.integers(0, self.max_slack_steps + 1, size=len(arrival))
        workloads = assign_workloads(gpus, duration, self.workload_rule,
                                     self.mix, rng)
        return [
            Job(job_id=i, arrival=int(a), duration=int(d), num_gpus=int(g),
                workload=w, deadline=int(a + d + s))
            for i, (a, d, g, s, w) in enumerate(
                zip(arrival, duration, gpus, slack, workloads))
        ]


class SyntheticJobGenerator(JobGenerator):
    """Synthetic jobs with a daily arrival cycle. Needs no data download.

    Arrivals follow a Poisson process whose rate peaks in the afternoon.
    Durations are log-normal (median ``median_hours``), and GPU counts are
    drawn from ``gpu_sizes``. The arrival rate is set so that the expected
    GPU-hours requested over the day equal ``offered_load`` times the
    cluster's GPU-hours.

    Args:
        offered_load: requested GPU-hours as a fraction of cluster capacity
        median_hours: median job duration
        max_duration_hours: durations are clipped to this value
        gpu_sizes: mapping from GPU count to probability
        **kwargs: see `JobGenerator`

    Attributes:
        offered_load: float, requested GPU-hours as a fraction of capacity
        median_steps: float, median job duration, in steps
        max_duration_steps: int, longest job duration, in steps
        sizes: np.ndarray, possible GPU counts
        size_p: np.ndarray, probability of each GPU count
    """
    def __init__(self, offered_load: float = 0.6, median_hours: float = 1.0,
                 max_duration_hours: float = 12.0,
                 gpu_sizes: Mapping[int, float] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.offered_load = offered_load
        self.median_steps = median_hours * STEPS_PER_HOUR
        self.max_duration_steps = int(max_duration_hours * STEPS_PER_HOUR)
        sizes = {1: 0.6, 2: 0.2, 4: 0.12, 8: 0.08} if gpu_sizes is None else gpu_sizes
        self.sizes = np.array(list(sizes.keys()), dtype=int)
        self.size_p = np.array(list(sizes.values()), dtype=float)
        self.size_p /= self.size_p.sum()

    def _sample(self, day: pd.Timestamp, num_gpus: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # only draw job sizes that fit, so the offered load is met on small clusters
        fits = self.sizes <= num_gpus
        if not fits.any():
            raise ValueError(f'no job size in {self.sizes.tolist()} fits on {num_gpus} GPUs')
        sizes, size_p = self.sizes[fits], self.size_p[fits] / self.size_p[fits].sum()
        sigma = 1.0
        mean_dur = min(self.median_steps * np.exp(sigma ** 2 / 2),
                       self.max_duration_steps)
        mean_gpus = float((sizes * size_p).sum())
        n_mean = self.offered_load * num_gpus * STEPS_PER_DAY / (mean_dur * mean_gpus)
        n = rng.poisson(n_mean)
        # diurnal arrival profile: more submissions during the working day
        steps = np.arange(STEPS_PER_DAY)
        weight = 1.0 + 0.6 * np.sin(2 * np.pi * (steps / STEPS_PER_DAY - 0.375))
        arrival = rng.choice(STEPS_PER_DAY, size=n, p=weight / weight.sum())
        duration = np.clip(np.round(rng.lognormal(np.log(self.median_steps), sigma, n)),
                           1, self.max_duration_steps).astype(int)
        gpus = rng.choice(sizes, size=n, p=size_p)
        return arrival.astype(int), duration, gpus.astype(int)


class TraceJobGenerator(JobGenerator):
    """Jobs replayed from a GPU cluster trace.

    The trace is a CSV with columns ``submit_time`` (local time, no time
    zone), ``duration_s`` and ``num_gpus``. One episode replays the jobs
    submitted on the episode's calendar day, at the same local clock time.
    Jobs shorter than 5 minutes or longer than ``max_duration_hours`` are
    dropped. The remaining jobs are thinned at random so that the day's
    requested GPU-hours equal ``offered_load`` times the cluster's GPU-hours
    (days with less demand are kept whole).

    ``sustaingym/data/data_center/prepare_helios.py`` writes such a CSV from
    the public Helios traces.

    Args:
        trace_path: path to the trace CSV
        offered_load: target requested GPU-hours as a fraction of capacity
        max_duration_hours: longest job kept
        **kwargs: see `JobGenerator`

    Attributes:
        by_day: dict[pd.Timestamp, pd.DataFrame], trace jobs grouped by
            submission day
        offered_load: float, target requested GPU-hours as a fraction of
            capacity
    """
    def __init__(self, trace_path: str, offered_load: float = 0.6,
                 max_duration_hours: float = 12.0, **kwargs: Any):
        super().__init__(**kwargs)
        df = pd.read_csv(trace_path, parse_dates=['submit_time'])
        df = df[(df['duration_s'] >= 60 * STEP_MINUTES)
                & (df['duration_s'] <= max_duration_hours * 3600)
                & (df['num_gpus'] >= 1)]
        df = df.assign(day=df['submit_time'].dt.normalize())
        self.by_day: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(str(day)): g for day, g in df.groupby('day')}
        self.offered_load = offered_load
        self.max_duration_steps = int(np.ceil(max_duration_hours * 3600 / (60 * STEP_MINUTES)))

    def available_days(self) -> list[pd.Timestamp]:
        return sorted(self.by_day.keys())

    def _sample(self, day: pd.Timestamp, num_gpus: int, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = pd.Timestamp(day.date())
        if key not in self.by_day:
            raise ValueError(f'trace has no jobs on {key.date()}')
        g = self.by_day[key]
        g = g[g['num_gpus'] <= num_gpus]
        minutes = (g['submit_time'] - key).dt.total_seconds().to_numpy() / 60
        arrival = (minutes // STEP_MINUTES).astype(int)
        duration = np.ceil(g['duration_s'].to_numpy() / (60 * STEP_MINUTES)).astype(int)
        gpus = g['num_gpus'].to_numpy().astype(int)
        demand = float((duration * gpus).sum())
        target = self.offered_load * num_gpus * STEPS_PER_DAY
        p = min(1.0, target / demand) if demand > 0 else 1.0
        keep = rng.random(len(arrival)) < p
        return arrival[keep], duration[keep], gpus[keep]
