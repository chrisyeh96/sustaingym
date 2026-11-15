"""
The module implements the DataCenterEnv class.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from gymnasium import Env, spaces
import numpy as np
import pandas as pd
import pytz

from sustaingym.data.load_moer import load_moer
from .jobs import (STEP_MINUTES, STEPS_PER_DAY, STEPS_PER_HOUR, Job,
                   JobGenerator, SyntheticJobGenerator)
from .power import PowerModel, make_power_model

FORECAST_STEPS = 36
GLOBAL_FEATURES = 6 + FORECAST_STEPS
SLOT_FEATURES = 5
MOER_CLIP = 2.0  # kg CO2 / kWh; the bundled SGIP data stay below 0.7
LOCAL_TZ = pytz.timezone('US/Pacific')


class DataCenterEnv(Env[np.ndarray, np.ndarray]):
    """DataCenterEnv class.

    This class simulates carbon-aware admission control for a cluster of
    identical GPUs. Batch jobs arrive during the first day of an episode and
    wait in a queue. Every 5 minutes the agent picks which of the most urgent
    queued jobs to start; the others keep waiting. A started job holds its
    GPUs until it finishes (no preemption). Each step costs the carbon of the
    cluster's energy, priced at the historical marginal operating emissions
    rate (MOER) of a California balancing authority, plus a penalty for each
    job that passes its deadline unfinished.

    Energy comes from a power model (see power.py). Episodes last
    ``horizon_days`` from local midnight. Jobs only arrive on the first day,
    so every policy is charged over the same window.

    This environment's API is known to be compatible with Gymnasium v0.28.

    In what follows:

    - ``k`` = ``queue_window``, the number of queued jobs the agent acts on
    - ``f`` = 36, the number of 5-minute MOER forecast steps

    Actions:

    .. code:: none

        Type: MultiBinary(k)
        Action                                      Shape   Values
        start the job in window slot i, if enough   k       0 or 1
        GPUs are free (slots are taken in order)

    Observations:

    .. code:: none

        Type: Box(42 + 5k)
                                                    Shape   Min     Max
        sin and cos of the time of day              2       -1      1
        step / horizon                              1       0       1
        fraction of GPUs free                       1       0       1
        queued jobs beyond the window / 100         1       0       1
        current MOER (kg CO2 / kWh)                 1       0       2
        forecasted MOER (kg CO2 / kWh)              f       0       2
        for window slot i (index 42 + 5i onward):
            1 if the slot has a job                 k       0       1
            job GPUs / num_gpus                     k       0       1
            job duration / longest duration         k       0       1
            (latest start - step) / max slack       k       -1      1
            job power per GPU / TDP                 k       0       1

    Window slots hold queued jobs sorted by latest feasible start time.

    Reward: ``-(carbon_kg + miss_penalty_kg * new_misses)`` each step.

    Args:
        job_generator: source of jobs. Defaults to
            `sustaingym.envs.data_center.jobs.SyntheticJobGenerator`.
        num_gpus: cluster size
        gpu: GPU type, a row of the power table (e.g. ``'H100'``, ``'L4'``)
        power_model: power model name or instance
        moer_ba: balancing authority of the bundled MOER data,
            ``'SGIP_CAISO_PGE'`` or ``'SGIP_CAISO_SCE'``
        date_range: first and last local day an episode may start on
        queue_window: number of queued jobs the agent decides on each step
        horizon_days: episode length in days
        miss_penalty_kg: penalty per missed deadline, in kg CO2-equivalent
        moer_dir: optional folder with MOER files (see
            `sustaingym.data.load_moer.load_moer`)

    Attributes:
        # attributes required by gymnasium.Env
        action_space: spaces.MultiBinary, structure of actions expected by env
        observation_space: spaces.Box, structure of observations

        # attributes specific to DataCenterEnv
        job_generator: JobGenerator, source of each episode's jobs
        num_gpus: int, cluster size
        gpu: str, GPU type
        power: PowerModel, model that turns GPU states into watts
        moer_ba: str, balancing authority of the MOER data
        queue_window: int, number of queued jobs the agent acts on
        horizon: int, episode length in steps
        miss_penalty_kg: float, penalty per missed deadline
        tdp: float, TDP of one GPU, in W
        moer_df: pd.DataFrame, MOER and forecasts over the whole date range
        days: list[pd.Timestamp], days an episode can start on
        day: pd.Timestamp, day of the current episode
        t: int, current step in the episode
        jobs: list[Job], all jobs of the current episode
        queue: list[Job], jobs waiting to start
        running: list[Job], jobs holding GPUs
    """
    metadata: dict[str, Any] = {'render_modes': []}

    def __init__(self, job_generator: JobGenerator | None = None,
                 num_gpus: int = 64, gpu: str = 'H100',
                 power_model: str | PowerModel = 'measured',
                 moer_ba: str = 'SGIP_CAISO_PGE',
                 date_range: tuple[str, str] = ('2020-05-01', '2020-09-30'),
                 queue_window: int = 10, horizon_days: int = 3,
                 miss_penalty_kg: float = 10.0,
                 moer_dir: str | None = None):
        if num_gpus < 1 or queue_window < 1 or horizon_days < 2:
            raise ValueError('need num_gpus >= 1, queue_window >= 1, horizon_days >= 2')
        self.job_generator = SyntheticJobGenerator() if job_generator is None else job_generator
        self.num_gpus = num_gpus
        self.gpu = gpu
        self.power = make_power_model(power_model)
        if self.job_generator.workload_rule == 'shape':
            workloads = {'W3', 'W4', 'W5', 'W6'}
        else:
            workloads = set(self.job_generator.mix)
        self.power.check(gpu, workloads)
        self.moer_ba = moer_ba
        self.queue_window = queue_window
        self.horizon = horizon_days * STEPS_PER_DAY
        self.miss_penalty_kg = miss_penalty_kg
        max_duration = self.job_generator.max_duration_steps
        if max_duration is not None:
            # a job that arrives in the last step of day 1 with the longest
            # duration and slack must still be due inside the episode
            latest_due = STEPS_PER_DAY - 1 + max_duration + self.job_generator.max_slack_steps
            if latest_due > self.horizon:
                raise ValueError(f'jobs can be due at step {latest_due}, after the '
                                 f'{self.horizon}-step horizon; raise horizon_days')
        self.tdp = self.power.tdp_w(gpu)
        self._idle_w = self.power.idle_w(gpu)
        self._start_j = self.power.start_j(gpu)

        first = pd.Timestamp(date_range[0])
        last = pd.Timestamp(date_range[1])
        # a day of margin on each side; at the edges of the data, fall back
        # to the range itself (days without enough data are dropped below)
        start, end = first - timedelta(days=1), last + timedelta(days=horizon_days + 1)
        moer = None
        for s, e in [(start, end), (first, end), (start, last), (first, last)]:
            try:
                moer = load_moer(s.to_pydatetime(), e.to_pydatetime(), moer_ba, moer_dir)
                break
            except FileNotFoundError:
                continue
        if moer is None:
            raise ValueError(f'no {moer_ba} MOER data for {first.date()} to {last.date()}')
        moer = moer[~moer.index.duplicated(keep='first')].sort_index()
        self.moer_df = moer[['moer'] + [f'f{i}' for i in range(1, FORECAST_STEPS + 1)]]

        days = list(pd.date_range(first, last, freq='D'))
        trace_days = self.job_generator.available_days()
        if trace_days is not None:
            allowed = {pd.Timestamp(d.date()) for d in trace_days}
            days = [d for d in days if d in allowed]
        lo, hi = self.moer_df.index[0], self.moer_df.index[-1]
        span = timedelta(minutes=STEP_MINUTES * (self.horizon + FORECAST_STEPS))
        self.days = [d for d in days
                     if lo <= self._utc_start(d) and self._utc_start(d) + span <= hi]
        if not self.days:
            raise ValueError('no day in date_range has both jobs and MOER data')

        n_obs = GLOBAL_FEATURES + SLOT_FEATURES * queue_window
        self._n_obs = n_obs
        low = np.zeros(n_obs, dtype=np.float32)
        high = np.ones(n_obs, dtype=np.float32)
        low[0:2] = -1.0
        high[5:GLOBAL_FEATURES] = MOER_CLIP
        low[GLOBAL_FEATURES + 3::SLOT_FEATURES] = -1.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.MultiBinary(queue_window)

        self.jobs: list[Job] = []
        self.queue: list[Job] = []
        self.running: list[Job] = []
        self.t = 0

    @staticmethod
    def _utc_start(day: pd.Timestamp) -> pd.Timestamp:
        local = LOCAL_TZ.localize(datetime(day.year, day.month, day.day))
        return pd.Timestamp(local.astimezone(pytz.UTC))

    @property
    def free_gpus(self) -> int:
        """Number of idle GPUs."""
        return self.num_gpus - sum(j.num_gpus for j in self.running)

    def queue_window_jobs(self) -> list[Job]:
        """Queued jobs in window order (most urgent first)."""
        self.queue.sort(key=lambda j: (j.latest_start, j.arrival, j.job_id))
        return self.queue[:self.queue_window]

    def moer_now(self) -> float:
        """Current MOER, kg CO2 / kWh."""
        return float(self._moer[min(self.t, self.horizon - 1)])

    def moer_forecast(self) -> np.ndarray:
        """Forecast MOER for the next 36 steps."""
        return self._forecast[min(self.t, self.horizon - 1)].copy()

    def moer_future(self, n: int) -> np.ndarray:
        """Actual MOER for steps t .. t+n-1 (for non-causal reference policies)."""
        idx = np.clip(np.arange(self.t, self.t + n), 0, self.horizon - 1)
        return self._moer[idx].copy()

    def schedule(self) -> list[tuple[int, int]]:
        """(job_id, start step) of every started job, sorted by job id."""
        return sorted((j.job_id, j.start) for j in self.jobs if j.start >= 0)

    def episode_summary(self) -> dict[str, Any]:
        """Totals for the episode so far.

        Returns:
            summary: dict with ``carbon_kg`` (and its ``active_``, ``idle_`` and
                ``start_`` parts), ``energy_kwh`` (likewise split into
                ``active_kwh``, ``idle_kwh`` and ``start_kwh``), ``misses``,
                ``jobs``, ``finished``, ``mean_wait_h`` and ``day``
        """
        s: dict[str, Any] = dict(self._totals)
        s['jobs'] = len(self.jobs)
        s['finished'] = sum(j.finish >= 0 for j in self.jobs)
        waits = [j.start - j.arrival for j in self.jobs if j.start >= 0]
        s['mean_wait_h'] = float(np.mean(waits)) / STEPS_PER_HOUR if waits else 0.0
        s['day'] = str(self.day.date())
        return s

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None
              ) -> tuple[np.ndarray, dict[str, Any]]:
        """Starts an episode.

        Args:
            seed: seeds ``np_random``, which picks the day and draws the jobs.
                Fixing ``options['day']`` changes the jobs drawn for a seed.
            options: ``{'day': 'YYYY-MM-DD'}`` fixes the episode's day

        Returns:
            obs: first observation
            info: dict with the episode's ``day`` and number of ``jobs``
        """
        super().reset(seed=seed)
        if options is not None and 'day' in options:
            self.day = pd.Timestamp(options['day'])
            if self.day not in self.days:
                raise ValueError(f'day {self.day.date()} not available')
        else:
            self.day = self.days[int(self.np_random.integers(len(self.days)))]

        idx = pd.date_range(self._utc_start(self.day),
                            periods=self.horizon, freq=f'{STEP_MINUTES}min')
        block = self.moer_df.reindex(idx, method='ffill')
        values = np.clip(block.to_numpy(dtype=np.float32), 0.0, MOER_CLIP)
        self._moer = values[:, 0]
        self._forecast = values[:, 1:]

        self.jobs = self.job_generator.generate(self.day, self.num_gpus, self.np_random)
        if any(j.deadline > self.horizon for j in self.jobs):
            raise ValueError('a job is due after the end of the episode; raise horizon_days')
        self._arrivals: dict[int, list[Job]] = {}
        for job in self.jobs:
            self._arrivals.setdefault(job.arrival, []).append(job)
        self._max_duration = max([j.duration for j in self.jobs], default=1)
        self._max_slack = max(1, self.job_generator.max_slack_steps)
        self._job_power = {w: self.power.active_w(self.gpu, w)
                           for w in {j.workload for j in self.jobs}}

        self.queue, self.running = [], []
        self.t = 0
        self._totals = dict(carbon_kg=0.0, active_carbon_kg=0.0,
                            idle_carbon_kg=0.0, start_carbon_kg=0.0,
                            energy_kwh=0.0, active_kwh=0.0, idle_kwh=0.0,
                            start_kwh=0.0, misses=0)
        self.queue.extend(self._arrivals.get(0, []))
        return self._obs(), {'day': str(self.day.date()), 'jobs': len(self.jobs)}

    def step(self, action: np.ndarray
             ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Starts the chosen jobs, then advances the cluster by 5 minutes.

        Args:
            action: 0/1 for each window slot; 1 starts that job if enough GPUs
                are free

        Returns:
            obs: next observation
            reward: ``-(carbon_kg + miss_penalty_kg * new_misses)``
            terminated: always False
            truncated: whether the episode reached ``horizon_days``
            info: dict with ``carbon_kg``, ``energy_kwh``, ``moer``,
                ``busy_gpus``, ``num_started``, ``queue_len`` and
                ``new_misses`` for this step
        """
        action = np.asarray(action).reshape(-1)
        started = []
        for i, job in enumerate(self.queue_window_jobs()):
            if action[i] and job.num_gpus <= self.free_gpus:
                job.start = self.t
                self.running.append(job)
                started.append(job)
        for job in started:
            self.queue.remove(job)

        dt_h = STEP_MINUTES / 60
        busy = sum(j.num_gpus for j in self.running)
        active_kwh = sum(j.num_gpus * self._job_power[j.workload]
                         for j in self.running) * dt_h / 1000
        idle_kwh = (self.num_gpus - busy) * self._idle_w * dt_h / 1000
        start_kwh = sum(j.num_gpus for j in started) * self._start_j / 3.6e6
        energy_kwh = active_kwh + idle_kwh + start_kwh
        moer = self.moer_now()
        carbon = energy_kwh * moer

        self.t += 1
        finished = [j for j in self.running if j.start + j.duration <= self.t]
        for job in finished:
            job.finish = self.t
            self.running.remove(job)
        self.queue.extend(self._arrivals.get(self.t, []))

        truncated = self.t >= self.horizon
        new_misses = 0
        # earlier finishers were checked when they finished; an unfinished
        # job is late once t reaches its deadline (deadlines <= horizon)
        for job in finished + self.queue + self.running:
            late = job.finish > job.deadline if job.finish >= 0 else self.t >= job.deadline
            if late and not job.missed:
                job.missed = True
                new_misses += 1

        tot = self._totals
        tot['carbon_kg'] += carbon
        tot['active_carbon_kg'] += active_kwh * moer
        tot['idle_carbon_kg'] += idle_kwh * moer
        tot['start_carbon_kg'] += start_kwh * moer
        tot['energy_kwh'] += energy_kwh
        tot['active_kwh'] += active_kwh
        tot['idle_kwh'] += idle_kwh
        tot['start_kwh'] += start_kwh
        tot['misses'] += new_misses

        reward = -(carbon + self.miss_penalty_kg * new_misses)
        info = dict(carbon_kg=carbon, energy_kwh=energy_kwh, moer=moer,
                    busy_gpus=busy, num_started=len(started),
                    queue_len=len(self.queue), new_misses=new_misses)
        return self._obs(), float(reward), False, truncated, info

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self._n_obs, dtype=np.float32)
        t = min(self.t, self.horizon - 1)
        frac = (self.t % STEPS_PER_DAY) / STEPS_PER_DAY
        obs[0] = np.sin(2 * np.pi * frac)
        obs[1] = np.cos(2 * np.pi * frac)
        obs[2] = self.t / self.horizon
        obs[3] = self.free_gpus / self.num_gpus
        obs[4] = min(1.0, max(0, len(self.queue) - self.queue_window) / 100)
        obs[5] = self._moer[t]
        obs[6:GLOBAL_FEATURES] = self._forecast[t]
        for i, job in enumerate(self.queue_window_jobs()):
            b = GLOBAL_FEATURES + SLOT_FEATURES * i
            obs[b] = 1.0
            obs[b + 1] = job.num_gpus / self.num_gpus
            obs[b + 2] = min(1.0, job.duration / self._max_duration)
            obs[b + 3] = np.clip((job.latest_start - self.t) / self._max_slack, -1, 1)
            obs[b + 4] = min(1.0, self._job_power[job.workload] / self.tdp)
        return obs

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
