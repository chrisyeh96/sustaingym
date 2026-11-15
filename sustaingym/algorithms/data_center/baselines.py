"""Baseline policies for DataCenterEnv.

None of these policies uses job power, so each makes the same schedule under
every power model. By default a policy may hold a job until its own latest
start, even if the GPUs will be full then. With ``deadline_safe=True`` it
plans queued jobs against GPU capacity first; see `alap_plan`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from sustaingym.envs.data_center import DataCenterEnv, Job
from sustaingym.envs.data_center.env import FORECAST_STEPS


class DataCenterPolicy:
    """Base class: maps the environment's state to a start/wait action.

    Each policy has a short ``name`` for result tables, with ``_safe``
    appended when ``deadline_safe`` is on. Not a ``BaseAlgorithm``, because
    algorithms/base.py imports ray at module level.

    Args:
        env: the environment the policy acts in
        deadline_safe: hold a job only until its start in `alap_plan`, and
            start one early only if the rest of the plan still fits
        reserve_frac: with ``deadline_safe``, the fraction of GPUs the plan
            keeps free for jobs that have not arrived yet

    Attributes:
        env: DataCenterEnv, the environment the policy acts in
        deadline_safe: bool, whether jobs are planned against capacity
        reserve_gpus: int, GPUs the plan keeps free for future jobs
    """
    name = 'base'

    def __init__(self, env: DataCenterEnv, deadline_safe: bool = False,
                 reserve_frac: float = 0.0):
        if not 0.0 <= reserve_frac < 1.0:
            raise ValueError('reserve_frac must be in [0, 1)')
        self.env = env
        self.deadline_safe = deadline_safe
        self.reserve_gpus = int(round(reserve_frac * env.num_gpus))
        if deadline_safe:
            self.name = type(self).name + '_safe'

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Returns a MultiBinary action for the current queue window."""
        if self.deadline_safe:
            return self._safe_action()
        env = self.env
        action = np.zeros(env.queue_window, dtype=np.int8)
        for i, job in enumerate(env.queue_window_jobs()):
            if job.latest_start <= env.t or self.start_now(job, job.latest_start):
                action[i] = 1
        return action

    def _safe_action(self) -> np.ndarray:
        env, t = self.env, self.env.t
        action = np.zeros(env.queue_window, dtype=np.int8)
        planned, occ, late = alap_plan(env, self.reserve_gpus)
        cap = env.num_gpus - self.reserve_gpus
        free = env.free_gpus
        for i, job in enumerate(env.queue_window_jobs()):
            if job.num_gpus > free:
                continue
            s, d, g = planned[job.job_id], job.duration, job.num_gpus
            if job.job_id in late:
                occ[t:t + d] += g
            elif s > t:
                if not self.start_now(job, s):
                    continue
                # Starting early moves the job's GPUs from [s, s+d) to
                # [t, t+d). Allow it only if the plan still fits.
                trial = occ.copy()
                trial[s:s + d] -= g
                trial[t:t + d] += g
                if trial[t:t + d].max() > cap:
                    continue
                occ = trial
            action[i] = 1
            free -= g
        return action

    def start_now(self, job: Job, latest_start: int) -> bool:
        """Whether to start a job now that could wait until ``latest_start``."""
        raise NotImplementedError

    def run_episode(self, seed: int | None = None, day: str | None = None
                    ) -> dict[str, Any]:
        """Runs one episode and returns ``env.episode_summary()``."""
        options = None if day is None else {'day': day}
        obs, _ = self.env.reset(seed=seed, options=options)
        truncated = False
        while not truncated:
            obs, _, _, truncated, _ = self.env.step(self.get_action(obs))
        return self.env.episode_summary()


class ASAP(DataCenterPolicy):
    """Starts every job as soon as GPUs are free (carbon-unaware)."""
    name = 'asap'

    def start_now(self, job: Job, latest_start: int) -> bool:
        return True


class CarbonThreshold(DataCenterPolicy):
    """Starts a job when the current MOER is in the lowest ``quantile`` of
    the current value and the 3-hour forecast.

    Args:
        env: environment
        quantile: threshold quantile, between 0 and 1
        **kwargs: ``deadline_safe`` and ``reserve_frac``, see `DataCenterPolicy`

    Attributes:
        quantile: float, threshold quantile
    """
    name = 'threshold'

    def __init__(self, env: DataCenterEnv, quantile: float = 0.3, **kwargs: Any):
        super().__init__(env, **kwargs)
        self.quantile = quantile
        self._cache_key: tuple[Any, int] | None = None
        self._cache_ok = False

    def start_now(self, job: Job, latest_start: int) -> bool:
        # the answer only depends on this step's MOER, so compute it once per step
        key = (self.env.day, self.env.t)
        if self._cache_key != key:
            now = self.env.moer_now()
            series = np.concatenate([[now], self.env.moer_forecast()])
            self._cache_key = key
            self._cache_ok = bool(now <= np.quantile(series, self.quantile))
        return self._cache_ok


class ForecastWindow(DataCenterPolicy):
    """Starts a job now unless the forecast shows a lower-carbon start within
    the next 3 hours and before the job's latest start.

    The cost of a candidate start is the mean forecast MOER over the part of
    the job's run that the forecast covers.
    """
    name = 'forecast_window'

    def start_now(self, job: Job, latest_start: int) -> bool:
        series = np.concatenate([[self.env.moer_now()], self.env.moer_forecast()])
        return best_offset(series, job.duration,
                           min(latest_start - self.env.t, FORECAST_STEPS)) == 0


class OracleWindow(DataCenterPolicy):
    """Like `ForecastWindow`, but with the actual future MOER over the whole
    slack window. A reference only: no real scheduler knows the future MOER.
    """
    name = 'oracle_window'

    def start_now(self, job: Job, latest_start: int) -> bool:
        slack = latest_start - self.env.t
        series = self.env.moer_future(slack + job.duration)
        return best_offset(series, job.duration, slack) == 0


def best_offset(series: np.ndarray, duration: int, max_offset: int) -> int:
    """Start offset in [0, max_offset] with the lowest mean of ``series``
    over the job's run (truncated to the end of ``series``). Ties go to the
    earliest offset.
    """
    max_offset = max(0, min(max_offset, len(series) - 1))
    csum = np.concatenate([[0.0], np.cumsum(series, dtype=np.float64)])
    starts = np.arange(max_offset + 1)
    ends = np.minimum(starts + duration, len(series))
    means = (csum[ends] - csum[starts]) / (ends - starts)
    return int(np.argmin(means))


def alap_plan(env: DataCenterEnv, reserve_gpus: int = 0
              ) -> tuple[dict[int, int], np.ndarray, set[int]]:
    """Plans every queued job to start as late as capacity allows.

    Running jobs keep their GPUs until they finish. Queued jobs are placed
    latest deadline first, each at the latest start that meets its deadline
    with planned use at or below ``num_gpus - reserve_gpus``, or else within
    the full cluster. A job that fits neither is marked late and takes no
    planned capacity. Policies start late jobs as soon as GPUs are free, which
    can take GPUs an on-time job needed. The placement is greedy, so a better
    packing may fit a job it marks late.

    Args:
        env: environment, read at its current step
        reserve_gpus: GPUs the plan keeps free for jobs not yet arrived

    Returns:
        planned: planned start step of each queued job, by job id
        occ: planned GPUs in use at each step, indexed by absolute step
        late: ids of the jobs the plan could not place before their deadlines
    """
    t = env.t
    end = max([t + 1]
              + [j.start + j.duration for j in env.running]
              + [max(j.deadline, t + j.duration) for j in env.queue])
    occ = np.zeros(end + 1, dtype=np.int64)
    for job in env.running:
        occ[t:job.start + job.duration] += job.num_gpus
    caps = [env.num_gpus - reserve_gpus, env.num_gpus]
    planned: dict[int, int] = {}
    late: set[int] = set()
    for job in sorted(env.queue, key=lambda j: (-j.deadline, -j.num_gpus, j.job_id)):
        s = None
        for cap in caps:
            s = _latest_fit(occ, t, job, cap)
            if s is not None:
                break
        if s is None:
            planned[job.job_id] = t
            late.add(job.job_id)
            continue
        planned[job.job_id] = s
        occ[s:s + job.duration] += job.num_gpus
    return planned, occ, late


def _latest_fit(occ: np.ndarray, t: int, job: Job, cap: int) -> int | None:
    """Latest start in [t, job.latest_start] at which the job's GPUs fit under
    ``cap`` for its whole run, or None."""
    d = job.duration
    if job.latest_start < t or job.num_gpus > cap:
        return None
    over = occ[t:job.latest_start + d] + job.num_gpus > cap
    csum = np.concatenate([[0], np.cumsum(over)])
    fits = np.flatnonzero(csum[d:] == csum[:-d])
    return t + int(fits[-1]) if len(fits) else None


BASELINES = (ASAP, CarbonThreshold, ForecastWindow, OracleWindow)
