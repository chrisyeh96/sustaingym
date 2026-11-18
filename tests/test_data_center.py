from __future__ import annotations

import os
import tempfile
from typing import Any
import unittest

import gymnasium as gym
import gymnasium.utils.env_checker
import numpy as np
import pandas as pd

import sustaingym  # noqa: F401  (registers the environments)
from sustaingym.algorithms.data_center import (
    ASAP, BASELINES, DataCenterPolicy, ForecastWindow, alap_plan, best_offset)
from sustaingym.envs.data_center import (
    DataCenterEnv, Job, JobGenerator, SyntheticJobGenerator, TraceJobGenerator,
    make_power_model)
from sustaingym.envs.data_center.env import GLOBAL_FEATURES, SLOT_FEATURES
from sustaingym.envs.data_center.jobs import STEP_MINUTES, STEPS_PER_DAY, STEPS_PER_HOUR


class FixedJobs(JobGenerator):
    """Returns the same hand-written jobs every episode."""
    def __init__(self, jobs: list[tuple[int, int, int, str, int]]):
        super().__init__(max_slack_hours=1, workload_rule='mix',
                         mix={'W4': 1.0})
        self.spec = jobs

    def generate(self, day: pd.Timestamp, num_gpus: int,
                 rng: np.random.Generator) -> list[Job]:
        return [Job(job_id=i, arrival=a, duration=d, num_gpus=g, workload=w,
                    deadline=dl) for i, (a, d, g, w, dl) in enumerate(self.spec)]


class WaitAsLongAsAllowed(DataCenterPolicy):
    """Holds every job back until it has to start."""
    name = 'wait'

    def start_now(self, job: Job, latest_start: int) -> bool:
        return False


def small_env(**kwargs: Any) -> DataCenterEnv:
    kwargs.setdefault('num_gpus', 16)
    kwargs.setdefault('date_range', ('2020-06-01', '2020-06-10'))
    return DataCenterEnv(**kwargs)


class TestDataCenterEnv(unittest.TestCase):
    def test_check_env(self) -> None:
        env = small_env()
        gymnasium.utils.env_checker.check_env(env, skip_render_check=True)

    def test_gym_make(self) -> None:
        env = gym.make('sustaingym/DataCenter-v0')
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)

    def test_same_seed_same_episode(self) -> None:
        env = small_env()
        a = ForecastWindow(env).run_episode(seed=7)
        schedule = env.schedule()
        b = ForecastWindow(env).run_episode(seed=7)
        self.assertEqual(a, b)
        self.assertEqual(schedule, env.schedule())

    def test_all_jobs_finish_with_baselines(self) -> None:
        env = small_env()
        for policy in BASELINES:
            for safe in (False, True):
                summary = policy(env, deadline_safe=safe).run_episode(seed=3)
                self.assertEqual(summary['finished'], summary['jobs'])

    def test_schedule_does_not_depend_on_power_model(self) -> None:
        # the baselines never look at job power
        for policy, safe in [(ASAP, False), (ForecastWindow, True)]:
            schedules = set()
            for name in ('tdp', 'tdp_idle40', 'measured', 'measured_released'):
                env = small_env(power_model=name)
                policy(env, deadline_safe=safe).run_episode(seed=1, day='2020-06-03')
                schedules.add(tuple(env.schedule()))
            self.assertEqual(len(schedules), 1)

    def test_energy_follows_power_model(self) -> None:
        summaries = {}
        for name in ('tdp', 'tdp_idle40', 'measured', 'measured_released'):
            env = small_env(power_model=name)
            summaries[name] = ASAP(env).run_episode(seed=1, day='2020-06-03')
        gpu_steps = sum(j.num_gpus * j.duration for j in env.jobs)
        idle_steps = env.num_gpus * env.horizon - gpu_steps
        dt_h = STEP_MINUTES / 60
        tdp = make_power_model('tdp').tdp_w(env.gpu)
        idle = make_power_model('measured').idle_w(env.gpu)
        self.assertAlmostEqual(summaries['tdp']['energy_kwh'],
                               gpu_steps * tdp * dt_h / 1000, places=6)
        self.assertAlmostEqual(summaries['tdp_idle40']['energy_kwh'],
                               (gpu_steps + 0.4 * idle_steps) * tdp * dt_h / 1000, places=6)
        self.assertAlmostEqual(summaries['measured']['idle_kwh'],
                               idle_steps * idle * dt_h / 1000, places=6)
        self.assertEqual(summaries['measured_released']['idle_kwh'], 0.0)
        self.assertGreater(summaries['measured_released']['start_kwh'], 0.0)
        for s in summaries.values():
            parts = s['active_carbon_kg'] + s['idle_carbon_kg'] + s['start_carbon_kg']
            self.assertAlmostEqual(parts, s['carbon_kg'], places=9)

    def test_deadline_miss_when_cluster_full(self) -> None:
        # Two 4-GPU jobs on a 4-GPU cluster. The second cannot start until
        # the first finishes, so it misses its deadline by 6 steps.
        jobs = [(0, 12, 4, 'W4', 12), (0, 6, 4, 'W4', 12)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        summary = ASAP(env).run_episode(seed=0)
        self.assertEqual(summary['misses'], 1)
        self.assertEqual(summary['finished'], 2)

    def test_deadline_safe_avoids_capacity_miss(self) -> None:
        # Two 4-GPU jobs of 6 steps on a 4-GPU cluster, both due at step 12.
        # Waiting until each job's own latest start (step 6) misses one.
        # The capacity plan starts one now and the other at step 6.
        jobs = [(0, 6, 4, 'W4', 12), (0, 6, 4, 'W4', 12)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        self.assertEqual(WaitAsLongAsAllowed(env).run_episode(seed=0)['misses'], 1)
        policy = WaitAsLongAsAllowed(env, deadline_safe=True)
        self.assertEqual(policy.name, 'wait_safe')
        self.assertEqual(policy.run_episode(seed=0)['misses'], 0)
        self.assertEqual(sorted(start for _, start in env.schedule()), [0, 6])

    def test_deadline_safe_blocks_early_start_that_breaks_plan(self) -> None:
        # Job 0 (1 GPU, 8 steps, latest start 5) comes first in the window.
        # Job 1 (4 GPUs, 4 steps) must finish by step 10. Plain ASAP starts
        # job 0 at once, so job 1 waits until step 8 and misses. The safe
        # ASAP sees that starting job 0 now leaves no room for job 1, starts
        # job 1 first and job 0 when job 1 is done.
        jobs = [(0, 8, 1, 'W4', 13), (0, 4, 4, 'W4', 10)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        self.assertEqual(ASAP(env).run_episode(seed=0)['misses'], 1)
        self.assertEqual(ASAP(env, deadline_safe=True).run_episode(seed=0)['misses'], 0)
        self.assertEqual(env.schedule(), [(0, 4), (1, 0)])

    def test_deadline_safe_starts_overdue_job_when_gpus_free(self) -> None:
        # Job 0 fills the 2-GPU cluster until step 5, so job 1 (latest start
        # 1) cannot make its deadline. The plan marks it late, and the safe
        # policy starts it the moment GPUs come free instead of waiting.
        jobs = [(0, 5, 2, 'W4', 5), (0, 3, 1, 'W4', 4)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=2)
        summary = WaitAsLongAsAllowed(env, deadline_safe=True).run_episode(seed=0)
        self.assertEqual(env.schedule(), [(0, 0), (1, 5)])
        self.assertEqual(summary['misses'], 1)

    def test_misses_are_counted_when_the_deadline_comes(self) -> None:
        # Never-started jobs: one due at step 5, one due at the last step.
        # Each miss is counted at the step its deadline arrives.
        jobs = [(0, 3, 1, 'W4', 5), (0, 3, 1, 'W4', 3 * STEPS_PER_DAY)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=1)
        env.reset(seed=0)
        truncated = False
        total = 0.0
        miss_steps = []
        while not truncated:
            _, reward, _, truncated, info = env.step(np.zeros(env.queue_window))
            total += reward
            miss_steps += [env.t] * info['new_misses']
        self.assertEqual(miss_steps, [5, env.horizon])
        self.assertEqual(env.episode_summary()['misses'], 2)
        self.assertLess(total, -2 * env.miss_penalty_kg + 1e-9)

    def test_rejects_deadlines_after_the_horizon(self) -> None:
        with self.assertRaises(ValueError):
            small_env(job_generator=SyntheticJobGenerator(max_slack_hours=48),
                      horizon_days=2)
        env = small_env(job_generator=FixedJobs([(0, 3, 1, 'W4', 4 * STEPS_PER_DAY)]))
        with self.assertRaises(ValueError):
            env.reset(seed=0)

    def test_date_range_at_the_edges_of_the_moer_data(self) -> None:
        # The bundled MOER data run from May 2019 to August 2021. Days at the
        # edges load, and days whose horizon runs past the data are dropped.
        first = small_env(date_range=('2019-05-01', '2019-05-10'))
        self.assertLessEqual(first.days[-1], pd.Timestamp('2019-05-10'))
        last = small_env(date_range=('2021-08-20', '2021-08-31'))
        self.assertGreaterEqual(last.days[0], pd.Timestamp('2021-08-20'))
        self.assertLess(last.days[-1], pd.Timestamp('2021-08-31'))

    def test_tdp_model_needs_only_the_gpu(self) -> None:
        # W6 was not measured on the T4. TDP pricing does not need it, the
        # measured model does.
        env = small_env(gpu='T4', power_model='tdp')
        self.assertEqual(env.tdp, make_power_model('tdp').tdp_w('T4'))
        with self.assertRaises(ValueError):
            small_env(gpu='T4', power_model='measured')

    def test_observation_flags_window_slots(self) -> None:
        jobs = [(0, 3, 1, 'W4', 100), (0, 3, 2, 'W4', 50)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4, queue_window=3)
        obs, _ = env.reset(seed=0)
        slots = obs[GLOBAL_FEATURES:].reshape(3, SLOT_FEATURES)
        np.testing.assert_array_equal(slots[:, 0], [1, 1, 0])
        # the job with the earlier latest start is first
        self.assertAlmostEqual(slots[0, 1], 2 / 4)

    def test_duration_feature_ignores_later_jobs(self) -> None:
        # a long job arriving later must not change how the first job's
        # duration is scaled at step 0
        firsts = []
        for later in ([], [(100, 140, 1, 'W4', 400)]):
            gen = FixedJobs([(0, 12, 1, 'W4', 100)] + later)
            gen.max_duration_steps = 144
            env = small_env(job_generator=gen, num_gpus=4)
            obs, _ = env.reset(seed=0)
            firsts.append(obs[GLOBAL_FEATURES + 2])
        self.assertEqual(firsts[0], firsts[1])
        self.assertAlmostEqual(firsts[0], 12 / 144, places=6)


class TestBestOffset(unittest.TestCase):
    def test_picks_lowest_window(self) -> None:
        series = np.array([5.0, 4.0, 1.0, 1.0, 6.0])
        self.assertEqual(best_offset(series, 2, 3), 2)
        self.assertEqual(best_offset(series, 2, 1), 1)
        self.assertEqual(best_offset(series, 2, 0), 0)

    def test_ties_go_to_earliest(self) -> None:
        self.assertEqual(best_offset(np.ones(10), 3, 5), 0)


class TestAlapPlan(unittest.TestCase):
    def test_places_jobs_as_late_as_capacity_allows(self) -> None:
        jobs = [(0, 6, 4, 'W4', 12), (0, 6, 4, 'W4', 12), (0, 3, 2, 'W4', 30)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        env.reset(seed=0)
        planned, occ, late = alap_plan(env)
        self.assertEqual(planned, {0: 6, 1: 0, 2: 27})
        self.assertLessEqual(occ.max(), 4)
        self.assertEqual(late, set())
        # keeping 2 GPUs free: job 2 still fits at its latest start, the
        # 4-GPU jobs fall back to the full cluster
        planned, _, _ = alap_plan(env, reserve_gpus=2)
        self.assertEqual(planned, {0: 6, 1: 0, 2: 27})

    def test_late_job_takes_no_capacity(self) -> None:
        # Job 0 holds 3 of 4 GPUs until step 10, so the 4-GPU job 1 (latest
        # start 5) cannot fit and is marked late. Job 1 has the later deadline
        # and is placed first; had it taken GPUs, the 1-GPU job 2 would not
        # fit beside job 0 at its latest start, step 4.
        jobs = [(0, 10, 3, 'W4', 10), (0, 4, 4, 'W4', 9), (0, 4, 1, 'W4', 8)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        env.reset(seed=0)
        env.step(np.array([1] + [0] * (env.queue_window - 1)))  # start job 0
        planned, occ, late = alap_plan(env)
        self.assertEqual(late, {1})
        self.assertEqual(planned[2], 4)
        self.assertEqual(int(occ[4:8].max()), 4)

    def test_reserve_moves_small_jobs_earlier(self) -> None:
        jobs = [(0, 4, 2, 'W4', 10), (0, 4, 2, 'W4', 10)]
        env = small_env(job_generator=FixedJobs(jobs), num_gpus=4)
        env.reset(seed=0)
        self.assertEqual(alap_plan(env)[0], {0: 6, 1: 6})
        self.assertEqual(alap_plan(env, reserve_gpus=2)[0], {0: 6, 1: 2})


class TestSyntheticJobGenerator(unittest.TestCase):
    def test_small_cluster_gets_the_offered_load(self) -> None:
        # The default sizes include 8-GPU jobs. On a 4-GPU cluster none are
        # drawn, and the requested GPU-hours still match the offered load.
        gen = SyntheticJobGenerator(offered_load=0.6)
        rng = np.random.default_rng(0)
        loads = []
        for _ in range(200):
            jobs = gen.generate(pd.Timestamp('2020-06-01'), 4, rng)
            self.assertTrue(all(j.num_gpus <= 4 for j in jobs))
            loads.append(sum(j.duration * j.num_gpus for j in jobs) / (4 * STEPS_PER_DAY))
        self.assertAlmostEqual(float(np.mean(loads)), 0.6, delta=0.06)


class TestTraceJobGenerator(unittest.TestCase):
    def test_replays_one_day(self) -> None:
        rows = pd.DataFrame({
            'submit_time': ['2020-06-02 00:07:00', '2020-06-02 13:00:00',
                            '2020-06-02 13:00:00', '2020-06-03 01:00:00'],
            'duration_s': [600, 3 * 3600, 100, 600],
            'num_gpus': [1, 8, 1, 2],
        })
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'trace.csv')
            rows.to_csv(path, index=False)
            gen = TraceJobGenerator(path, offered_load=1.0)
        self.assertEqual([str(x.date()) for x in gen.available_days()],
                         ['2020-06-02', '2020-06-03'])
        jobs = gen.generate(pd.Timestamp('2020-06-02'), 16, np.random.default_rng(0))
        # the 100 s job is dropped (shorter than one step)
        self.assertEqual([(j.arrival, j.duration, j.num_gpus) for j in jobs],
                         [(1, 2, 1), (13 * STEPS_PER_HOUR, 3 * STEPS_PER_HOUR, 8)])


if __name__ == '__main__':
    unittest.main()
