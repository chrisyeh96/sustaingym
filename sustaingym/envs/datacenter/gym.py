from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from sustaingym.envs.datacenter.cluster import Cluster
from sustaingym.envs.datacenter.task import Task

HOURS_PER_DAY = 24
MICROSEC_PER_HOUR = 60*60*1000000


class DatacenterGym(gym.Env):
    """
    Attributes:
        # attributes required by gym.Env
        action_space: spaces.Box, structure of actions expected by env
        observation_space: spaces.Dict, structure of observations returned by env
        reward_range: tuple[float, float], min and max rewards
        spec: EnvSpec, info used to initialize env from gymnasium.make()
        metadata: dict[str, Any], unused
        np_random: np.random.Generator, random number generator for the env

        # attributes specific to DatacenterGym
        datacenter: Cluster, represents a datacenter cluster
        task_data: pd.DataFrame, TODO
    """
    TASK_DATA_PATH = "sustaingym/data/datacenter/daily_events"
    EPISODE_LEN = 672  # In hours
    START_DELAY = 600  # trace period starts at 600 seconds
    START_DELAY_H = 600 / 3600  # measured in hours
    PRIORITY_THRESH = 120  # priority values geq are considered inflexible

    def __init__(self, env_config: dict,
                 balancing_authority: str = 'SGIP_CAISO_PGE'):
        """
        Args:
            env_config: TODO, describe what this is
            balancing_authority: where to get MOER values from
        """
        self.datacenter = Cluster(self.EPISODE_LEN,
                                  env_config["sim_start_time"],
                                  env_config["sim_end_time"],
                                  balancing_authority)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            'VCC':              gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            'capacity':         gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            'num_queued_tasks': gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32),
            'moers':            gym.spaces.Box(0, np.inf, shape=(24,), dtype=np.float32),
            # avg_task_duration
            # avg_task_priority
            # avg_task_capacity
        })

        self.task_data: pd.DataFrame | None = None
        self.time_window = MICROSEC_PER_HOUR

    def make(self):
        # TODO
        return

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None
              ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.datacenter.t = 0
        self.datacenter.daily_capacity_req = [0 for _ in range(31)]  # 31 days
        obs = self.datacenter.get_state()
        info: dict[str, Any] = {}
        return obs, info

    def render(self):
        print(self.datacenter.get_state())

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert action.shape == self.action_space.shape  # (1,)
        VCC = action.item()  # only care about scalar

        self.datacenter.stop_finished_tasks()

        self.datacenter.set_VCC(VCC)

        new_tasks = self.get_new_tasks()
        self.datacenter.enqueue_tasks(new_tasks)
        self.datacenter.schedule_tasks()

        obs = self.datacenter.get_state()
        reward = self.compute_reward()

        self.datacenter.t += 1

        terminated = (self.datacenter.t >= self.EPISODE_LEN)
        truncated = False
        info: dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def close(self):
        # TODO
        return super().close()

    def compute_reward(self) -> float:
        """Times -1 to make it reward"""
        SLO_violation_cost = self.datacenter.compute_SLO_violation_cost()
        carbon_cost = self.datacenter.compute_carbon_cost()

        return -1*(SLO_violation_cost + carbon_cost)

    def update_daily_capacity_req(self, task_start_time: int, task: Task) -> None:
        """
        TODO document
        """
        curr_day = task_start_time // HOURS_PER_DAY
        curr_hour_of_day = task_start_time % HOURS_PER_DAY
        hours_remaining_in_day = (HOURS_PER_DAY - curr_hour_of_day)

        if task.duration > hours_remaining_in_day:
            curr_day_capacity_req = task.capacity * hours_remaining_in_day
            self.datacenter.add_daily_capacity_req(curr_day, curr_day_capacity_req)

            # count the number of days in the future with an entire day's full worth of capacity
            task_duration_after_today = task.duration - hours_remaining_in_day
            entire_days = task_duration_after_today // HOURS_PER_DAY
            for future_day in range(curr_day + 1, curr_day + 1 + entire_days):
                self.datacenter.add_daily_capacity_req(future_day, task.capacity * HOURS_PER_DAY)
            # add the leftover capacity in the last day
            task_duration_in_last_day = task_duration_after_today % HOURS_PER_DAY
            self.datacenter.add_daily_capacity_req(curr_day + entire_days + 1, task.capacity * task_duration_in_last_day)
        else:
            self.datacenter.add_daily_capacity_req(curr_day, task.capacity * task.duration)

    def get_new_tasks(self) -> list[Task]:
        """
        TODO: document
        """
        curr_t = self.datacenter.t

        if curr_t % HOURS_PER_DAY == 0:
            curr_d = curr_t // HOURS_PER_DAY
            self.task_data = pd.read_csv(f"{self.TASK_DATA_PATH}/day_{curr_d}.csv")
        assert self.task_data is not None

        start = (curr_t + self.START_DELAY_H) * MICROSEC_PER_HOUR
        end = (curr_t + self.START_DELAY_H + 1) * MICROSEC_PER_HOUR

        tasks = []
        new_task_data = self.task_data[(start <= self.task_data['time']) & (self.task_data['time'] < end)]
        for _, row in new_task_data.iterrows():
            task_duration = (row['duration'] // self.time_window) + 1  # calculate number of timesteps tasks lasts
            new_task = Task(row['task_id'], task_duration, row['cpu'])
            if row["priority"] < self.PRIORITY_THRESH:  # it's a flexible task
                tasks.append(new_task)
            self.update_daily_capacity_req(curr_t, new_task)

        return tasks
