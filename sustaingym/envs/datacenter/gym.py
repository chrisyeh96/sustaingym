from sustaingym.envs.datacenter.cluster import *
import pandas as pd
import gymnasium as gym


TASK_DATA_PATH = "sustaingym/data/datacenter/daily_events"
SIMULATION_LENGTH = 672  # In hours
HOURS_PER_DAY = 24
MICROSEC_PER_HOUR = 60*60*1000000
START_DELAY = 600  # trace period starts at 600 seconds
START_DELAY_H = 600 / 3600  # measured in hours
BALANCING_AUTHORITY = "SGIP_CAISO_PGE"
PRIORITY_THRESH = 120  # priority values geq are considered inflexible


class DatacenterGym(gym.Env):
    def __init__(self, env_config: dict):
        self.datacenter = Cluster(SIMULATION_LENGTH,
                                  env_config["sim_start_time"],
                                  env_config["sim_end_time"],
                                  BALANCING_AUTHORITY)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
        self.task_data = None
        self.time_window = MICROSEC_PER_HOUR
        self.episode_len = SIMULATION_LENGTH
    
    def make(self):
        # TODO
        return

    def reset(self, *, seed=None, options=None):
        super().reset()
        self.datacenter.t = 0
        self.datacenter.daily_capacity_req = [0 for _ in range(31)]  # 31 days
        obs = self.datacenter.get_state()
        info = {}
        return obs, info
    
    def render(self):
        print(self.datacenter.get_state())

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        VCC = action
        assert VCC.shape == self.action_space.shape  # (1,)

        VCC = VCC[0] # only care about scalar
        self.datacenter.stop_finished_tasks()

        self.datacenter.set_VCC(VCC)

        new_tasks = self.get_new_tasks()
        self.datacenter.enqueue_tasks(new_tasks)
        self.datacenter.schedule_tasks()

        obs = self.datacenter.get_state()
        reward = self.compute_reward()

        self.datacenter.t += 1

        terminated = self.datacenter.t >= self.episode_len
        truncated = False
        info = {}

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
            self.task_data = pd.read_csv(f"{TASK_DATA_PATH}/day_{curr_d}.csv")

        start = (curr_t + START_DELAY_H)*MICROSEC_PER_HOUR
        end = (curr_t + START_DELAY_H + 1)*MICROSEC_PER_HOUR

        tasks = []
        new_task_data = self.task_data[(start <= self.task_data['time']) & (self.task_data['time'] < end)]
        for _, row in new_task_data.iterrows():
            task_duration = (row['duration'] // self.time_window) + 1  # calculate number of timesteps tasks lasts
            new_task = Task(row['task_id'], task_duration, row['cpu'])
            if row["priority"] < PRIORITY_THRESH:  # it's a flexible task
                tasks.append(new_task)
            self.update_daily_capacity_req(curr_t, new_task)

        return tasks
