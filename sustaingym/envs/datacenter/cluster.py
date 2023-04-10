from sustaingym.envs.datacenter.util.event import *
from sustaingym.data.load_moer import load_moer
from sustaingym.envs.datacenter.machine import *
from sustaingym.envs.datacenter.task import *

from datetime import datetime, timedelta, timezone
from collections.abc import Sequence
from collections import deque
import pandas as pd
from heapq import *
import numpy as np
import random


HOURS_PER_DAY = 24
REMOVED_EVENT = 'R'
MACHINE_EVENTS_PATH = "sustaingym/data/datacenter/machine_events_sample.json"
MOER_PATH = "sustaingym/data/moer/"
TZ_INFO = timezone(timedelta(0,0,0,0,0,0,0))  # corresponds to UTC
ONEDAY = timedelta(days=1)


class Cluster:
    def __init__(self, simulation_length: int, start_time: datetime,
                end_time: datetime, balancing_authority: str):
        self.VCC = 0  # fraction of self.max_capacity that can be used
        self.VCC_hist = [0 for _ in range(HOURS_PER_DAY)]  # VCC over last day

        self.capacity = 0
        self.max_capacity = 0  # sum of all machines' max_capacities

        self.carbon_intensities = self.get_carbon_data(start_time, end_time, balancing_authority)

        # self.daily_capacity_req[d] = "total daily capacity required by all tasks in day 'd'"
        self.daily_capacity_req = [0 for _ in range(31)]  # 31 days

        self.t = 0  # in hours

        self.machines = self.init_machines()  # dict: machine_id -> Machine obj
        for machine in self.machines.values():
            self.max_capacity += machine.max_capacity

        self.event_q = []  # Min heap with end times as priority
        self.task_to_eq_entry = {}
        self.task_q = deque()  # Holds tasks before being scheduled

    # *** INITIALIZATION & DATA LOADING ***
    def init_machines(self) -> dict[str, Machine]:
        """Returns mapping from machine_id to Machine object."""
        machines = {}
        data = pd.read_json(MACHINE_EVENTS_PATH, lines=True)
        for i in range(len(data)):
            # TODO: consider adding machines only if event is ADD
            machine_id = data.iloc[i]['machine_id']
            max_capacity = data.iloc[i]['capacity']['cpus']
            machines[machine_id] = Machine(machine_id, max_capacity)
        return machines
    
    def get_carbon_data(self, start_time: datetime, end_time: datetime,
                       balancing_authority: str):
        step_size = 60 // 5  # datacenter simulation is in hours, and MOER data every 5 mins
        data = load_moer(start_time, end_time, balancing_authority, MOER_PATH).values
        return data[::step_size, 0]  # column 0 has ground truth

        # curr_day_num = self.t // HOURS_PER_DAY
        # curr_datetime = self.sim_start_time + (curr_day_num)*ONEDAY

        # data = self.MOER_loader.retrieve(curr_datetime)
        # assert data.shape == (289, 37)

        # curr_day_MOER = data[:, 0]  # in five min timesteps

        # five_min_timesteps_per_hour = 12
        # return [curr_day_MOER[i] for i in range(0, 289, five_min_timesteps_per_hour)]

    # *** APIs ***
    def get_state(self):
        """
        Returns observation of the state of the Cluster
        """
        state = []

        state.append(self.VCC)
        state.append(self.capacity)
        state.append(len(self.task_q))
        state += list(self.carbon_intensities[self.t: self.t + 24])

        return np.array(state)
    
    def set_VCC(self, new_VCC: float) -> None:
        """
        Sets current VCC and records it in day's history.

        If 'new_VCC' is less than current 'capacity', then evicts tasks.
        """
        self.VCC = new_VCC
        hour_of_day = self.t % HOURS_PER_DAY
        self.VCC_hist[hour_of_day] = new_VCC

        self.obey_vcc()

    def compute_SLO_violation_cost(self) -> float:
        """
        if EOD
            r = max(0, 0.97*[day capacity required] - [sum of VCC-allocated capacity over previous day])
        otherwise
            r = 0
        """
        # Only computed at EOD
        if self.t % HOURS_PER_DAY != 23:
            return 0

        current_day = self.t // HOURS_PER_DAY
        capacity_requirement = self.daily_capacity_req[current_day]
        total_allocated_capacity = sum(self.VCC_hist) * self.max_capacity

        penalty = max(0, 0.97*capacity_requirement - total_allocated_capacity)
        self.VCC_hist = [0 for _ in range(HOURS_PER_DAY)]  # reset VCC history for next day
        
        return penalty

    def compute_carbon_cost(self):
        """
        ['Carbon cost at time t'] = [f(['capacity at time t'])] * ['carbon intensity at time t']
        """
        pow_usage = self.get_power_usage()
        carbon_intensity = self.carbon_intensities[self.t]

        return pow_usage * carbon_intensity

    # *** INTERNAL DATACENTER FUNCTIONALITY ***
    def select_machine_to_schedule(self):
        return random.choice(list(self.machines.keys()))
    
    def schedule_task(self, task: Task) -> bool:
        """
        Assigns 'task' to a machine for it to run on.

        Returns: boolean indicating if schedule was successful.
        """
        # find a machine and run task on it
        attempt_count = 0
        while True:
            machine_id = self.select_machine_to_schedule()
            if (self.machines[machine_id].capacity + task.capacity
                    <= self.machines[machine_id].max_capacity):
                self.machines[machine_id].start_task(task)
                break
            elif attempt_count > len(self.machines):
                # if no machine has capacity, simply re-enqueue task
                self.task_q.append(task)
                return False
            attempt_count += 1

        self.capacity += task.capacity

        task_end_time = self.t + task.duration
        task.end_time = task_end_time

        assert not machine_id is None
        event = Event(EventType.TASK_FINISHED, machine_id, task.id)

        eq_entry = EventQueueEntry(task_end_time, event)
        self.task_to_eq_entry[task] = eq_entry
        heappush(self.event_q, eq_entry)
        
        return True
    
    def schedule_tasks(self) -> None:
        """
        Dequeues tasks and schedules them as long as VCC isn't exceeded.
        """
        failed_schedule = False
        while True:
            # NOTE: if there is a huge task (in terms of capacity requirement) at the front of the queue, 
                # it will block potentially smaller tasks behind it from being scheduled.
            if (failed_schedule or
                len(self.task_q) == 0 or
                self.capacity + self.task_q[0].capacity > self.VCC):
                break
            task = self.task_q.popleft()
            failed_schedule = self.schedule_task(task)

    def stop_finished_tasks(self) -> None:
        """
        TODO: document
        """
        while True:
            if len(self.event_q) == 0:
                break

            eq_entry = self.event_q[0]
            end_time = eq_entry.end_time
            event = eq_entry.event
            if end_time > self.t:
                break
            elif event == REMOVED_EVENT:
                continue
            elif event.type == EventType.TASK_FINISHED:
                heappop(self.event_q)
                stopped_task = self.machines[event.machine_id].stop_task(event.task_id)
                self.capacity -= stopped_task.capacity
            else:
                raise Exception("UNKNOWN EVENT TYPE")

    def enqueue_tasks(self, tasks: Sequence[Task]) -> None:
        """
        Takes a list of all tasks submitted at time T and enqueues them.
        """
        self.task_q.extend(tasks)

    def select_machine_to_evict(self) -> str:
        while True:
            machine_id = random.choice(list(self.machines.keys()))
            if len(self.machines[machine_id].tasks) > 0:
                break
        return machine_id

    def obey_vcc(self) -> None:
        """
        It may be possible that the VCC is reduced, so that a capacity
        that was previously allowed is now above the maximum allowed
        capacity set by the new VCC. Evict until capacity falls below
        VCC.
        """
        while self.capacity > self.VCC*self.max_capacity:
            machine_id = self.select_machine_to_evict()
            evicted_task = self.machines[machine_id].evict()

            # mark the entry as invalid in the priority queue
            eq_entry = self.task_to_eq_entry[evicted_task]
            eq_entry.event = REMOVED_EVENT

            # re-enqueue evicted task
            prev_end_time = eq_entry.end_time
            remaining_duration = prev_end_time - self.t
            new_task = Task(evicted_task.id, remaining_duration, evicted_task.capacity)
            self.task_q.append(new_task)

    def add_daily_capacity_req(self, day, capacity):
        self.daily_capacity_req[day] += capacity

    def get_power_usage(self):
        """Compute power usage at time 't'."""
        proportionality_constant = 1
        return self.capacity * proportionality_constant
