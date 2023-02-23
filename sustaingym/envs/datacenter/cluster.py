from collections import deque
from collections.abc import Sequence
from sustaingym.envs.datacenter.machine import *
from heapq import *
from sustaingym.envs.datacenter.util.event import *
import pandas as pd
from sustaingym.envs.datacenter.task import *
import random


HOURS_PER_DAY = 24
REMOVED_EVENT = 'R'
MACHINE_EVENTS_PATH = "sustaingym/data/datacenter/machine_events_sample.json"


class Cluster:
    def __init__(self, simulation_length: int):
        self.VCC = 0
        self.VCC_hist = [0 for _ in range(HOURS_PER_DAY)]  # keeps track of the VCC values in the day

        self.capacity = 0
        self.max_capacity = 0

        # daily_capacity_req[d] = "total daily capacity required by all tasks in day 'd'"
        self.daily_capacity_req = [0 for _ in range(simulation_length // HOURS_PER_DAY)]

        self.t = 0

        self.machines = self.init_machines()  # dict: machine_id -> Machine obj
        for machine in self.machines.values():
            self.max_capacity += machine.max_capacity

        self.event_q = []  # Min heap with end times as priority
        self.task_to_eq_entry = {}
        self.task_q = deque()  # Holds tasks before being scheduled
    
    def init_machines(self) -> dict[str, Machine]:
        machines = {}
        data = pd.read_json(MACHINE_EVENTS_PATH, lines=True)
        for i in range(len(data)):
            # TODO: consider adding machines only if event is ADD
            machine_id = data.iloc[i]['machine_id']
            max_capacity = data.iloc[i]['capacity']['cpus']
            machines[machine_id] = Machine(machine_id, max_capacity)
        return machines
    
    def set_VCC(self, new_VCC: float) -> None:
        """
        Sets current VCC and records it in day's history.

        If 'new_VCC' is less than current 'capacity', then evicts tasks.
        """
        self.VCC = new_VCC
        hour_of_day = self.t % HOURS_PER_DAY
        self.VCC_hist[hour_of_day] = new_VCC

        self.obey_vcc()
    
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
            if self.machines[machine_id].capacity + task.capacity <= self.machines[machine_id].max_capacity:
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
        while self.capacity > self.VCC:
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