from sustaingym.envs.datacenter.task import Task
import random


class Machine:
    def __init__(self, id, max_capacity):
        self.id = id
        self.capacity = 0
        self.max_capacity = max_capacity
        self.tasks = dict()  # Task.id -> Task object
    
    def start_task(self, task: Task):
        assert self.capacity + task.capacity <= self.max_capacity
        self.capacity += task.capacity
        self.tasks[task.id] = task
    
    def stop_task(self, task_id) -> Task:
        """
        - Return: the stopped task object.
        """
        stopped_task = self.tasks[task_id]
        self.capacity -= stopped_task.capacity
        del self.tasks[task_id]
        return stopped_task

    def select_task_to_evict(self):
        """
        - Return: task ID of chosen task.
        """
        return random.choice(list(self.tasks.keys()))

    def evict(self) -> Task:
        """
        - Return the evicted task.
        """
        evicted_task_id = self.select_task_to_evict()
        evicted_task = self.stop_task(evicted_task_id)
        return evicted_task
