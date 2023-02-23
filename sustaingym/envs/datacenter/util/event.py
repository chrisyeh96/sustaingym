from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from sustaingym.envs.datacenter.task import *

class EventType(Enum):
    TASK_FINISHED = 1


class Event:
    """An element in an 'EventQueue'."""
    def __init__(self, type, machine_id, task_id):
        self.t = 0
        self.type = type
        self.machine_id = machine_id
        self.task_id = task_id


@dataclass(order=True)
class EventQueueEntry:
    end_time: int
    event: Any=field(compare=False)
