from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    TASK_FINISHED = 1


class Event:
    """An element in an 'EventQueue'."""
    def __init__(self, type, machine_id: str, task_id):
        self.t = 0
        self.type = type
        self.machine_id = machine_id
        self.task_id = task_id


@dataclass(order=True)
class EventQueueEntry:
    end_time: int
    event: Any = field(compare=False)
