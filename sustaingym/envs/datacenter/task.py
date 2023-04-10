class Task:
    def __init__(self, id, duration, capacity, priority=0):
        self.id = id 
        self.duration = duration
        self.capacity = capacity  # fraction of a machine's max_capacity
        self.priority = priority
        self.end_time = None  # this field is populated once task is ran
