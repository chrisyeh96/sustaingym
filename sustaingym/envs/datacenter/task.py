class Task:
    def __init__(self, id, duration, capacity):
        self.id = id 
        self.duration = duration
        self.capacity = capacity
        self.end_time = None  # this field is populated once task is ran
        # TODO: priority
