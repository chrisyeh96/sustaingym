"""
Parse 'sample_instance_events.json'. Create two files:
    - 'inflexible_daily_load.txt': total daily load of inflexible load
    - 'flexible_task_durations.csv': colums are time,task_id,duration,cpu
"""

"""Calculate duration of each task"""

import pandas as pd
from enum import IntEnum


class DurationCounter:
    def __init__(self):
        self.duration = 0
        self.is_running = False
        self.initial_start_time = None
        self.latest_start_time = None


def initialize_counters(task_ids):
    task_id_2_DC = {}
    for task_id in task_ids:
        task_id_2_DC[task_id] = DurationCounter()
    return task_id_2_DC


class InstanceEvent(IntEnum):
    SUBMIT = 0
    QUEUE = 1
    ENABLE = 2
    SCHEDULE = 3
    EVICT = 4
    FAIL = 5
    FINISH = 6
    KILL = 7
    LOST = 8
    UPDATE_PENDING = 9
    UPDATE_RUNNING = 10


def is_start_event(event_type):
    return event_type == InstanceEvent.SCHEDULE


def is_pause_event(event_type):
    return event_type == InstanceEvent.QUEUE or event_type == InstanceEvent.EVICT


def is_end_event(event_type):
    return (event_type == InstanceEvent.FINISH or
            event_type == InstanceEvent.KILL or
            event_type == InstanceEvent.FAIL or
            event_type == InstanceEvent.LOST)


def save_data_to_csv(task_id_to_duration, task_id_to_cpu, task_id_to_priority):
    with open('sustaingym/data/datacenter/task_durations_W_PRIORITY.csv', 'w') as f:
        f.write("time,task_id,duration,cpu,priority\n")
        for task_id in task_id_to_duration:
            f.write("%d,%s,%d,%2f,%d\n"%(task_id_to_duration[task_id].initial_start_time,
                                         task_id,
                                         task_id_to_duration[task_id].duration,
                                         task_id_to_cpu[task_id],
                                         task_id_to_priority[task_id]))


def transform_features(df):
    """
    Transform column of dicts 'resource_request' into two separate columns 'cpus' and 'memory'.
    """
    print("Begin feature transformation...")
    cpus = []
    memory = []
    cpu_error_count = 0
    memory_error_count = 0
    for _, row in df.iterrows():
        try:
            cpus.append(row["resource_request"]["cpus"])
        except TypeError:  # float is not subscriptable, sometimes row["resource_request"] is nan
            cpus.append(None)
            cpu_error_count += 1
        try:
            memory.append(row["resource_request"]["memory"])
        except TypeError:
            memory.append(None)
            memory_error_count += 1
    print("ERROR REPORT: Transform features:")
    print(f"\t - CPU errors: {cpu_error_count}/{len(df)}={100*cpu_error_count/len(df)}%")
    print(f"\t - Memory errors: {memory_error_count}/{len(df)}={100*memory_error_count/len(df)}%")
    return cpus, memory


def calculate_task_duration(fname):
    df = pd.read_json(fname, lines=True)
    print("Loaded dataframe...")
    non_first_day = df[(df['time'] < 600*1000000) & (df['time'] > 87000*1000000)].index
    df.drop(non_first_day, inplace=True)

    # add new columns
    cpus, memory = transform_features(df)
    df["cpus"] = cpus
    df["memory"] = memory
    # drop old columns no longer needed after creating new features
    df.drop(columns=["resource_request"])

    df.sort_values(by=['time'], inplace=True)

    task_id_2_DC = initialize_counters(df.task_id.unique())
    task_id_to_cpu = dict()
    task_id_to_priority = dict()

    for index, row in df.iterrows():
        if index % (len(df) // 10) == 0:
            print("10% more")
        curr_tid = row["task_id"]
        curr_ev_type = row["type"]

        # record CPU usage
        if (not curr_tid in task_id_to_cpu):
            task_id_to_cpu[curr_tid] = row["cpus"]
        else:
            # TODO: why not (maybe weighted) average?
            pass

        # record task priority
        if not curr_tid in task_id_to_priority:
            task_id_to_priority[curr_tid] = row["priority"]
        else:
            #  if this doesn't hold true, maybe it holds that 
            assert task_id_to_priority[curr_tid] == row["priority"]

        # add duration
        if not task_id_2_DC[curr_tid].is_running:  # it either is paused or not started
            if not is_start_event(curr_ev_type):
                # remain in unstarted or paused status
                continue

            if task_id_2_DC[curr_tid].initial_start_time is None:  # it was previously unstarted
                task_id_2_DC[curr_tid].initial_start_time = row["time"]
            
            # Regardless if paused or not started, record current time and update status
            task_id_2_DC[curr_tid].latest_start_time = row["time"]
            task_id_2_DC[curr_tid].is_running = True
        else:
            if not (is_pause_event(curr_ev_type) or is_end_event(curr_ev_type)):
                # remain in running status
                continue
            assert not task_id_2_DC[curr_tid].initial_start_time is None
            task_id_2_DC[curr_tid].duration += (row["time"] - task_id_2_DC[curr_tid].latest_start_time)
            task_id_2_DC[curr_tid].is_running = False
    
    save_data_to_csv(task_id_2_DC, task_id_to_cpu, task_id_to_priority)


if __name__ == "__main__":
    calculate_task_duration("sustaingym/data/datacenter/preproc/sample_instance_events.json")
