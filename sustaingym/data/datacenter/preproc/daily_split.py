"""
- Split `sample_instance_events.json` and save events of each day to separate
  file located to directory `daily_events`
- Sort by time each of those files
- `transform_features` (i.e. separate cpu and memory columns)
"""


import pandas as pd


NUM_DAYS = 30
START_OFFSET = 600  # simulation starts at second 600
MICROSEC_PER_SEC = 1000000
SEC_PER_DAY = 24*60*60
SAMPLE_EVENTS_PATH = "sustaingym/data/datacenter/sample_instance_events.json"
SAVE_PATH = "sustaingym/data/datacenter/daily_events"


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
    for d in range(NUM_DAYS):
        print(f"LOG: day #{d}")
        start_time = (START_OFFSET + d*SEC_PER_DAY)*MICROSEC_PER_SEC
        end_time = (START_OFFSET + (d+1)*SEC_PER_DAY)*MICROSEC_PER_SEC
        with open(f'{SAVE_PATH}/day_{d}.csv', 'w') as f:
            f.write("time,task_id,duration,cpu,priority\n")
            for task_id in task_id_to_duration:
                init_start_time = task_id_to_duration[task_id].initial_start_time
                duration = task_id_to_duration[task_id].duration
                cpu = task_id_to_cpu[task_id]
                priority = task_id_to_priority[task_id]
                
                # # DEBUG
                # if init_start_time is None:
                #     print("INIT IS NONE")
                # if duration is None:
                #     print("duration is None")
                # if priority is None:
                #     print("PRIORITY IS NONE")

                if (init_start_time is None or  # no 'SCHEDULE' event
                    init_start_time == 0 or
                    duration == 0 or
                    cpu == 0 or
                    (init_start_time < start_time or init_start_time >= end_time)):
                    continue
                
                f.write("%d,%s,%d,%2f,%d\n"%(init_start_time,
                                            task_id,
                                            duration,
                                            cpu,
                                            priority))


def transform_features(df):
    """
    Transform column of dicts 'resource_request' into two separate columns 'cpus' and 'memory'.
    """
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


if __name__ == "__main__":
    print("LOG: begin")
    df = pd.read_json(SAMPLE_EVENTS_PATH, lines=True)
    print("LOG: loaded dataframe, transforming features.")

    cpus, memory = transform_features(df)
    df["cpus"] = cpus  # add new columns
    df["memory"] = memory
    df.drop(columns=["resource_request"], inplace=True)  # drop old column
    print(f"LOG: transformed features, sorting by time...")
    df.sort_values(by="time", inplace=True)
    print(f"LOG: sorting finished.")

    task_id_to_DC = initialize_counters(df.task_id.unique())
    task_id_to_cpu = {}
    task_id_to_priority = {}

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
        if not task_id_to_DC[curr_tid].is_running:  # it either is paused or not started
            if not is_start_event(curr_ev_type):
                # remain in unstarted or paused status
                continue

            if task_id_to_DC[curr_tid].initial_start_time is None:  # it was previously unstarted
                task_id_to_DC[curr_tid].initial_start_time = row["time"]
                # debug_initialized_tids.add(curr_tid)
            
            # Regardless if paused or not started, record current time and update status
            task_id_to_DC[curr_tid].latest_start_time = row["time"]
            task_id_to_DC[curr_tid].is_running = True
        else:
            if not (is_pause_event(curr_ev_type) or is_end_event(curr_ev_type)):
                # remain in running status
                continue
            assert not task_id_to_DC[curr_tid].initial_start_time is None
            task_id_to_DC[curr_tid].duration += (row["time"] - task_id_to_DC[curr_tid].latest_start_time)
            task_id_to_DC[curr_tid].is_running = False
    
        # DEBUG
        # if not len(debug_initialized_tids) == len(task_id_to_DC):
        #     print(f"initialized: {len(debug_initialized_tids)}")
        #     print(f"DC: {len(task_id_to_DC)}")
        # if not len(debug_initialized_tids) == len(df_curr_day.task_id.unique()):
        #     print(f"initialized: {len(debug_initialized_tids)}")
        #     print(f"curr_day: {len(df_curr_day.task_id.unique())}")

    print("LOG: saving to CSV")
    save_data_to_csv(task_id_to_DC, task_id_to_cpu, task_id_to_priority)
