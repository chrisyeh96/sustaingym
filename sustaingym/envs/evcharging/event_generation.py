"""
TODO
"""
from __future__ import annotations

from collections.abc import Set, Generator

from datetime import datetime, timedelta
import pytz

from acnportal.acndata.data_client import DataClient
from acnportal.acndata.utils import parse_http_date 
from acnportal.acnsim.events import PluginEvent, RecomputeEvent, EventQueue
from acnportal.acnsim.models.battery import Battery, Linear2StageBattery
from acnportal.acnsim.models.ev import EV


MINS_IN_DAY = 1440
DT_STRING_FORMAT = "%a, %d %b %Y 7:00:00 GMT"
API_TOKEN = "DEMO_TOKEN"


def datetime_to_timestamp(dt: datetime, period: int) -> int:
    """
    Helper function to get_sessions. Returns simulation timestamp of datetime.
    """
    return (dt.hour * 60 + dt.minute) // period


def get_sessions(start_date: datetime,
                 end_date: datetime,
                 site: str ="caltech", return_count=True) -> Generator[dict]:
    """
    Retrieves charging sessions from site between start_date and end_date using
    ACNData Python API. 

    Args:
    start_date (datetime): beginning time of interval
    end_date (datetime): ending time of interval
    site (str): 'caltech' or 'jpl'
    return_count (bool) - whether to return count as well

    Returns:
    sessions - generator of sessions that had a connection time starting on
    `start_date` and ending the day before `end_date`
    count (bool) - also returns count if return_count is True

    Notes:
    For `start_date` and `end_date` arguments, only year, month, and day are
    considered.

    Ex. get_sessions(datetime(2020, 8, 1), datetime(2020, 11, 1)).
    """
    start_time = start_date.strftime(DT_STRING_FORMAT)
    end_time = end_date.strftime(DT_STRING_FORMAT)

    cond = f'connectionTime>="{start_time}" and connectionTime<="{end_time}"'
    data_client = DataClient(api_token=API_TOKEN)
    sessions = data_client.get_sessions(site, cond=cond)

    if return_count:
        count = data_client.count_sessions(site, cond=cond)
        return sessions, int(count)
    else:
        return sessions


def get_real_event_queue(day: datetime, period: int, recompute_freq: int,
                         station_ids: Set[str], site: str,
                         use_unclaimed: bool=False) -> EventQueue:
    """
    Creates an ACN Simulator event queue from real traces in ACNData. Ignores
    unclaimed user sessions.

    Arguments:
    day (datetime): date of collection, only year, month, and day are
        considered.
    period (int): number of minutes of each time interval in simulation
    recompute_freq (int): number of periods for recurring recompute
    station_ids (set): station ids in charging network to compare with
    site (string): either 'caltech' or 'jpl' garage to get events from
    use_unclaimed (boolean): whether to use unclaimed data. If True
        - Battery init_charge: 100 - session['kWhDelivered']
        - estimated_departure: disconnect timestamp
    
    Returns:
    event_queue (EventQueue): event queue for simulation

    Assumes:
    sessions are in Pacific time
    """
    sessions = get_sessions(day, day+timedelta(days=1.0), site=site, return_count=False)

    events = []

    for session in sessions:
        userInputs = session['userInputs']
        if not use_unclaimed and (
            not userInputs or not type(userInputs) == list or len(userInputs) == 0):
            continue

        connection_timestamp = datetime_to_timestamp(session['connectionTime'],
                                                     period)
        disconnect_timestamp = datetime_to_timestamp(session['disconnectTime'],
                                                     period)
        # Discard sessions with disconnect - connection <= 0, which occurs when
        # EV stays overnight
        if disconnect_timestamp - connection_timestamp <= 0:
            continue

        # Discard sessions with station id's not in dataset 
        if session['spaceID'] not in station_ids:
            continue

        if use_unclaimed and not userInputs:
            est_depart_timestamp = disconnect_timestamp
            requested_energy = session['kWhDelivered']
        else:
            est_depart_time = userInputs[0]['requestedDeparture']
            est_depart_dt = parse_http_date(est_depart_time,
                                            pytz.timezone('US/Pacific'))
            est_depart_timestamp = datetime_to_timestamp(
                                    est_depart_dt,
                                    period)
            requested_energy = userInputs[0]['kWhRequested']
            

        battery = Battery(capacity=100,
                          init_charge=100-requested_energy,
                          max_power=100)

        ev = EV(
            arrival=connection_timestamp,
            departure=disconnect_timestamp,
            requested_energy=requested_energy,
            station_id=session['spaceID'],
            session_id=session['sessionID'],
            battery=battery,
            estimated_departure=est_depart_timestamp
        )

        event = PluginEvent(connection_timestamp, ev)
        # no need for UnplugEvent as the simulator takes care of it
        events.append(event)
    
    # add recompute event every every `recompute_freq` periods
    for i in range(MINS_IN_DAY // (period * recompute_freq) + 1):
        event = RecomputeEvent(i * recompute_freq)
        events.append(event)
    
    events = EventQueue(events)
    return events
