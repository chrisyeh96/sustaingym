from datetime import datetime, timedelta
import pytz

from acnportal.acnsim.events import PluginEvent, RecomputeEvent, EventQueue
from acnportal.acnsim.models.battery import Battery, Linear2StageBattery
from acnportal.acnsim.models.ev import EV
from acnportal.acndata.data_client import DataClient
from acnportal.acndata.utils import parse_http_date 


MINS_IN_DAY = 1440


def datetime_to_timestamp(dt: datetime, period: int):
    """
    Helper function to get_sessions. Returns simulation timestamp of datetime.
    """
    return (dt.hour * 60 + dt.minute) // period


def get_sessions(start_date: datetime,
                 end_date: datetime,
                 site: str ="caltech"):
    """
    Retrieves charging sessions from site between start_date and end_date using
    ACNData Python API. 

    Args:
    start_date (datetime): beginning time of interval
    end_date (datetime): ending time of interval

    Returns:
    sessions - generator of sessions that had a connection time starting on
    `start_date` and ending the day before `end_date`

    Notes:
    For `start_date` and `end_date` arguments, only year, month, and day are
    considered.

    Ex. get_sessions(datetime(2020, 8, 1), datetime(2020, 11, 1)).
    """
    start_time = start_date.strftime("%a, %d %b %Y 7:00:00 GMT")
    end_time = end_date.strftime("%a, %d %b %Y 7:00:00 GMT")

    cond = f'connectionTime>="{start_time}" and connectionTime<="{end_time}"'

    api_token = "DEMO_TOKEN"
    data_client = DataClient(api_token=api_token)
    
    sessions = data_client.get_sessions(site, cond=cond)
    return sessions


def retrieve_sessions(day_date: datetime, site: str ="caltech"):
    """
    Retrieves sessions on a specific day from site.
    """
    return get_sessions(day_date, day_date + timedelta(days=1.0), site=site)


def get_real_event_queue(day_date, period, recompute_freq, station_ids, site="caltech"):
    """
    Creates an ACN Simulator event queue from real traces in ACNData. Ignores
    unclaimed user sessions.

    Arguments:
    day_date (datetime): date of collection, only year, month, and day are
        considered.
    period (int): number of minutes of each time interval in simulation
    recompute_freq (int): number of periods for recurring recompute
    station_ids (set): station ids in charging network to compare with
    
    Returns:
    event_queue

    Assumes:
    sessions are in Pacific time
    """
    sessions = retrieve_sessions(day_date, site=site)

    events = []

    for session in sessions:
        userInputs = session['userInputs']
        if not userInputs or not type(userInputs) == list or len(userInputs) == 0:
            continue

        battery = Battery(capacity=100,
                          init_charge=100-userInputs[0]['kWhRequested'],
                          max_power=100)

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

        est_depart_time = userInputs[0]['requestedDeparture']
        est_depart_dt = parse_http_date(est_depart_time,
                                        pytz.timezone('US/Pacific'))
        est_depart_timestamp = datetime_to_timestamp(
                                  est_depart_dt,
                                  period)

        ev = EV(
            arrival=connection_timestamp, # TODO process
            departure=disconnect_timestamp,
            requested_energy=userInputs[0]['kWhRequested'],
            station_id=session['spaceID'],
            session_id=session['sessionID'],
            battery=battery,
            estimated_departure=est_depart_timestamp
        )

        event = PluginEvent(connection_timestamp, ev)
        # no need for UnplugEvent as the simulator takes care of it
        events.append(event)
    
    for i in range(MINS_IN_DAY // (period * recompute_freq) + 1):
        event = RecomputeEvent(i * recompute_freq)
        events.append(event)
    
    events = EventQueue(events)
    return events