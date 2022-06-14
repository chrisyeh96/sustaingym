import pytz
from datetime import datetime

import matplotlib.pyplot as plt

from acnportal import acnsim
from acnportal import algorithms


def generate_events():
    """Copied from https://github.com/zach401/acnportal/blob/master/tutorials/lesson1_running_an_experiment.ipynb. """
    # Timezone of the ACN we are using.
    timezone = pytz.timezone('America/Los_Angeles')

    # Start and End times are used when collecting data.
    start = timezone.localize(datetime(2018, 9, 5))
    end = timezone.localize(datetime(2018, 9, 6))

    # How long each time discrete time interval in the simulation should be.
    period = 5  # minutes

    # Voltage of the network.
    voltage = 220  # volts

    # Default maximum charging rate for each EV battery.
    default_battery_power = 32 * voltage / 1000 # kW

    # Identifier of the site where data will be gathered.
    site = 'caltech'

    # For this experiment we use the predefined CaltechACN network.
    cn = acnsim.sites.caltech_acn(basic_evse=True, voltage=voltage)

    API_KEY = 'DEMO_TOKEN'
    events = acnsim.acndata_events.generate_events(API_KEY, site, start, end, period, voltage, default_battery_power)

    return events