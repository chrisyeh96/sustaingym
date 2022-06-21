from datetime import timedelta, datetime
from random import randrange


def random_date(start, end) -> datetime:
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    days_interval = delta.days + 1
    random_day = randrange(days_interval)
    return start + timedelta(random_day)
