from random import randrange
from datetime import timedelta



def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    days_interval = delta.days + 1
    random_day = randrange(days_interval)
    return start + timedelta(random_day)