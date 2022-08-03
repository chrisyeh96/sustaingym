"""Script to download data from ACNData."""
from datetime import datetime
import os

from sustaingym.envs.evcharging.utils import *


print(DEFAULT_DATE_RANGES)
print(DATE_FORMAT)

for start, end in DEFAULT_DATE_RANGES:
    for site in ['caltech', 'jpl']:
        start_dt = datetime.strptime(start, DATE_FORMAT)
        end_dt = datetime.strptime(end, DATE_FORMAT)
        df = fetch_real_events(start_dt, end_dt, site=site)

        fdir = os.path.join('sustaingym', 'data', 'acn_evcharging_data', site)
        os.makedirs(fdir, exist_ok=True)
        fname = f'{start} {end}.csv'
        fpath = os.path.join(fdir, fname)

        df.to_csv(fpath, compression='gzip', index=False)
