"""Writes a DataCenterEnv job trace from the public Helios GPU cluster traces.

The Helios traces (SenseTime, April to September 2020) are released under
CC BY 4.0 at https://github.com/S-Lab-System-Group/HeliosData. If you use
them, cite: Hu et al., "Characterization and Prediction of Deep Learning
Workloads in Large-Scale GPU Datacenters", SC '21,
https://doi.org/10.1145/3458817.3476223.

The output CSV has one row per GPU job with columns ``submit_time`` (local
time as recorded in the trace), ``duration_s`` and ``num_gpus``. Pass it to
`sustaingym.envs.data_center.TraceJobGenerator`.

Usage::

    python -m sustaingym.data.data_center.prepare_helios --cluster Venus \
        --out helios_venus.csv.gz

Add ``--zip PATH`` to use an already downloaded ``data.zip``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile

import pandas as pd

DATA_URL = 'https://raw.githubusercontent.com/S-Lab-System-Group/HeliosData/master/data.zip'
DATA_SHA256 = '3d22a5f6c0ae669e2fcbfe4200fa9c48664507bc397c677bad8f085222c032ac'
CLUSTERS = ('Earth', 'Saturn', 'Uranus', 'Venus')


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_cluster(zip_path: str, cluster: str) -> pd.DataFrame:
    """Reads one cluster's GPU jobs from the Helios zip."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f'data/{cluster}/cluster_log.csv') as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'),
                             usecols=['gpu_num', 'submit_time', 'start_time',
                                      'end_time', 'duration'])
    df = df[(df['gpu_num'] >= 1) & df['start_time'].notna()
            & df['duration'].notna() & (df['duration'] > 0)]
    return pd.DataFrame({
        'submit_time': pd.to_datetime(df['submit_time']),
        'duration_s': df['duration'].astype(int),
        'num_gpus': df['gpu_num'].astype(int),
    }).sort_values('submit_time', kind='stable')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--cluster', choices=CLUSTERS, default='Venus')
    p.add_argument('--out', required=True, help='output .csv or .csv.gz')
    p.add_argument('--zip', help='path to a downloaded Helios data.zip')
    args = p.parse_args()

    zip_path = args.zip
    if zip_path is None:
        zip_path = os.path.join(tempfile.gettempdir(), 'helios_data.zip')
        if not os.path.exists(zip_path) or sha256(zip_path) != DATA_SHA256:
            print(f'downloading {DATA_URL} (36 MB)')
            urllib.request.urlretrieve(DATA_URL, zip_path)
    digest = sha256(zip_path)
    if digest != DATA_SHA256:
        raise SystemExit(f'{zip_path} has SHA-256 {digest}, expected {DATA_SHA256}')

    df = load_cluster(zip_path, args.cluster)
    df.to_csv(args.out, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print(f'wrote {len(df)} GPU jobs from {args.cluster} to {args.out}')


if __name__ == '__main__':
    main()
