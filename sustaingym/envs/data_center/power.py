"""GPU power models for DataCenterEnv.

A power model gives the board power of one GPU running a workload, the power
of one idle GPU, and the energy charged when a job starts. Two families are
provided so that the same schedule can be scored both ways:

- `TDPPowerModel`: a busy GPU draws its TDP, an idle GPU draws a fixed
  fraction of TDP (0 by default).
- `MeasuredPowerModel`: a busy GPU draws the measured mean board power of
  its workload on that GPU, an idle GPU draws the measured reserved idle
  power. With ``idle_mode='released'`` idle GPUs are handed back (0 W) and
  every job start pays the measured cold-start energy instead.

Measured values come from ``data/data_center/gpu_power.csv``. See the README
in that folder for how they were measured and what they exclude.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import pandas as pd

from sustaingym.data.utils import read_csv

POWER_TABLE_PATH = 'data/data_center/gpu_power.csv'
POWER_MODEL_NAMES = ('tdp', 'tdp_idle40', 'measured', 'measured_released')


@lru_cache(maxsize=1)
def load_power_table() -> pd.DataFrame:
    """Returns the measured power table indexed by (gpu, workload)."""
    df = read_csv(POWER_TABLE_PATH)
    return df.set_index(['gpu', 'workload']).sort_index()


def gpu_names() -> list[str]:
    """GPU types in the measured power table."""
    return sorted(str(g) for g in load_power_table().index.get_level_values('gpu').unique())


class PowerModel:
    """Base class for per-GPU power models. All powers are in watts. Each
    model has a short ``name``, e.g. ``'measured'``.

    Attributes:
        table: pd.DataFrame, measured power table indexed by (gpu, workload)
    """
    name = 'base'

    def __init__(self) -> None:
        self.table = load_power_table()
        per_gpu = self.table.groupby(level='gpu').first()
        self._tdp = {str(g): float(v) for g, v in per_gpu['tdp_w'].items()}
        self._idle = {str(g): float(v) for g, v in per_gpu['idle_w'].items()}
        self._coldstart = {str(g): float(v) for g, v in per_gpu['coldstart_j'].items()}
        flat = self.table.reset_index()
        self._active = {(str(g), str(w)): float(v) for g, w, v in
                        zip(flat['gpu'], flat['workload'], flat['active_w'])}

    def tdp_w(self, gpu: str) -> float:
        """Thermal design power of the GPU."""
        return self._tdp[gpu]

    def active_w(self, gpu: str, workload: str) -> float:
        """Power of one GPU running ``workload``."""
        raise NotImplementedError

    def idle_w(self, gpu: str) -> float:
        """Power of one idle GPU."""
        raise NotImplementedError

    def start_j(self, gpu: str) -> float:
        """Energy charged per GPU when a job starts, in joules."""
        return 0.0

    def check(self, gpu: str, workloads: set[str]) -> None:
        """Raises ValueError if the table lacks ``gpu`` or a workload on it."""
        if gpu not in self._tdp:
            raise ValueError(f'GPU {gpu!r} not in power table; have {gpu_names()}')
        missing = {w for w in workloads if (gpu, w) not in self._active}
        if missing:
            raise ValueError(f'no measurements of {sorted(missing)} on {gpu}')


class TDPPowerModel(PowerModel):
    """Busy GPUs draw TDP. Idle GPUs draw ``idle_fraction`` times TDP.

    Args:
        idle_fraction: idle power as a fraction of TDP

    Attributes:
        idle_fraction: float, idle power as a fraction of TDP
    """
    def __init__(self, idle_fraction: float = 0.0):
        super().__init__()
        self.idle_fraction = idle_fraction
        self.name = 'tdp' if idle_fraction == 0 else f'tdp_idle{round(100 * idle_fraction)}'

    def active_w(self, gpu: str, workload: str) -> float:
        return self.tdp_w(gpu)

    def idle_w(self, gpu: str) -> float:
        return self.idle_fraction * self.tdp_w(gpu)

    def check(self, gpu: str, workloads: set[str]) -> None:
        # TDP does not depend on the workload, so only the GPU must be known
        if gpu not in self._tdp:
            raise ValueError(f'GPU {gpu!r} not in power table; have {gpu_names()}')


class MeasuredPowerModel(PowerModel):
    """Measured board power per workload, with measured idle and cold start.

    Args:
        idle_mode: ``'powered'`` keeps idle GPUs on at their measured reserved
            idle power. ``'released'`` returns idle GPUs (0 W) and charges the
            measured cold-start energy per GPU at each job start.

    Attributes:
        idle_mode: str, ``'powered'`` or ``'released'``
    """
    def __init__(self, idle_mode: Literal['powered', 'released'] = 'powered'):
        super().__init__()
        if idle_mode not in ('powered', 'released'):
            raise ValueError(f'unknown idle_mode {idle_mode!r}')
        self.idle_mode = idle_mode
        self.name = 'measured' if idle_mode == 'powered' else 'measured_released'

    def active_w(self, gpu: str, workload: str) -> float:
        return self._active[(gpu, workload)]

    def idle_w(self, gpu: str) -> float:
        return 0.0 if self.idle_mode == 'released' else self._idle[gpu]

    def start_j(self, gpu: str) -> float:
        return self._coldstart[gpu] if self.idle_mode == 'released' else 0.0


def make_power_model(name: str | PowerModel) -> PowerModel:
    """Returns a power model by name, or ``name`` itself if it is a model.

    Names: ``'tdp'``, ``'tdp_idle40'`` (idle at 40% of TDP), ``'measured'``,
    ``'measured_released'``.
    """
    if isinstance(name, PowerModel):
        return name
    if name == 'tdp':
        return TDPPowerModel(0.0)
    if name == 'tdp_idle40':
        return TDPPowerModel(0.4)
    if name == 'measured':
        return MeasuredPowerModel('powered')
    if name == 'measured_released':
        return MeasuredPowerModel('released')
    raise ValueError(f'unknown power model {name!r}; choose from {POWER_MODEL_NAMES}')
