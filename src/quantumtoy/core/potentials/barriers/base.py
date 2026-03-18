from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BarrierComponent:
    name: str
    kind: str
    V_real: np.ndarray
    barrier_core: np.ndarray
    slit_masks: dict[str, np.ndarray]