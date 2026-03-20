from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PotentialComponent:
    name: str
    kind: str

    # Real and imaginary contributions
    V_real: np.ndarray
    W: np.ndarray

    # Geometry/debug/visualization
    barrier_core: np.ndarray
    wall_mask: np.ndarray
    slit_masks: dict[str, np.ndarray] = field(default_factory=dict)