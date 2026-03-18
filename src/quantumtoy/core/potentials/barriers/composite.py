from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import BarrierComponent


@dataclass
class CompositeBarrierResult:
    V_real: np.ndarray
    components: list[BarrierComponent]


class CompositeBarrierSystem:
    def __init__(self, *, V_clip_max: float):
        self.V_clip_max = float(V_clip_max)
        self._barriers = []

    def add(self, barrier):
        self._barriers.append(barrier)

    def build(self, X: np.ndarray, Y: np.ndarray) -> CompositeBarrierResult:
        V_real = np.zeros_like(X, dtype=float)
        components: list[BarrierComponent] = []

        for barrier in self._barriers:
            comp = barrier.build(X, Y)
            components.append(comp)
            V_real += comp.V_real

        V_real = np.minimum(V_real, self.V_clip_max)

        return CompositeBarrierResult(
            V_real=V_real,
            components=components,
        )