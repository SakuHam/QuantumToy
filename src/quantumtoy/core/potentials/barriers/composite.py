from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import PotentialComponent


@dataclass
class CompositeBarrierResult:
    V_real: np.ndarray
    W: np.ndarray
    components: list[PotentialComponent]


class CompositeBarrierSystem:
    def __init__(self, *, V_clip_max: float | None = None):
        self.V_clip_max = None if V_clip_max is None else float(V_clip_max)
        self._components = []

    def add(self, component_builder):
        self._components.append(component_builder)

    def build(self, X: np.ndarray, Y: np.ndarray) -> CompositeBarrierResult:
        V_real = np.zeros_like(X, dtype=float)
        W = np.zeros_like(X, dtype=float)
        components: list[PotentialComponent] = []

        for builder in self._components:
            comp = builder.build(X, Y)
            components.append(comp)
            V_real += comp.V_real
            W += comp.W

        if self.V_clip_max is not None:
            V_real = np.minimum(V_real, self.V_clip_max)

        return CompositeBarrierResult(
            V_real=V_real,
            W=W,
            components=components,
        )