from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class TheoryStepResult:
    state: np.ndarray
    aux: dict[str, Any] | None = None


class TheoryModel(ABC):
    @abstractmethod
    def initialize_state(self, state0: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def initialize_click_state(self, x_click: float, y_click: float, sigma_click: float) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step_forward(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        raise NotImplementedError

    @abstractmethod
    def step_backward_adjoint(self, state: np.ndarray, dt: float) -> TheoryStepResult:
        raise NotImplementedError

    @abstractmethod
    def density(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def current(self, state_vis: np.ndarray):
        raise NotImplementedError

    def velocity(self, state_vis: np.ndarray, eps_rho: float = 1e-10):
        jx, jy, rho = self.current(state_vis)
        denom = np.maximum(rho, eps_rho)
        vx = jx / denom
        vy = jy / denom
        sp = np.hypot(vx, vy)
        return vx, vy, sp