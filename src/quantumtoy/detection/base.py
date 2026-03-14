from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class DetectorStepResult:
    clicked: bool
    click_x: float | None = None
    click_y: float | None = None
    click_ix: int | None = None
    click_iy: int | None = None
    click_time: float | None = None
    aux: dict | None = None


class DetectorModel(ABC):
    """
    Common interface for interchangeable detector models.
    """

    @abstractmethod
    def reset(self, **kwargs) -> None:
        pass

    @abstractmethod
    def step(
        self,
        psi: np.ndarray,
        dt: float,
        t: float | None = None,
    ) -> DetectorStepResult:
        pass

    @abstractmethod
    def has_clicked(self) -> bool:
        pass