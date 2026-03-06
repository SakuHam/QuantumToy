from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GridSpec:
    visible_lx: float
    visible_ly: float
    n_visible_x: int
    n_visible_y: int
    pad_factor: int
    Lx: float
    Ly: float
    Nx: int
    Ny: int
    dx: float
    dy: float
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    cx: int
    cy: int
    hx: int
    hy: int
    xs: slice
    ys: slice
    x_vis_1d: np.ndarray
    y_vis_1d: np.ndarray
    X_vis: np.ndarray
    Y_vis: np.ndarray
    x_vis_min: float
    x_vis_max: float
    y_vis_min: float
    y_vis_max: float
    mask_visible: np.ndarray


@dataclass
class PotentialSpec:
    V_real: np.ndarray
    W: np.ndarray
    screen_mask_full: np.ndarray
    screen_mask_vis: np.ndarray
    barrier_core: np.ndarray
    slit1_mask: np.ndarray
    slit2_mask: np.ndarray