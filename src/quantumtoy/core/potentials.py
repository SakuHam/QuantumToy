from __future__ import annotations
import numpy as np
from core.simulation_types import PotentialSpec


def smooth_cap_edge(X, Y, Lx, Ly, cap_width=8.0, strength=2.0, power=4):
    dist_to_x = (Lx / 2) - np.abs(X)
    dist_to_y = (Ly / 2) - np.abs(Y)
    dist_to_edge = np.minimum(dist_to_x, dist_to_y)

    W = np.zeros_like(X, dtype=float)
    mask = dist_to_edge < cap_width
    s = (cap_width - dist_to_edge[mask]) / cap_width
    W[mask] = strength * (s ** power)
    return W


def build_double_slit_and_caps(grid, cfg) -> PotentialSpec:
    X = grid.X
    Y = grid.Y

    barrier_core = np.abs(X - cfg.barrier_center_x) < (cfg.barrier_thickness / 2.0)
    slit1_mask = np.abs(Y - cfg.slit_center_offset) < cfg.slit_half_height
    slit2_mask = np.abs(Y + cfg.slit_center_offset) < cfg.slit_half_height

    if cfg.BARRIER_SMOOTH <= 0.0:
        barrier_mask = barrier_core.copy()
        barrier_mask[slit1_mask] = False
        barrier_mask[slit2_mask] = False
        V_real = np.zeros_like(X, dtype=float)
        V_real[barrier_mask] = cfg.V_barrier
    else:
        dist = np.abs(X - cfg.barrier_center_x) - (cfg.barrier_thickness / 2.0)
        wall = 1.0 / (1.0 + np.exp(dist / cfg.BARRIER_SMOOTH))
        wall[slit1_mask] = 0.0
        wall[slit2_mask] = 0.0
        V_real = cfg.V_barrier * wall

    W_edge = smooth_cap_edge(
        X, Y, grid.Lx, grid.Ly,
        cap_width=cfg.CAP_WIDTH,
        strength=cfg.CAP_STRENGTH,
        power=cfg.CAP_POWER,
    )

    screen_mask_full = np.abs(X - cfg.screen_center_x) < cfg.screen_eval_width
    screen_mask_vis = screen_mask_full[grid.ys, grid.xs]

    W_screen = np.zeros_like(X, dtype=float)
    if cfg.USE_SCREEN_CAP:
        W_screen[screen_mask_full] = cfg.SCREEN_CAP_STRENGTH

    W = W_edge + W_screen

    return PotentialSpec(
        V_real=V_real,
        W=W,
        screen_mask_full=screen_mask_full,
        screen_mask_vis=screen_mask_vis,
        barrier_core=barrier_core,
        slit1_mask=slit1_mask,
        slit2_mask=slit2_mask,
    )