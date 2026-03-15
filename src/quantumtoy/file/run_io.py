from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================
# Generic helpers
# ============================================================

def _jsonable(obj: Any):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)


def cfg_to_dict(cfg) -> dict:
    out = {}
    for k, v in vars(cfg).items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        out[k] = _jsonable(v)
    return out


def apply_cfg_dict(cfg, cfg_dict: dict):
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    return cfg


def ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Optional array helpers
# ============================================================

def pack_optional_array(arr, dtype=None):
    if arr is None:
        if dtype is None:
            dtype = float
        return np.array([], dtype=dtype)
    return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)


def unpack_optional_array(arr: np.ndarray):
    if arr.size == 0:
        return None
    return arr


def pack_points_2d(points):
    if points is None:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(points, dtype=float)
    return arr.reshape(-1, 2)


# ============================================================
# Save / load
# ============================================================

def save_run_bundle(
    *,
    output_prefix: str | Path,
    cfg,
    grid,
    potential,
    debug_free_case: bool,
    times,
    frames_density,
    state_vis_frames,
    norms,
    screen_int,
    phi_tau_frames,
    x_click,
    y_click,
    t_det,
    idx_det,
    detector_clicked,
    sigma_init,
    ridge_x_init,
    ridge_y_init,
    ridge_s_init,
    cos_th_init,
    speed_init,
    ux_init,
    uy_init,
    div_v_init,
    vref,
    speed_ref,
    bohm_traj_x,
    bohm_traj_y,
    bohm_traj_alive,
    bohm_init_points,
):
    prefix = Path(output_prefix)
    ensure_parent_dir(prefix)
    npz_path = prefix.with_suffix(".npz")
    meta_path = prefix.with_suffix(".json")

    np.savez_compressed(
        npz_path,
        times=np.asarray(times),
        frames_density=np.asarray(frames_density),
        state_vis_frames=pack_optional_array(state_vis_frames),
        norms=np.asarray(norms),
        screen_int=np.asarray(screen_int),
        phi_tau_frames=np.asarray(phi_tau_frames),

        x_click=np.array([x_click], dtype=float),
        y_click=np.array([y_click], dtype=float),
        t_det=np.array([t_det], dtype=float),
        idx_det=np.array([idx_det], dtype=int),
        detector_clicked=np.array([bool(detector_clicked)], dtype=bool),

        sigma_init=np.array([sigma_init], dtype=float),

        ridge_x_init=np.asarray(ridge_x_init),
        ridge_y_init=np.asarray(ridge_y_init),
        ridge_s_init=np.asarray(ridge_s_init),

        cos_th_init=pack_optional_array(cos_th_init, dtype=float),
        speed_init=pack_optional_array(speed_init, dtype=float),
        ux_init=pack_optional_array(ux_init, dtype=float),
        uy_init=pack_optional_array(uy_init, dtype=float),
        div_v_init=pack_optional_array(div_v_init, dtype=float),

        vref=np.array([vref], dtype=float),
        speed_ref=np.array([speed_ref], dtype=float),

        bohm_traj_x=pack_optional_array(bohm_traj_x, dtype=float),
        bohm_traj_y=pack_optional_array(bohm_traj_y, dtype=float),
        bohm_traj_alive=pack_optional_array(bohm_traj_alive, dtype=bool),
        bohm_init_points=pack_points_2d(bohm_init_points),

        x_vis_1d=np.asarray(grid.x_vis_1d),
        y_vis_1d=np.asarray(grid.y_vis_1d),
        X_vis=np.asarray(grid.X_vis),
        Y_vis=np.asarray(grid.Y_vis),

        screen_mask_vis=np.asarray(potential.screen_mask_vis),
    )

    meta = {
        "config": cfg_to_dict(cfg),
        "output_prefix": str(prefix),
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "debug_free_case": bool(debug_free_case),
        "has_state_vis_frames": bool(state_vis_frames is not None),
        "has_bohmian": bool(bohm_traj_x is not None),
        "visible_extent": {
            "x_vis_min": float(grid.x_vis_min),
            "x_vis_max": float(grid.x_vis_max),
            "y_vis_min": float(grid.y_vis_min),
            "y_vis_max": float(grid.y_vis_max),
            "dx": float(grid.dx),
            "dy": float(grid.dy),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] data -> {npz_path}")
    print(f"[SAVE] meta -> {meta_path}")


def load_run_bundle(npz_path: str | Path, meta_path: str | Path | None = None) -> dict:
    npz_path = Path(npz_path)
    if meta_path is None:
        meta_path = npz_path.with_suffix(".json")
    else:
        meta_path = Path(meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    raw = np.load(npz_path, allow_pickle=True)

    out = {
        "meta": meta,
        "times": raw["times"],
        "frames_density": raw["frames_density"],
        "state_vis_frames": unpack_optional_array(raw["state_vis_frames"]),
        "norms": raw["norms"],
        "screen_int": raw["screen_int"],
        "phi_tau_frames": raw["phi_tau_frames"],

        "x_click": float(raw["x_click"][0]),
        "y_click": float(raw["y_click"][0]),
        "t_det": float(raw["t_det"][0]),
        "idx_det": int(raw["idx_det"][0]),
        "detector_clicked": bool(raw["detector_clicked"][0]),

        "sigma_init": float(raw["sigma_init"][0]),

        "ridge_x_init": raw["ridge_x_init"],
        "ridge_y_init": raw["ridge_y_init"],
        "ridge_s_init": raw["ridge_s_init"],

        "cos_th_init": unpack_optional_array(raw["cos_th_init"]),
        "speed_init": unpack_optional_array(raw["speed_init"]),
        "ux_init": unpack_optional_array(raw["ux_init"]),
        "uy_init": unpack_optional_array(raw["uy_init"]),
        "div_v_init": unpack_optional_array(raw["div_v_init"]),

        "vref": float(raw["vref"][0]),
        "speed_ref": float(raw["speed_ref"][0]),

        "bohm_traj_x": unpack_optional_array(raw["bohm_traj_x"]),
        "bohm_traj_y": unpack_optional_array(raw["bohm_traj_y"]),
        "bohm_traj_alive": unpack_optional_array(raw["bohm_traj_alive"]),
        "bohm_init_points": raw["bohm_init_points"],

        "x_vis_1d": raw["x_vis_1d"],
        "y_vis_1d": raw["y_vis_1d"],
        "X_vis": raw["X_vis"],
        "Y_vis": raw["Y_vis"],
        "screen_mask_vis": raw["screen_mask_vis"],
    }

    return out