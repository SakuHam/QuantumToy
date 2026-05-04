# analysis/entanglement_posthoc_trf.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


CHANNELS = ["++", "+-", "-+", "--"]


@dataclass
class EntanglementPosthocTRFConfig:
    enabled: bool = True
    sigmaT: float = 0.5
    K_JITTER: int = 13

    use_adaptive_ref: bool = True
    ref_t_min_frac: float = 0.30
    ref_t_max_frac: float = 0.95
    valid_total_evidence_eps: float = 1e-12

    use_worldline: bool = True
    wl_track_radius_px: int = 20
    wl_tube_sigma_px: float = 10.0
    wl_gain_strength: float = 2.0
    wl_outside_damp: float = 0.20
    wl_time_ramp_frac: float = 0.12


@dataclass
class EntanglementPosthocTRFResult:
    valid: bool
    ref_idx: int | None
    ref_time: float | None
    chosen_channel: str | None
    channel_evidence: dict[str, float]
    channel_probs: dict[str, float]
    E_trf: float
    total_evidence: float
    dominance: float
    ratio: float
    aux: dict


def rotate_state_to_measurement_basis(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> np.ndarray:
    tmp = np.einsum("ia,xyab->xyib", Ua.conj().T, psi)
    out = np.einsum("xyib,jb->xyij", tmp, Ub.conj().T)
    return out


def channel_component_densities(
    psi: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> dict[str, np.ndarray]:
    psi_m = rotate_state_to_measurement_basis(psi, Ua, Ub)
    return {
        "++": np.abs(psi_m[:, :, 0, 0]) ** 2,
        "+-": np.abs(psi_m[:, :, 0, 1]) ** 2,
        "-+": np.abs(psi_m[:, :, 1, 0]) ** 2,
        "--": np.abs(psi_m[:, :, 1, 1]) ** 2,
    }


def channel_component_frames(
    psi_frames: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
) -> dict[str, np.ndarray]:
    out = {ch: [] for ch in CHANNELS}
    for psi in psi_frames:
        comps = channel_component_densities(psi.astype(np.complex128), Ua, Ub)
        for ch in CHANNELS:
            out[ch].append(comps[ch].astype(np.float32))
    return {ch: np.asarray(out[ch], dtype=np.float32) for ch in CHANNELS}


def gaussian_weights(times: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        w = np.zeros_like(times, dtype=float)
        w[int(np.argmin(np.abs(times - mu)))] = 1.0
        return w

    z = (times - mu) / sigma
    w = np.exp(-0.5 * z * z)
    s = float(np.sum(w))
    return w / s if s > 0.0 else w


def build_entangled_Emix_density_from_phi_tau(
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    sigmaT: float,
    tau_step: float,
    K_JITTER: int = 13,
) -> np.ndarray:
    """
    phi_tau_frames shape:
        (Nt, NxB, NxA, 2, 2)

    returns:
        Emix_density shape:
        (Nt, NxB, NxA)
    """
    Nt = len(times)
    halfK = int(K_JITTER) // 2
    idx_det = int(np.argmin(np.abs(times - t_det)))

    k_inds = np.arange(idx_det - halfK, idx_det + halfK + 1)
    k_inds = np.clip(k_inds, 0, Nt - 1)
    k_inds = np.unique(k_inds)

    Tk = times[k_inds]
    w = gaussian_weights(Tk, t_det, sigmaT)

    phi_density = np.sum(np.abs(phi_tau_frames) ** 2, axis=(-2, -1)).astype(np.float32)
    Emix_density = np.zeros_like(phi_density, dtype=np.float32)

    for i, ti in enumerate(times):
        tau = Tk - ti
        valid = tau >= 0.0
        if not np.any(valid):
            continue

        j = np.rint(tau[valid] / tau_step).astype(int)
        j = np.clip(j, 0, Nt - 1)

        Emix_density[i] = np.sum(
            w[valid][:, None, None] * phi_density[j],
            axis=0,
        ).astype(np.float32)

    return Emix_density


def make_entangled_base_rho_density_product(
    psi_frames: np.ndarray,
    Emix_density: np.ndarray,
    dx: float,
) -> np.ndarray:
    fwd_density = np.sum(np.abs(psi_frames) ** 2, axis=(-2, -1)).astype(np.float32)

    out = np.zeros_like(fwd_density, dtype=np.float32)
    dx2 = dx * dx

    for i in range(fwd_density.shape[0]):
        rho = fwd_density[i].astype(np.float64) * Emix_density[i].astype(np.float64)
        s = float(np.sum(rho) * dx2)
        if s > 0.0:
            rho /= s
        out[i] = rho.astype(np.float32)

    return out


def make_channel_rho_density_product(
    psi_frames: np.ndarray,
    Emix_density: np.ndarray,
    Ua: np.ndarray,
    Ub: np.ndarray,
    dx: float,
) -> dict[str, np.ndarray]:
    comps = channel_component_frames(psi_frames, Ua, Ub)
    out = {}
    dx2 = dx * dx

    for ch in CHANNELS:
        arr = np.zeros_like(comps[ch], dtype=np.float32)
        for i in range(comps[ch].shape[0]):
            rho = comps[ch][i].astype(np.float64) * Emix_density[i].astype(np.float64)
            s = float(np.sum(rho) * dx2)
            if s > 0.0:
                rho /= s
            arr[i] = rho.astype(np.float32)
        out[ch] = arr

    return out


def compute_channel_evidence_for_frame(
    channel_rho: dict[str, np.ndarray],
    idx: int,
    dx: float,
) -> dict[str, float]:
    dx2 = dx * dx
    ev = {}
    for ch in CHANNELS:
        ev[ch] = float(np.sum(channel_rho[ch][idx]) * dx2)
    return ev


def normalize_evidence(ev: dict[str, float]) -> dict[str, float]:
    total = float(sum(ev.values()))
    if total <= 0.0:
        return {ch: 0.0 for ch in CHANNELS}
    return {ch: float(ev[ch] / total) for ch in CHANNELS}


def evidence_E(ev_probs: dict[str, float]) -> float:
    return float(ev_probs["++"] + ev_probs["--"] - ev_probs["+-"] - ev_probs["-+"])


def choose_entangled_trf_channel(
    channel_rho: dict[str, np.ndarray],
    times: np.ndarray,
    dx: float,
    cfg: EntanglementPosthocTRFConfig,
) -> dict:
    Nt = len(times)

    if cfg.use_adaptive_ref:
        t_min = float(cfg.ref_t_min_frac * times[-1])
        t_max = float(cfg.ref_t_max_frac * times[-1])
        cand = np.where((times >= t_min) & (times <= t_max))[0]
        if len(cand) == 0:
            cand = np.arange(Nt)
    else:
        cand = np.asarray([int(np.argmin(np.abs(times - 0.55 * times[-1])))])

    best = None

    for idx in cand:
        ev = compute_channel_evidence_for_frame(channel_rho, int(idx), dx)
        probs = normalize_evidence(ev)
        total = float(sum(ev.values()))
        chosen = max(CHANNELS, key=lambda ch: ev[ch])
        max_ev = float(ev[chosen])
        min_nonzero = min([v for v in ev.values() if v > 0.0], default=1e-30)

        dominance = float(max_ev / max(total, 1e-30))
        ratio = float(max_ev / max(min_nonzero, 1e-30))
        score = float(total * dominance)

        rec = {
            "idx": int(idx),
            "time": float(times[idx]),
            "ev": ev,
            "probs": probs,
            "chosen": chosen,
            "total": total,
            "dominance": dominance,
            "ratio": ratio,
            "score": score,
            "E": evidence_E(probs),
        }

        if best is None or rec["score"] > best["score"]:
            best = rec

    valid = bool(best is not None and best["total"] >= cfg.valid_total_evidence_eps)

    return {
        "valid": valid,
        "ref_idx": None if best is None else int(best["idx"]),
        "ref_time": None if best is None else float(best["time"]),
        "chosen_channel": None if not valid else str(best["chosen"]),
        "channel_evidence": {ch: 0.0 for ch in CHANNELS} if best is None else best["ev"],
        "channel_probs": {ch: 0.0 for ch in CHANNELS} if best is None else best["probs"],
        "E_trf": 0.0 if best is None else float(best["E"]),
        "total_evidence": 0.0 if best is None else float(best["total"]),
        "dominance": 0.0 if best is None else float(best["dominance"]),
        "ratio": 0.0 if best is None else float(best["ratio"]),
    }


def extract_local_peak(arr: np.ndarray, iy0: int, ix0: int, radius_px: int):
    ny, nx = arr.shape
    y0 = max(0, iy0 - radius_px)
    y1 = min(ny, iy0 + radius_px + 1)
    x0 = max(0, ix0 - radius_px)
    x1 = min(nx, ix0 + radius_px + 1)

    sub = arr[y0:y1, x0:x1]
    flat = int(np.argmax(sub))
    sy, sx = np.unravel_index(flat, sub.shape)
    return y0 + sy, x0 + sx, float(sub[sy, sx])


def track_worldline_2d(
    rho_frames: np.ndarray,
    start_i: int,
    start_iy: int,
    start_ix: int,
    radius_px: int,
):
    Nt = rho_frames.shape[0]
    path_y = np.full(Nt, int(start_iy), dtype=int)
    path_x = np.full(Nt, int(start_ix), dtype=int)

    iy, ix = int(start_iy), int(start_ix)
    for i in range(start_i + 1, Nt):
        iy, ix, _ = extract_local_peak(rho_frames[i], iy, ix, radius_px)
        path_y[i] = iy
        path_x[i] = ix

    iy, ix = int(start_iy), int(start_ix)
    for i in range(start_i - 1, -1, -1):
        iy, ix, _ = extract_local_peak(rho_frames[i], iy, ix, radius_px)
        path_y[i] = iy
        path_x[i] = ix

    return path_y, path_x


def build_tube(shape, path_y: np.ndarray, path_x: np.ndarray, sigma_px: float):
    Nt, ny, nx = shape
    yy = np.arange(ny)[:, None]
    xx = np.arange(nx)[None, :]

    tube = np.zeros(shape, dtype=np.float32)
    inv2s2 = 1.0 / max(2.0 * sigma_px * sigma_px, 1e-12)

    for i in range(Nt):
        tube[i] = np.exp(-((yy - path_y[i]) ** 2 + (xx - path_x[i]) ** 2) * inv2s2)

    return tube


def smooth_time_ramp(Nt: int, center_idx: int, ramp_frac: float):
    ramp_len = max(3, int(ramp_frac * Nt))
    gate = np.zeros(Nt, dtype=np.float32)

    for i in range(Nt):
        if i <= center_idx - ramp_len:
            gate[i] = 0.0
        elif i >= center_idx + ramp_len:
            gate[i] = 1.0
        else:
            u = (i - (center_idx - ramp_len)) / (2.0 * ramp_len)
            gate[i] = 0.5 - 0.5 * np.cos(np.pi * u)

    return gate


def apply_channel_worldline_selection(
    base_rho: np.ndarray,
    selected_channel_rho: np.ndarray,
    ref_idx: int,
    cfg: EntanglementPosthocTRFConfig,
    dx: float,
):
    ref = selected_channel_rho[ref_idx]
    iy, ix = np.unravel_index(int(np.argmax(ref)), ref.shape)

    path_y, path_x = track_worldline_2d(
        selected_channel_rho,
        start_i=ref_idx,
        start_iy=int(iy),
        start_ix=int(ix),
        radius_px=int(cfg.wl_track_radius_px),
    )

    tube = build_tube(
        base_rho.shape,
        path_y,
        path_x,
        sigma_px=float(cfg.wl_tube_sigma_px),
    )

    time_gate = smooth_time_ramp(
        base_rho.shape[0],
        ref_idx,
        ramp_frac=float(cfg.wl_time_ramp_frac),
    )

    out = np.zeros_like(base_rho, dtype=np.float32)
    dx2 = dx * dx

    for i in range(base_rho.shape[0]):
        g = float(time_gate[i])
        field = (
            float(cfg.wl_gain_strength) * g * tube[i]
            - float(cfg.wl_outside_damp) * g * (1.0 - tube[i])
        )
        rho = base_rho[i].astype(np.float64) * np.exp(field)
        s = float(np.sum(rho) * dx2)
        if s > 0.0:
            rho /= s
        out[i] = rho.astype(np.float32)

    return out, {
        "seed_ix": int(ix),
        "seed_iy": int(iy),
        "path_x": path_x,
        "path_y": path_y,
        "tube": tube,
        "time_gate": time_gate,
    }


def run_entanglement_posthoc_trf(
    psi_frames: np.ndarray,
    phi_tau_frames: np.ndarray,
    times: np.ndarray,
    t_det: float,
    tau_step: float,
    dx: float,
    Ua: np.ndarray,
    Ub: np.ndarray,
    cfg: EntanglementPosthocTRFConfig,
) -> EntanglementPosthocTRFResult:
    if not cfg.enabled:
        return EntanglementPosthocTRFResult(
            valid=False,
            ref_idx=None,
            ref_time=None,
            chosen_channel=None,
            channel_evidence={ch: 0.0 for ch in CHANNELS},
            channel_probs={ch: 0.0 for ch in CHANNELS},
            E_trf=0.0,
            total_evidence=0.0,
            dominance=0.0,
            ratio=0.0,
            aux={"enabled": False},
        )

    Emix_density = build_entangled_Emix_density_from_phi_tau(
        phi_tau_frames=phi_tau_frames,
        times=times,
        t_det=t_det,
        sigmaT=float(cfg.sigmaT),
        tau_step=tau_step,
        K_JITTER=int(cfg.K_JITTER),
    )

    base_rho = make_entangled_base_rho_density_product(
        psi_frames=psi_frames,
        Emix_density=Emix_density,
        dx=dx,
    )

    channel_rho = make_channel_rho_density_product(
        psi_frames=psi_frames,
        Emix_density=Emix_density,
        Ua=Ua,
        Ub=Ub,
        dx=dx,
    )

    info = choose_entangled_trf_channel(
        channel_rho=channel_rho,
        times=times,
        dx=dx,
        cfg=cfg,
    )

    rho_selected = None
    wl_aux = {}

    if cfg.use_worldline and info["valid"]:
        ch = info["chosen_channel"]
        rho_selected, wl_aux = apply_channel_worldline_selection(
            base_rho=base_rho,
            selected_channel_rho=channel_rho[ch],
            ref_idx=int(info["ref_idx"]),
            cfg=cfg,
            dx=dx,
        )

    return EntanglementPosthocTRFResult(
        valid=bool(info["valid"]),
        ref_idx=info["ref_idx"],
        ref_time=info["ref_time"],
        chosen_channel=info["chosen_channel"],
        channel_evidence=info["channel_evidence"],
        channel_probs=info["channel_probs"],
        E_trf=float(info["E_trf"]),
        total_evidence=float(info["total_evidence"]),
        dominance=float(info["dominance"]),
        ratio=float(info["ratio"]),
        aux={
            "enabled": True,
            "Emix_density": Emix_density,
            "base_rho": base_rho,
            "channel_rho": channel_rho,
            "rho_selected": rho_selected,
            "worldline_aux": wl_aux,
        },
    )