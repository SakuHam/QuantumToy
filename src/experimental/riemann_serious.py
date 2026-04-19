from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Utilities
# ============================================================

def sieve_primes(n: int) -> np.ndarray:
    """Return all primes <= n."""
    if n < 2:
        return np.array([], dtype=int)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    limit = int(n ** 0.5) + 1
    for p in range(2, limit):
        if sieve[p]:
            sieve[p * p:n + 1:p] = False
    return np.flatnonzero(sieve)


def gaussian_kernel_fft(n: int, dt: float, ell: float) -> np.ndarray:
    """FFT-domain Gaussian kernel for circular convolution."""
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dt)
    return np.exp(-0.5 * (ell ** 2) * (k ** 2))


def smooth_field_real(f: np.ndarray, kernel_hat: np.ndarray) -> np.ndarray:
    """Circular Gaussian smoothing via FFT."""
    return np.fft.ifft(np.fft.fft(f) * kernel_hat).real


def laplacian_periodic_complex(u: np.ndarray, dt: float) -> np.ndarray:
    """Periodic 1D Laplacian for complex field."""
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dt * dt)


def l2_energy_complex(u: np.ndarray, dt: float) -> float:
    """Integral of |u|^2 dt."""
    return float(np.sum(np.abs(u) ** 2) * dt)


def complex_overlap(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Normalized overlap |<a,b>| / (||a|| ||b||)."""
    num = np.abs(np.sum(a * np.conj(b)) * dt)
    den = np.sqrt(np.sum(np.abs(a) ** 2) * dt) * np.sqrt(np.sum(np.abs(b) ** 2) * dt)
    return float(num / (den + 1e-12))


def build_edge_damping(
    t: np.ndarray,
    cap_start: float,
    cap_width: float,
    strength: float,
    power: float = 2.0,
) -> np.ndarray:
    """Edge damping profile W(t), used as -W(t)*psi."""
    W = np.zeros_like(t, dtype=float)
    abs_t = np.abs(t)
    mask = abs_t > cap_start
    if np.any(mask):
        x = (abs_t[mask] - cap_start) / cap_width
        x = np.clip(x, 0.0, 1.0)
        W[mask] = strength * (x ** power)
    return W


def chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split list into chunks."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


# ============================================================
# Source construction
# ============================================================

def build_log_points(mode: str, prime_max: int, seed: int = 0) -> np.ndarray:
    """
    Build log-space source points for different control modes.

    Modes:
      - real_primes
      - log_uniform
      - log_shuffled
      - random_integers
    """
    primes = sieve_primes(prime_max)
    if len(primes) == 0:
        return np.array([], dtype=float)

    logp = np.log(primes.astype(float))
    n = len(logp)

    if mode == "real_primes":
        return logp

    if mode == "log_uniform":
        return np.linspace(logp.min(), logp.max(), n)

    if mode == "log_shuffled":
        rng = np.random.default_rng(seed)
        out = logp.copy()
        rng.shuffle(out)
        return out

    if mode == "random_integers":
        rng = np.random.default_rng(seed)
        vals = rng.choice(np.arange(2, prime_max + 1), size=n, replace=False)
        vals.sort()
        return np.log(vals.astype(float))

    raise ValueError(f"Unknown source mode: {mode}")


def build_source_tables(
    source_mode: str,
    prime_max: int,
    t: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute:
      log_points
      exp(-i t log_points)
    """
    log_points = build_log_points(source_mode, prime_max, seed=seed)
    if len(log_points) == 0:
        return np.array([], dtype=float), np.zeros((0, t.size), dtype=np.complex128)

    phase_table = np.exp(-1j * np.outer(log_points, t))
    return log_points, phase_table


def compute_source(
    tau: float,
    sigma_weight: float,
    source_amp: float,
    source_width: float,
    log_points: np.ndarray,
    phase_table: np.ndarray,
    normalize_per_tau: bool,
) -> np.ndarray:
    """
    Source:
      S_sigma(t,tau) = A * sum_j w_tau(log_xj) * x_j^{-sigma} * exp(-i t log_xj)
    with x_j = exp(log_xj)
    """
    if len(log_points) == 0:
        return np.zeros(phase_table.shape[1], dtype=np.complex128)

    window = np.exp(-0.5 * ((log_points - tau) / source_width) ** 2)
    weights = window * np.exp(-sigma_weight * log_points)

    if normalize_per_tau:
        norm = np.sqrt(np.sum(weights ** 2))
        if norm > 1e-15:
            weights = weights / norm

    weights = source_amp * weights
    return (weights[:, None] * phase_table).sum(axis=0)


# ============================================================
# Model config
# ============================================================

@dataclass
class ModelConfig:
    sigma: float
    source_mode: str = "real_primes"

    # Base linear gain
    linear_gain: float = 0.015

    # PDE terms
    D: float = 0.03
    beta: float = 1.2
    ell: float = 1.2
    gamma: float = 2.0

    # Mirror-channel coupling
    kappa: float = 0.08

    # Optional phase potential
    use_phase_potential: bool = False
    phase_potential_c2: float = 0.0

    # Grid / integration
    t_min: float = -50.0
    t_max: float = 50.0
    n_t: int = 1024
    dtau: float = 0.0015
    n_steps: int = 8000
    save_every: int = 25

    # Initial condition
    init_noise: float = 1e-4
    seed: int = 0

    # Source
    use_source: bool = True
    prime_max: int = 3000
    source_amp: float = 0.08
    source_width: float = 0.14
    normalize_source_per_tau: bool = False

    # Observed window
    obs_t_min: float = -40.0
    obs_t_max: float = 40.0

    # Edge damping / CAP
    use_edge_damping: bool = True
    edge_cap_start: float = 40.0
    edge_cap_width: float = 10.0
    edge_cap_strength: float = 1.5
    edge_cap_power: float = 2.0


# ============================================================
# Simulation
# ============================================================

def simulate(cfg: ModelConfig) -> dict:
    """
    Mirror model:

      dPsi+/dtau = L[Psi+] + kappa (Psi- - Psi+) + S_sigma
      dPsi-/dtau = L[Psi-] + kappa (Psi+ - Psi-) + S_{1-sigma}

    Track:
      M_diff      = ∫|Psi+ - Psi-|² dt
      M_diff_norm = M_diff / (∫|Psi+|² + ∫|Psi-|² + eps)
      M_sum       = ∫|Psi+ + Psi-|² dt
      channel_overlap = |<Psi+, Psi->| / (||Psi+|| ||Psi-||)
      R_lock      = M_sum / (M_diff + eps)
    """
    rng = np.random.default_rng(cfg.seed)

    t = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t, endpoint=False)
    dt = t[1] - t[0]

    obs_mask = (t >= cfg.obs_t_min) & (t <= cfg.obs_t_max)

    kernel_hat = gaussian_kernel_fft(cfg.n_t, dt=dt, ell=cfg.ell)
    log_points, phase_table = build_source_tables(
        source_mode=cfg.source_mode,
        prime_max=cfg.prime_max,
        t=t,
        seed=cfg.seed,
    )

    psi_p = (
        cfg.init_noise * rng.standard_normal(cfg.n_t)
        + 1j * cfg.init_noise * rng.standard_normal(cfg.n_t)
    ).astype(np.complex128)

    psi_m = (
        cfg.init_noise * rng.standard_normal(cfg.n_t)
        + 1j * cfg.init_noise * rng.standard_normal(cfg.n_t)
    ).astype(np.complex128)

    if cfg.use_phase_potential:
        Vt = cfg.phase_potential_c2 * (t ** 2)
    else:
        Vt = np.zeros_like(t)

    if cfg.use_edge_damping:
        edge_W = build_edge_damping(
            t=t,
            cap_start=cfg.edge_cap_start,
            cap_width=cfg.edge_cap_width,
            strength=cfg.edge_cap_strength,
            power=cfg.edge_cap_power,
        )
    else:
        edge_W = np.zeros_like(t)

    n_save = cfg.n_steps // cfg.save_every + 1

    taus = np.zeros(n_save, dtype=float)
    frames_p = np.zeros((n_save, cfg.n_t), dtype=np.complex128)
    frames_m = np.zeros((n_save, cfg.n_t), dtype=np.complex128)

    energy_obs_p = np.zeros(n_save, dtype=float)
    energy_obs_m = np.zeros(n_save, dtype=float)
    m_diff = np.zeros(n_save, dtype=float)
    m_diff_norm = np.zeros(n_save, dtype=float)
    m_sum = np.zeros(n_save, dtype=float)
    channel_overlap = np.zeros(n_save, dtype=float)
    r_lock = np.zeros(n_save, dtype=float)

    source_energy_p = np.zeros(n_save, dtype=float)
    source_energy_m = np.zeros(n_save, dtype=float)

    def save(idx: int, tau: float, field_p: np.ndarray, field_m: np.ndarray,
             src_p: np.ndarray, src_m: np.ndarray) -> None:
        fp_obs = field_p[obs_mask]
        fm_obs = field_m[obs_mask]
        sp_obs = src_p[obs_mask]
        sm_obs = src_m[obs_mask]

        diff_obs = fp_obs - fm_obs
        sum_obs = fp_obs + fm_obs

        e_p = l2_energy_complex(fp_obs, dt)
        e_m = l2_energy_complex(fm_obs, dt)
        e_diff = l2_energy_complex(diff_obs, dt)
        e_sum = l2_energy_complex(sum_obs, dt)

        frames_p[idx] = field_p
        frames_m[idx] = field_m
        taus[idx] = tau

        energy_obs_p[idx] = e_p
        energy_obs_m[idx] = e_m
        m_diff[idx] = e_diff
        m_diff_norm[idx] = float(e_diff / (e_p + e_m + 1e-12))
        m_sum[idx] = e_sum
        channel_overlap[idx] = complex_overlap(fp_obs, fm_obs, dt)
        r_lock[idx] = float(e_sum / (e_diff + 1e-12))

        source_energy_p[idx] = l2_energy_complex(sp_obs, dt)
        source_energy_m[idx] = l2_energy_complex(sm_obs, dt)

    src0_p = (
        compute_source(
            tau=0.0,
            sigma_weight=cfg.sigma,
            source_amp=cfg.source_amp,
            source_width=cfg.source_width,
            log_points=log_points,
            phase_table=phase_table,
            normalize_per_tau=cfg.normalize_source_per_tau,
        )
        if cfg.use_source
        else np.zeros_like(psi_p)
    )

    src0_m = (
        compute_source(
            tau=0.0,
            sigma_weight=(1.0 - cfg.sigma),
            source_amp=cfg.source_amp,
            source_width=cfg.source_width,
            log_points=log_points,
            phase_table=phase_table,
            normalize_per_tau=cfg.normalize_source_per_tau,
        )
        if cfg.use_source
        else np.zeros_like(psi_m)
    )

    save_idx = 0
    save(save_idx, 0.0, psi_p, psi_m, src0_p, src0_m)
    save_idx += 1

    for step in range(1, cfg.n_steps + 1):
        tau = step * cfg.dtau

        lap_p = laplacian_periodic_complex(psi_p, dt)
        lap_m = laplacian_periodic_complex(psi_m, dt)

        dens_p = np.abs(psi_p) ** 2
        dens_m = np.abs(psi_m) ** 2

        comp_p = smooth_field_real(dens_p, kernel_hat)
        comp_m = smooth_field_real(dens_m, kernel_hat)

        src_p = (
            compute_source(
                tau=tau,
                sigma_weight=cfg.sigma,
                source_amp=cfg.source_amp,
                source_width=cfg.source_width,
                log_points=log_points,
                phase_table=phase_table,
                normalize_per_tau=cfg.normalize_source_per_tau,
            )
            if cfg.use_source
            else np.zeros_like(psi_p)
        )

        src_m = (
            compute_source(
                tau=tau,
                sigma_weight=(1.0 - cfg.sigma),
                source_amp=cfg.source_amp,
                source_width=cfg.source_width,
                log_points=log_points,
                phase_table=phase_table,
                normalize_per_tau=cfg.normalize_source_per_tau,
            )
            if cfg.use_source
            else np.zeros_like(psi_m)
        )

        norm2_p = np.sum(dens_p) * dt
        norm2_m = np.sum(dens_m) * dt

        rhs_p = (
            cfg.linear_gain * psi_p
            + cfg.D * lap_p
            - cfg.beta * comp_p * psi_p
            - cfg.gamma * norm2_p * psi_p
            - edge_W * psi_p
            + cfg.kappa * (psi_m - psi_p)
            + src_p
            + 1j * Vt * psi_p
        )

        rhs_m = (
            cfg.linear_gain * psi_m
            + cfg.D * lap_m
            - cfg.beta * comp_m * psi_m
            - cfg.gamma * norm2_m * psi_m
            - edge_W * psi_m
            + cfg.kappa * (psi_p - psi_m)
            + src_m
            + 1j * Vt * psi_m
        )

        psi_p = psi_p + cfg.dtau * rhs_p
        psi_m = psi_m + cfg.dtau * rhs_m

        if (
            not np.all(np.isfinite(psi_p.real))
            or not np.all(np.isfinite(psi_p.imag))
            or not np.all(np.isfinite(psi_m.real))
            or not np.all(np.isfinite(psi_m.imag))
        ):
            raise FloatingPointError(
                f"Non-finite values encountered for sigma={cfg.sigma:.4f}, "
                f"source_mode={cfg.source_mode}, step={step}"
            )

        if step % cfg.save_every == 0:
            save(save_idx, tau, psi_p, psi_m, src_p, src_m)
            save_idx += 1

    return {
        "cfg": cfg,
        "t": t,
        "dt": dt,
        "taus": taus,
        "frames_p": frames_p,
        "frames_m": frames_m,
        "energy_obs_p": energy_obs_p,
        "energy_obs_m": energy_obs_m,
        "m_diff": m_diff,
        "m_diff_norm": m_diff_norm,
        "m_sum": m_sum,
        "channel_overlap": channel_overlap,
        "r_lock": r_lock,
        "source_energy_p": source_energy_p,
        "source_energy_m": source_energy_m,
    }


# ============================================================
# Sweeps
# ============================================================

def sweep_sigmas_for_mode(
    sigma_values: list[float],
    base_cfg: ModelConfig,
    source_mode: str,
) -> list[dict]:
    results = []

    for i, sigma in enumerate(sigma_values):
        cfg = ModelConfig(
            sigma=sigma,
            source_mode=source_mode,
            linear_gain=base_cfg.linear_gain,
            D=base_cfg.D,
            beta=base_cfg.beta,
            ell=base_cfg.ell,
            gamma=base_cfg.gamma,
            kappa=base_cfg.kappa,
            use_phase_potential=base_cfg.use_phase_potential,
            phase_potential_c2=base_cfg.phase_potential_c2,
            t_min=base_cfg.t_min,
            t_max=base_cfg.t_max,
            n_t=base_cfg.n_t,
            dtau=base_cfg.dtau,
            n_steps=base_cfg.n_steps,
            save_every=base_cfg.save_every,
            init_noise=base_cfg.init_noise,
            seed=base_cfg.seed + i,
            use_source=base_cfg.use_source,
            prime_max=base_cfg.prime_max,
            source_amp=base_cfg.source_amp,
            source_width=base_cfg.source_width,
            normalize_source_per_tau=base_cfg.normalize_source_per_tau,
            obs_t_min=base_cfg.obs_t_min,
            obs_t_max=base_cfg.obs_t_max,
            use_edge_damping=base_cfg.use_edge_damping,
            edge_cap_start=base_cfg.edge_cap_start,
            edge_cap_width=base_cfg.edge_cap_width,
            edge_cap_strength=base_cfg.edge_cap_strength,
            edge_cap_power=base_cfg.edge_cap_power,
        )

        print("=" * 96)
        print(
            f"mode={cfg.source_mode:>15s} | sigma={cfg.sigma:.4f} | "
            f"mirror={1.0 - cfg.sigma:.4f}"
        )

        res = simulate(cfg)
        results.append(res)

        print(
            f"[done] final M_diff={res['m_diff'][-1]:.6e} | "
            f"final M_diff_norm={res['m_diff_norm'][-1]:.6e} | "
            f"final M_sum={res['m_sum'][-1]:.6e} | "
            f"final overlap={res['channel_overlap'][-1]:.6e} | "
            f"max lock={np.max(res['r_lock']):.6e}"
        )

    return results


def run_all_modes(
    sigma_values: list[float],
    base_cfg: ModelConfig,
    source_modes: list[str],
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for mode in source_modes:
        out[mode] = sweep_sigmas_for_mode(sigma_values, base_cfg, mode)
    return out


# ============================================================
# Summaries
# ============================================================

def summarize_mode(results: list[dict]) -> dict:
    sigmas = np.array([res["cfg"].sigma for res in results], dtype=float)

    final_m_diff = np.array([res["m_diff"][-1] for res in results], dtype=float)
    max_m_diff = np.array([np.max(res["m_diff"]) for res in results], dtype=float)

    final_m_diff_norm = np.array([res["m_diff_norm"][-1] for res in results], dtype=float)
    max_m_diff_norm = np.array([np.max(res["m_diff_norm"]) for res in results], dtype=float)

    final_m_sum = np.array([res["m_sum"][-1] for res in results], dtype=float)
    max_m_sum = np.array([np.max(res["m_sum"]) for res in results], dtype=float)

    final_overlap = np.array([res["channel_overlap"][-1] for res in results], dtype=float)
    max_overlap = np.array([np.max(res["channel_overlap"]) for res in results], dtype=float)

    final_lock = np.array([res["r_lock"][-1] for res in results], dtype=float)
    max_lock = np.array([np.max(res["r_lock"]) for res in results], dtype=float)

    return {
        "sigmas": sigmas,
        "final_m_diff": final_m_diff,
        "max_m_diff": max_m_diff,
        "final_m_diff_norm": final_m_diff_norm,
        "max_m_diff_norm": max_m_diff_norm,
        "final_m_sum": final_m_sum,
        "max_m_sum": max_m_sum,
        "final_overlap": final_overlap,
        "max_overlap": max_overlap,
        "final_lock": final_lock,
        "max_lock": max_lock,
    }


# ============================================================
# Plotting
# ============================================================

def _add_shared_legend(fig, axes):
    handles = None
    labels = None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(9, len(labels)))


def plot_time_series_grid(
    results_by_mode: dict[str, list[dict]],
    metric_specs: list[tuple[str, str]],
    nrows: int = 4,
    ncols: int = 4,
    figsize: tuple[int, int] = (20, 18),
    output_path: str = "",
) -> None:
    """
    Each subplot = one (mode, metric) pair.
    Inside each subplot = all sigmas.
    """
    modes = list(results_by_mode.keys())
    panels = []
    for metric_key, metric_title in metric_specs:
        for mode in modes:
            panels.append((mode, metric_key, metric_title))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    for ax in axes:
        ax.set_visible(False)

    for i, (mode, metric_key, metric_title) in enumerate(panels[:len(axes)]):
        ax = axes[i]
        ax.set_visible(True)
        for res in results_by_mode[mode]:
            sigma = res["cfg"].sigma
            taus = res["taus"]
            y = res[metric_key]
            ax.plot(taus, y, label=f"{sigma:.2f}")
        ax.set_title(f"{metric_title}\n{mode}")
        ax.set_xlabel("tau")
        ax.grid(True, alpha=0.3)

    _add_shared_legend(fig, [ax for ax in axes if ax.get_visible()])

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[saved] time-series grid -> {output_path}")

    plt.show()


def plot_sigma_summary_grid(
    results_by_mode: dict[str, list[dict]],
    output_path: str = "",
) -> None:
    """
    One subplot per metric, each source_mode as a separate curve.
    """
    summaries = {mode: summarize_mode(reslist) for mode, reslist in results_by_mode.items()}
    modes = list(results_by_mode.keys())

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), constrained_layout=True)
    axes = axes.ravel()

    metric_panels = [
        ("final_m_diff", "Final M_diff vs sigma"),
        ("final_m_diff_norm", "Final M_diff_norm vs sigma"),
        ("final_m_sum", "Final M_sum vs sigma"),
        ("final_overlap", "Final channel overlap vs sigma"),
        ("max_lock", "Max R_lock vs sigma"),
        ("max_m_sum", "Max M_sum vs sigma"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_panels):
        for mode in modes:
            s = summaries[mode]["sigmas"]
            y = summaries[mode][metric_key]
            ax.plot(s, y, marker="o", label=mode)
        ax.axvline(0.5, ls="--", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("sigma")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[saved] sigma summary grid -> {output_path}")

    plt.show()


def plot_heatmap_grid(
    results_by_mode: dict[str, list[dict]],
    metric_key: str,
    metric_title: str,
    nrows: int = 2,
    ncols: int = 2,
    figsize: tuple[int, int] = (16, 10),
    output_path: str = "",
) -> None:
    """
    One subplot per source mode, heatmap over (tau, sigma).
    """
    modes = list(results_by_mode.keys())
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = axes.ravel()

    for ax in axes:
        ax.set_visible(False)

    for i, mode in enumerate(modes[:len(axes)]):
        ax = axes[i]
        ax.set_visible(True)

        reslist = results_by_mode[mode]
        taus = reslist[0]["taus"]
        sigmas = np.array([res["cfg"].sigma for res in reslist], dtype=float)
        Z = np.stack([res[metric_key] for res in reslist], axis=0)

        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=[taus[0], taus[-1], sigmas[0], sigmas[-1]],
        )
        ax.axhline(0.5, ls="--", alpha=0.7)
        ax.set_title(f"{metric_title}\n{mode}")
        ax.set_xlabel("tau")
        ax.set_ylabel("sigma")
        fig.colorbar(im, ax=ax)

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"[saved] heatmap grid -> {output_path}")

    plt.show()


def plot_compact_per_mode_figures(
    results_by_mode: dict[str, list[dict]],
    output_prefix: str = "",
    max_modes_per_figure: int = 4,
) -> None:
    """
    Compact per-mode figures:
    - each subplot = one source mode
    - all sigmas in same subplot
    - 4 stacked subplots per figure max
    """
    mode_items = list(results_by_mode.items())
    groups = chunk_list(mode_items, max_modes_per_figure)

    for gi, group in enumerate(groups, start=1):
        fig, axes = plt.subplots(len(group), 1, figsize=(18, 4.2 * len(group)), constrained_layout=True)
        if len(group) == 1:
            axes = [axes]

        for ax, (mode, reslist) in zip(axes, group):
            for res in reslist:
                sigma = res["cfg"].sigma
                ax.plot(res["taus"], res["m_diff_norm"], label=f"{sigma:.2f}")
            ax.set_title(f"M_diff_norm vs tau | {mode}")
            ax.set_xlabel("tau")
            ax.set_ylabel("M_diff_norm")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(9, len(reslist)))

        if output_prefix:
            out = f"{output_prefix}_compact_diffnorm_group_{gi}.png"
            fig.savefig(out, dpi=150)
            print(f"[saved] compact figure -> {out}")

        plt.show()

    for gi, group in enumerate(groups, start=1):
        fig, axes = plt.subplots(len(group), 1, figsize=(18, 4.2 * len(group)), constrained_layout=True)
        if len(group) == 1:
            axes = [axes]

        for ax, (mode, reslist) in zip(axes, group):
            for res in reslist:
                sigma = res["cfg"].sigma
                ax.plot(res["taus"], res["channel_overlap"], label=f"{sigma:.2f}")
            ax.set_title(f"Channel overlap vs tau | {mode}")
            ax.set_xlabel("tau")
            ax.set_ylabel("overlap")
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(9, len(reslist)))

        if output_prefix:
            out = f"{output_prefix}_compact_overlap_group_{gi}.png"
            fig.savefig(out, dpi=150)
            print(f"[saved] compact figure -> {out}")

        plt.show()


# ============================================================
# CLI
# ============================================================

def parse_sigma_list(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        s = part.strip()
        if s:
            vals.append(float(s))
    if not vals:
        raise ValueError("No sigma values parsed.")
    return vals


def parse_mode_list(text: str) -> list[str]:
    vals = []
    for part in text.split(","):
        s = part.strip()
        if s:
            vals.append(s)
    if not vals:
        raise ValueError("No source modes parsed.")
    return vals


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mirror-symmetric fake-prime control lab with grid plotting."
    )

    p.add_argument(
        "--sigmas",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70",
    )
    p.add_argument(
        "--source-modes",
        type=str,
        default="real_primes,log_uniform,log_shuffled,random_integers",
    )

    p.add_argument("--linear-gain", type=float, default=0.015)
    p.add_argument("--D", type=float, default=0.03)
    p.add_argument("--beta", type=float, default=1.2)
    p.add_argument("--ell", type=float, default=1.2)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--kappa", type=float, default=0.08)

    p.add_argument("--t-min", type=float, default=-50.0)
    p.add_argument("--t-max", type=float, default=50.0)
    p.add_argument("--n-t", type=int, default=1024)

    p.add_argument("--dtau", type=float, default=0.0015)
    p.add_argument("--n-steps", type=int, default=8000)
    p.add_argument("--save-every", type=int, default=25)

    p.add_argument("--init-noise", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--no-source", action="store_true")
    p.add_argument("--prime-max", type=int, default=3000)
    p.add_argument("--source-amp", type=float, default=0.08)
    p.add_argument("--source-width", type=float, default=0.14)
    p.add_argument("--normalize-source-per-tau", action="store_true")

    p.add_argument("--use-phase-potential", action="store_true")
    p.add_argument("--phase-potential-c2", type=float, default=0.0015)

    p.add_argument("--obs-t-min", type=float, default=-40.0)
    p.add_argument("--obs-t-max", type=float, default=40.0)

    p.add_argument("--no-edge-damping", action="store_true")
    p.add_argument("--edge-cap-start", type=float, default=40.0)
    p.add_argument("--edge-cap-width", type=float, default=10.0)
    p.add_argument("--edge-cap-strength", type=float, default=1.5)
    p.add_argument("--edge-cap-power", type=float, default=2.0)

    p.add_argument("--save-prefix", type=str, default="")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    sigma_values = parse_sigma_list(args.sigmas)
    source_modes = parse_mode_list(args.source_modes)

    base_cfg = ModelConfig(
        sigma=sigma_values[0],
        source_mode=source_modes[0],
        linear_gain=args.linear_gain,
        D=args.D,
        beta=args.beta,
        ell=args.ell,
        gamma=args.gamma,
        kappa=args.kappa,
        use_phase_potential=args.use_phase_potential,
        phase_potential_c2=args.phase_potential_c2,
        t_min=args.t_min,
        t_max=args.t_max,
        n_t=args.n_t,
        dtau=args.dtau,
        n_steps=args.n_steps,
        save_every=args.save_every,
        init_noise=args.init_noise,
        seed=args.seed,
        use_source=not args.no_source,
        prime_max=args.prime_max,
        source_amp=args.source_amp,
        source_width=args.source_width,
        normalize_source_per_tau=args.normalize_source_per_tau,
        obs_t_min=args.obs_t_min,
        obs_t_max=args.obs_t_max,
        use_edge_damping=not args.no_edge_damping,
        edge_cap_start=args.edge_cap_start,
        edge_cap_width=args.edge_cap_width,
        edge_cap_strength=args.edge_cap_strength,
        edge_cap_power=args.edge_cap_power,
    )

    results_by_mode = run_all_modes(
        sigma_values=sigma_values,
        base_cfg=base_cfg,
        source_modes=source_modes,
    )

    prefix = args.save_prefix

    # Compact 4-stacked mode figures
    plot_compact_per_mode_figures(
        results_by_mode,
        output_prefix=f"{prefix}_compact" if prefix else "",
        max_modes_per_figure=4,
    )

    # 4x4 grid: one (metric, mode) per panel, all sigmas inside each panel
    metric_specs = [
        ("m_diff", "M_diff"),
        ("m_diff_norm", "M_diff_norm"),
        ("m_sum", "M_sum"),
        ("channel_overlap", "Channel overlap"),
    ]
    plot_time_series_grid(
        results_by_mode,
        metric_specs=metric_specs,
        nrows=4,
        ncols=4,
        figsize=(20, 18),
        output_path=f"{prefix}_timeseries_grid.png" if prefix else "",
    )

    # Sigma summary comparison across modes
    plot_sigma_summary_grid(
        results_by_mode,
        output_path=f"{prefix}_sigma_summary_grid.png" if prefix else "",
    )

    # Heatmap grids
    plot_heatmap_grid(
        results_by_mode,
        metric_key="m_diff_norm",
        metric_title="M_diff_norm heatmap",
        nrows=2,
        ncols=2,
        figsize=(16, 10),
        output_path=f"{prefix}_heatmap_diffnorm.png" if prefix else "",
    )

    plot_heatmap_grid(
        results_by_mode,
        metric_key="channel_overlap",
        metric_title="Channel-overlap heatmap",
        nrows=2,
        ncols=2,
        figsize=(16, 10),
        output_path=f"{prefix}_heatmap_overlap.png" if prefix else "",
    )

    plot_heatmap_grid(
        results_by_mode,
        metric_key="r_lock",
        metric_title="R_lock heatmap",
        nrows=2,
        ncols=2,
        figsize=(16, 10),
        output_path=f"{prefix}_heatmap_lock.png" if prefix else "",
    )


if __name__ == "__main__":
    main()