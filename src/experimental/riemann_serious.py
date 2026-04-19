from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


# ============================================================
# Model config
# ============================================================

@dataclass
class ModelConfig:
    sigma: float

    # Base linear gain (same for both channels)
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

    # Prime source
    use_prime_source: bool = True
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
# Prime forcing
# ============================================================

def build_prime_tables(cfg: ModelConfig, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute:
        logp
        exp(-i t log p)
    """
    primes = sieve_primes(cfg.prime_max)
    if primes.size == 0:
        return np.array([], dtype=float), np.zeros((0, t.size), dtype=np.complex128)

    logp = np.log(primes.astype(float))
    phase_table = np.exp(-1j * np.outer(logp, t))
    return logp, phase_table


def compute_prime_source(
    tau: float,
    sigma_weight: float,
    source_amp: float,
    source_width: float,
    logp: np.ndarray,
    phase_table: np.ndarray,
    normalize_per_tau: bool,
) -> np.ndarray:
    """
    S_sigma(t,tau) = A * sum_p w_tau(log p) * p^{-sigma_weight} * exp(-i t log p)
    """
    if logp.size == 0:
        return np.zeros(phase_table.shape[1], dtype=np.complex128)

    window = np.exp(-0.5 * ((logp - tau) / source_width) ** 2)
    weights = window * np.exp(-sigma_weight * logp)

    if normalize_per_tau:
        norm = np.sqrt(np.sum(weights ** 2))
        if norm > 1e-15:
            weights = weights / norm

    weights = source_amp * weights
    return (weights[:, None] * phase_table).sum(axis=0)


# ============================================================
# Simulation
# ============================================================

def simulate(cfg: ModelConfig) -> dict:
    """
    Two-channel symmetric mirror model:

      dPsi+/dtau = L[Psi+] + kappa (Psi- - Psi+) + S_sigma
      dPsi-/dtau = L[Psi-] + kappa (Psi+ - Psi-) + S_{1-sigma}

    where L includes diffusion, competition, saturation, edge damping,
    and optional phase potential.
    """
    rng = np.random.default_rng(cfg.seed)

    t = np.linspace(cfg.t_min, cfg.t_max, cfg.n_t, endpoint=False)
    dt = t[1] - t[0]

    obs_mask = (t >= cfg.obs_t_min) & (t <= cfg.obs_t_max)

    kernel_hat = gaussian_kernel_fft(cfg.n_t, dt=dt, ell=cfg.ell)
    logp, phase_table = build_prime_tables(cfg, t)

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

    energy_p = np.zeros(n_save, dtype=float)
    energy_m = np.zeros(n_save, dtype=float)
    energy_sum = np.zeros(n_save, dtype=float)
    energy_diff = np.zeros(n_save, dtype=float)

    energy_obs_p = np.zeros(n_save, dtype=float)
    energy_obs_m = np.zeros(n_save, dtype=float)
    energy_obs_sum = np.zeros(n_save, dtype=float)
    energy_obs_diff = np.zeros(n_save, dtype=float)

    lock_ratio = np.zeros(n_save, dtype=float)

    overlap_src_p = np.zeros(n_save, dtype=float)
    overlap_src_m = np.zeros(n_save, dtype=float)

    source_energy_p = np.zeros(n_save, dtype=float)
    source_energy_m = np.zeros(n_save, dtype=float)

    maxabs_p = np.zeros(n_save, dtype=float)
    maxabs_m = np.zeros(n_save, dtype=float)

    def save(idx: int, tau: float, field_p: np.ndarray, field_m: np.ndarray,
             src_p: np.ndarray, src_m: np.ndarray) -> None:
        fp_obs = field_p[obs_mask]
        fm_obs = field_m[obs_mask]

        sum_field = field_p + field_m
        diff_field = field_p - field_m

        sum_obs = fp_obs + fm_obs
        diff_obs = fp_obs - fm_obs

        frames_p[idx] = field_p
        frames_m[idx] = field_m
        taus[idx] = tau

        energy_p[idx] = l2_energy_complex(field_p, dt)
        energy_m[idx] = l2_energy_complex(field_m, dt)
        energy_sum[idx] = l2_energy_complex(sum_field, dt)
        energy_diff[idx] = l2_energy_complex(diff_field, dt)

        energy_obs_p[idx] = l2_energy_complex(fp_obs, dt)
        energy_obs_m[idx] = l2_energy_complex(fm_obs, dt)
        energy_obs_sum[idx] = l2_energy_complex(sum_obs, dt)
        energy_obs_diff[idx] = l2_energy_complex(diff_obs, dt)

        lock_ratio[idx] = float(energy_obs_sum[idx] / (energy_obs_diff[idx] + 1e-12))

        overlap_src_p[idx] = complex_overlap(fp_obs, src_p[obs_mask], dt)
        overlap_src_m[idx] = complex_overlap(fm_obs, src_m[obs_mask], dt)

        source_energy_p[idx] = l2_energy_complex(src_p[obs_mask], dt)
        source_energy_m[idx] = l2_energy_complex(src_m[obs_mask], dt)

        maxabs_p[idx] = float(np.max(np.abs(field_p)))
        maxabs_m[idx] = float(np.max(np.abs(field_m)))

    src0_p = (
        compute_prime_source(
            tau=0.0,
            sigma_weight=cfg.sigma,
            source_amp=cfg.source_amp,
            source_width=cfg.source_width,
            logp=logp,
            phase_table=phase_table,
            normalize_per_tau=cfg.normalize_source_per_tau,
        )
        if cfg.use_prime_source
        else np.zeros_like(psi_p)
    )

    src0_m = (
        compute_prime_source(
            tau=0.0,
            sigma_weight=(1.0 - cfg.sigma),
            source_amp=cfg.source_amp,
            source_width=cfg.source_width,
            logp=logp,
            phase_table=phase_table,
            normalize_per_tau=cfg.normalize_source_per_tau,
        )
        if cfg.use_prime_source
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
            compute_prime_source(
                tau=tau,
                sigma_weight=cfg.sigma,
                source_amp=cfg.source_amp,
                source_width=cfg.source_width,
                logp=logp,
                phase_table=phase_table,
                normalize_per_tau=cfg.normalize_source_per_tau,
            )
            if cfg.use_prime_source
            else np.zeros_like(psi_p)
        )

        src_m = (
            compute_prime_source(
                tau=tau,
                sigma_weight=(1.0 - cfg.sigma),
                source_amp=cfg.source_amp,
                source_width=cfg.source_width,
                logp=logp,
                phase_table=phase_table,
                normalize_per_tau=cfg.normalize_source_per_tau,
            )
            if cfg.use_prime_source
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
                f"Non-finite values encountered for sigma={cfg.sigma:.4f} at step={step}"
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
        "energy_p": energy_p,
        "energy_m": energy_m,
        "energy_sum": energy_sum,
        "energy_diff": energy_diff,
        "energy_obs_p": energy_obs_p,
        "energy_obs_m": energy_obs_m,
        "energy_obs_sum": energy_obs_sum,
        "energy_obs_diff": energy_obs_diff,
        "lock_ratio": lock_ratio,
        "overlap_src_p": overlap_src_p,
        "overlap_src_m": overlap_src_m,
        "source_energy_p": source_energy_p,
        "source_energy_m": source_energy_m,
        "maxabs_p": maxabs_p,
        "maxabs_m": maxabs_m,
    }


# ============================================================
# Sweep / metrics
# ============================================================

def sweep_sigmas(sigma_values: list[float], base_cfg: ModelConfig) -> list[dict]:
    results = []

    for i, sigma in enumerate(sigma_values):
        cfg = ModelConfig(
            sigma=sigma,
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
            use_prime_source=base_cfg.use_prime_source,
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

        print("=" * 92)
        print(f"Running sigma={cfg.sigma:.4f}  and mirror 1-sigma={1.0-cfg.sigma:.4f}")
        print(
            f"gain={cfg.linear_gain:.5f}, D={cfg.D:.4f}, beta={cfg.beta:.4f}, "
            f"ell={cfg.ell:.3f}, gamma={cfg.gamma:.4f}, kappa={cfg.kappa:.4f}"
        )
        print(
            f"prime_source={cfg.use_prime_source}, prime_max={cfg.prime_max}, "
            f"source_amp={cfg.source_amp}, source_width={cfg.source_width}, "
            f"normalize_source_per_tau={cfg.normalize_source_per_tau}"
        )
        print(
            f"edge_damping={cfg.use_edge_damping}, cap_start={cfg.edge_cap_start}, "
            f"cap_width={cfg.edge_cap_width}, cap_strength={cfg.edge_cap_strength}, "
            f"cap_power={cfg.edge_cap_power}"
        )

        res = simulate(cfg)
        results.append(res)

        print(
            f"[done] sigma={cfg.sigma:.4f} | "
            f"final_diff={res['energy_obs_diff'][-1]:.6e} | "
            f"final_sum={res['energy_obs_sum'][-1]:.6e} | "
            f"final_lock={res['lock_ratio'][-1]:.6e} | "
            f"max_lock={np.max(res['lock_ratio']):.6e}"
        )

    return results


def compute_summary_metrics(results: list[dict]) -> dict:
    sigmas = np.array([res["cfg"].sigma for res in results], dtype=float)

    final_diff = np.array([res["energy_obs_diff"][-1] for res in results], dtype=float)
    max_diff = np.array([np.max(res["energy_obs_diff"]) for res in results], dtype=float)

    final_sum = np.array([res["energy_obs_sum"][-1] for res in results], dtype=float)
    max_sum = np.array([np.max(res["energy_obs_sum"]) for res in results], dtype=float)

    final_lock = np.array([res["lock_ratio"][-1] for res in results], dtype=float)
    max_lock = np.array([np.max(res["lock_ratio"]) for res in results], dtype=float)

    final_overlap_p = np.array([res["overlap_src_p"][-1] for res in results], dtype=float)
    final_overlap_m = np.array([res["overlap_src_m"][-1] for res in results], dtype=float)
    max_overlap_p = np.array([np.max(res["overlap_src_p"]) for res in results], dtype=float)
    max_overlap_m = np.array([np.max(res["overlap_src_m"]) for res in results], dtype=float)

    return {
        "sigmas": sigmas,
        "final_diff": final_diff,
        "max_diff": max_diff,
        "final_sum": final_sum,
        "max_sum": max_sum,
        "final_lock": final_lock,
        "max_lock": max_lock,
        "final_overlap_p": final_overlap_p,
        "final_overlap_m": final_overlap_m,
        "max_overlap_p": max_overlap_p,
        "max_overlap_m": max_overlap_m,
    }


# ============================================================
# Visualization
# ============================================================

def chunk_results(results: list[dict], chunk_size: int = 4) -> list[list[dict]]:
    """Split results into chunks of at most chunk_size."""
    return [results[i:i + chunk_size] for i in range(0, len(results), chunk_size)]

def make_summary_figure(
    results: list[dict],
    output_prefix: str = "",
    max_curves_per_figure: int = 4,
) -> None:
    """
    Plot summary curves in groups of at most max_curves_per_figure results.
    Uses wide figures and stacked subplots to avoid clutter.
    """
    groups = chunk_results(results, chunk_size=max_curves_per_figure)

    for gi, group in enumerate(groups, start=1):
        fig, axes = plt.subplots(5, 1, figsize=(16, 18), constrained_layout=True)
        ax0, ax1, ax2, ax3, ax4 = axes

        for res in group:
            cfg = res["cfg"]
            t = res["t"]
            taus = res["taus"]

            final_p = res["frames_p"][-1]
            final_m = res["frames_m"][-1]

            label = f"sigma={cfg.sigma:.2f}"

            ax0.plot(t, np.abs(final_p), label=label + " : |Psi+|")
            ax0.plot(t, np.abs(final_m), ls="--", alpha=0.85, label=label + " : |Psi-|")

            ax1.plot(taus, res["energy_obs_diff"], label=label)
            ax2.plot(taus, res["energy_obs_sum"], label=label)
            ax3.plot(taus, res["lock_ratio"], label=label)
            ax4.plot(taus, res["overlap_src_p"], label=label + " : ov+")
            ax4.plot(taus, res["overlap_src_m"], ls="--", alpha=0.85, label=label + " : ov-")

        ax0.set_title(f"Final amplitudes |Psi+| and |Psi-| (group {gi}/{len(groups)})")
        ax0.set_xlabel("t")
        ax0.set_ylabel("amplitude")
        ax0.grid(True, alpha=0.3)
        ax0.legend(ncol=2, fontsize=9)

        ax1.set_title("Observed antisymmetric energy M_diff = ∫|Psi+ - Psi-|² dt")
        ax1.set_xlabel("tau")
        ax1.set_ylabel("M_diff")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_title("Observed symmetric energy M_sum = ∫|Psi+ + Psi-|² dt")
        ax2.set_xlabel("tau")
        ax2.set_ylabel("M_sum")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3.set_title("Lock ratio R_lock = M_sum / (M_diff + eps)")
        ax3.set_xlabel("tau")
        ax3.set_ylabel("R_lock")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        ax4.set_title("Source overlaps")
        ax4.set_xlabel("tau")
        ax4.set_ylabel("overlap")
        ax4.grid(True, alpha=0.3)
        ax4.legend(ncol=2, fontsize=9)

        if output_prefix:
            out = Path(f"{output_prefix}_summary_group_{gi}.png")
            fig.savefig(out, dpi=150)
            print(f"[saved] summary group -> {out}")

        plt.show()

def make_phase_diagram(results: list[dict], output_prefix: str = "") -> None:
    metrics = compute_summary_metrics(results)
    sigmas = metrics["sigmas"]

    final_diff = metrics["final_diff"]
    max_diff = metrics["max_diff"]
    final_sum = metrics["final_sum"]
    max_sum = metrics["max_sum"]
    final_lock = metrics["final_lock"]
    max_lock = metrics["max_lock"]

    final_overlap_p = metrics["final_overlap_p"]
    final_overlap_m = metrics["final_overlap_m"]
    max_overlap_p = metrics["max_overlap_p"]
    max_overlap_m = metrics["max_overlap_m"]

    taus = results[0]["taus"]
    diff_map = np.stack([res["energy_obs_diff"] for res in results], axis=0)
    sum_map = np.stack([res["energy_obs_sum"] for res in results], axis=0)
    lock_map = np.stack([res["lock_ratio"] for res in results], axis=0)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(sigmas, final_diff, marker="o")
    axes[0].axvline(0.5, ls="--", alpha=0.7)
    axes[0].set_title("Final M_diff vs sigma")
    axes[0].set_xlabel("sigma")
    axes[0].set_ylabel("final M_diff")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigmas, max_diff, marker="o")
    axes[1].axvline(0.5, ls="--", alpha=0.7)
    axes[1].set_title("Max M_diff vs sigma")
    axes[1].set_xlabel("sigma")
    axes[1].set_ylabel("max M_diff")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(sigmas, final_sum, marker="o")
    axes[2].axvline(0.5, ls="--", alpha=0.7)
    axes[2].set_title("Final M_sum vs sigma")
    axes[2].set_xlabel("sigma")
    axes[2].set_ylabel("final M_sum")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(sigmas, max_sum, marker="o")
    axes[3].axvline(0.5, ls="--", alpha=0.7)
    axes[3].set_title("Max M_sum vs sigma")
    axes[3].set_xlabel("sigma")
    axes[3].set_ylabel("max M_sum")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(sigmas, final_lock, marker="o", label="final lock")
    axes[4].plot(sigmas, max_lock, marker="s", label="max lock")
    axes[4].axvline(0.5, ls="--", alpha=0.7)
    axes[4].set_title("Lock metrics vs sigma")
    axes[4].set_xlabel("sigma")
    axes[4].set_ylabel("R_lock")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].plot(sigmas, final_overlap_p, marker="o", label="final overlap +")
    axes[5].plot(sigmas, final_overlap_m, marker="s", label="final overlap -")
    axes[5].plot(sigmas, max_overlap_p, marker="^", label="max overlap +")
    axes[5].plot(sigmas, max_overlap_m, marker="v", label="max overlap -")
    axes[5].axvline(0.5, ls="--", alpha=0.7)
    axes[5].set_title("Source overlaps vs sigma")
    axes[5].set_xlabel("sigma")
    axes[5].set_ylabel("overlap")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(fontsize=8)

    im1 = axes[6].imshow(
        diff_map,
        aspect="auto",
        origin="lower",
        extent=[taus[0], taus[-1], sigmas[0], sigmas[-1]],
    )
    axes[6].axhline(0.5, ls="--", alpha=0.7)
    axes[6].set_title("M_diff heatmap")
    axes[6].set_xlabel("tau")
    axes[6].set_ylabel("sigma")
    fig.colorbar(im1, ax=axes[6], label="M_diff")

    im2 = axes[7].imshow(
        lock_map,
        aspect="auto",
        origin="lower",
        extent=[taus[0], taus[-1], sigmas[0], sigmas[-1]],
    )
    axes[7].axhline(0.5, ls="--", alpha=0.7)
    axes[7].set_title("R_lock heatmap")
    axes[7].set_xlabel("tau")
    axes[7].set_ylabel("sigma")
    fig.colorbar(im2, ax=axes[7], label="R_lock")

    if output_prefix:
        out = Path(f"{output_prefix}_phase_diagram.png")
        fig.savefig(out, dpi=150)
        print(f"[saved] phase diagram -> {out}")

    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    im3 = axes2[0].imshow(
        sum_map,
        aspect="auto",
        origin="lower",
        extent=[taus[0], taus[-1], sigmas[0], sigmas[-1]],
    )
    axes2[0].axhline(0.5, ls="--", alpha=0.7)
    axes2[0].set_title("M_sum heatmap")
    axes2[0].set_xlabel("tau")
    axes2[0].set_ylabel("sigma")
    fig2.colorbar(im3, ax=axes2[0], label="M_sum")

    im4 = axes2[1].imshow(
        lock_map,
        aspect="auto",
        origin="lower",
        extent=[taus[0], taus[-1], sigmas[0], sigmas[-1]],
    )
    axes2[1].axhline(0.5, ls="--", alpha=0.7)
    axes2[1].set_title("R_lock heatmap")
    axes2[1].set_xlabel("tau")
    axes2[1].set_ylabel("sigma")
    fig2.colorbar(im4, ax=axes2[1], label="R_lock")

    if output_prefix:
        out2 = Path(f"{output_prefix}_extra_heatmaps.png")
        fig2.savefig(out2, dpi=150)
        print(f"[saved] extra heatmaps -> {out2}")

    plt.show()

    idx_diff = int(np.argmin(final_diff))
    idx_lock = int(np.argmax(max_lock))
    idx_sum = int(np.argmax(max_sum))

    print("=" * 92)
    print("Best-sigma summary for mirror model")
    print(f"Best sigma by smallest final M_diff : {sigmas[idx_diff]:.4f}")
    print(f"Best sigma by largest max R_lock    : {sigmas[idx_lock]:.4f}")
    print(f"Best sigma by largest max M_sum     : {sigmas[idx_sum]:.4f}")
    print("=" * 92)


def make_animation(
    results: list[dict],
    interval_ms: int = 60,
    repeat: bool = True,
    output_prefix: str = "",
    max_curves_per_figure: int = 4,
) -> None:
    groups = chunk_results(results, chunk_size=max_curves_per_figure)

    for gi, group in enumerate(groups, start=1):
        n_panels = len(group)
        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(16, max(3.2, 3.3 * n_panels)),
            sharex=True,
            constrained_layout=True,
        )

        if n_panels == 1:
            axes = [axes]

        global_amp_max = 0.0
        for res in group:
            global_amp_max = max(global_amp_max, float(np.max(np.abs(res["frames_p"]))))
            global_amp_max = max(global_amp_max, float(np.max(np.abs(res["frames_m"]))))

        y_max = 1.08 * global_amp_max if global_amp_max > 0 else 1.0

        lines_p = []
        lines_m = []
        texts = []

        for ax, res in zip(axes, group):
            cfg = res["cfg"]
            t = res["t"]

            line_p, = ax.plot(t, np.abs(res["frames_p"][0]), lw=2.0, label="|Psi+|")
            line_m, = ax.plot(t, np.abs(res["frames_m"][0]), lw=2.0, ls="--", label="|Psi-|")

            lines_p.append(line_p)
            lines_m.append(line_m)

            ax.set_ylim(0.0, y_max)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("amplitude")
            ax.set_title(
                f"sigma={cfg.sigma:.3f}, mirror={1.0-cfg.sigma:.3f}, "
                f"kappa={cfg.kappa:.3f}"
            )
            ax.legend(loc="upper right")

            txt = ax.text(
                0.01,
                0.97,
                "",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            texts.append(txt)

        axes[-1].set_xlabel("t")
        n_frames = min(res["frames_p"].shape[0] for res in group)

        def update(frame_idx: int):
            artists = []
            for res, line_p, line_m, txt in zip(group, lines_p, lines_m, texts):
                psi_p = res["frames_p"][frame_idx]
                psi_m = res["frames_m"][frame_idx]

                tau = float(res["taus"][frame_idx])
                mdiff = float(res["energy_obs_diff"][frame_idx])
                msum = float(res["energy_obs_sum"][frame_idx])
                lock = float(res["lock_ratio"][frame_idx])
                ovp = float(res["overlap_src_p"][frame_idx])
                ovm = float(res["overlap_src_m"][frame_idx])

                line_p.set_ydata(np.abs(psi_p))
                line_m.set_ydata(np.abs(psi_m))

                txt.set_text(
                    f"tau={tau:.3f}\n"
                    f"M_diff={mdiff:.4e}\n"
                    f"M_sum={msum:.4e}\n"
                    f"R_lock={lock:.4e}\n"
                    f"ov+={ovp:.4e}\n"
                    f"ov-={ovm:.4e}"
                )
                artists.extend([line_p, line_m, txt])

            return artists

        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval_ms,
            blit=False,
            repeat=repeat,
        )

        if output_prefix:
            out = Path(f"{output_prefix}_anim_group_{gi}.gif")
            anim.save(out, writer="pillow", fps=max(1, int(1000 / interval_ms)))
            print(f"[saved] animation group -> {out}")

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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-channel mirror-symmetric prime-driven PDE lab."
    )

    p.add_argument("--sigmas", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")

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

    p.add_argument("--no-prime-source", action="store_true")
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

    p.add_argument("--no-animation", action="store_true")
    p.add_argument("--save-animation", type=str, default="")
    p.add_argument("--save-summary", type=str, default="")
    p.add_argument("--save-prefix", type=str, default="")

    return p


def main() -> None:
    args = build_argparser().parse_args()
    sigma_values = parse_sigma_list(args.sigmas)

    base_cfg = ModelConfig(
        sigma=sigma_values[0],
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
        use_prime_source=not args.no_prime_source,
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

    results = sweep_sigmas(sigma_values, base_cfg=base_cfg)

    summary_prefix = args.save_summary if args.save_summary else ""
    make_summary_figure(
        results,
        output_prefix=summary_prefix,
        max_curves_per_figure=4,
    )

    make_phase_diagram(results, output_prefix=args.save_prefix)

    if not args.no_animation:
        anim_prefix = args.save_animation if args.save_animation else ""
        make_animation(
            results,
            output_prefix=anim_prefix,
            max_curves_per_figure=4,
        )

if __name__ == "__main__":
    main()