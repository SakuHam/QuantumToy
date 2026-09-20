"""Measure sigma_T from a temporal response and test alpha on held-out g."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis.record_environment import RecordExperiment
from analysis.record_sensitivity import build_reference_runs
from analysis.trf_response import (
    ResponseSetting,
    excess_dephasing,
    fit_heldout_width,
    fit_response,
    half_rise_time,
    simulate_response_counts,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Independent sigma_T response measurement and held-out alpha test")
    parser.add_argument("--shots-per-setting", type=int, default=10_000)
    parser.add_argument("--lambda-signal", type=float, default=0.2)
    parser.add_argument("--alpha-signal", type=float, default=1.0)
    parser.add_argument("--ordinary-dephasing-rate", type=float, default=0.02)
    parser.add_argument("--ordinary-dephasing-sigma", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if (args.shots_per_setting <= 0 or args.lambda_signal < 0
            or args.alpha_signal <= 0 or args.ordinary_dephasing_rate < 0
            or args.ordinary_dephasing_sigma <= 0):
        parser.error("Invalid response-study parameters")

    experiment = RecordExperiment()
    runs = build_reference_runs(
        experiment, [0.5, 1.0, 2.0], duration=12, dt=0.02)
    tau_by_g = {run.g: run.stabilization.latency for run in runs}
    sigma_by_g = {g: args.alpha_signal * tau for g, tau in tau_by_g.items()}
    fractions = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5, 4.0]
    phases = [0.0, np.pi / 2]
    settings = tuple(
        ResponseSetting(g, fraction * tau_by_g[g], phase,
                        args.shots_per_setting)
        for g in sorted(tau_by_g) for fraction in fractions for phase in phases
    )
    counts = simulate_response_counts(
        experiment, settings,
        lambda_strength=args.lambda_signal,
        sigma_by_g=sigma_by_g,
        ordinary_dephasing_rate=args.ordinary_dephasing_rate,
        rng=np.random.default_rng(args.seed),
    )

    calibration_indices = [i for i, setting in enumerate(settings)
                           if setting.g == 1.0]
    calibration = fit_response(
        experiment,
        [settings[i] for i in calibration_indices],
        [counts[i] for i in calibration_indices],
        tau_by_g=tau_by_g,
        mode="universal_alpha",
        gamma_mean=args.ordinary_dephasing_rate,
        gamma_sigma=args.ordinary_dephasing_sigma,
    )
    calibrated_alpha = calibration.parameters[1]
    calibration_sigma = calibration.sigma_by_g[1.0]
    print("Calibration at g=1")
    print(f"  lambda_hat   = {calibration.lambda_strength:.6g}")
    print(f"  sigma_T_hat  = {calibration_sigma:.6g}")
    print(f"  tau_stab     = {tau_by_g[1.0]:.6g}")
    print(f"  alpha_hat    = {calibrated_alpha:.6g}")
    print(f"  gamma_hat    = {calibration.ordinary_dephasing_rate:.6g}")
    print(f"  t50_hat      = {half_rise_time(calibration_sigma):.6g}")

    heldout = []
    print("\nHeld-out prediction using sigma_T=alpha_cal*tau_stab")
    print("g       sigma_pred  sigma_fit   alpha_fit   sigma_SE    conditional_LR")
    for g in [0.5, 2.0]:
        indices = [i for i, setting in enumerate(settings) if setting.g == g]
        predicted_sigma = calibrated_alpha * tau_by_g[g]
        result = fit_heldout_width(
            experiment,
            [settings[i] for i in indices],
            [counts[i] for i in indices],
            lambda_strength=calibration.lambda_strength,
            ordinary_dephasing_rate=calibration.ordinary_dephasing_rate,
            predicted_sigma_t=predicted_sigma,
        )
        pair_indices = [i for i, setting in enumerate(settings)
                        if setting.g in (1.0, g)]
        pair_settings = [settings[i] for i in pair_indices]
        pair_counts = [counts[i] for i in pair_indices]
        pair_universal = fit_response(
            experiment, pair_settings, pair_counts,
            tau_by_g=tau_by_g, mode="universal_alpha",
            gamma_mean=args.ordinary_dephasing_rate,
            gamma_sigma=args.ordinary_dephasing_sigma)
        pair_free = fit_response(
            experiment, pair_settings, pair_counts,
            tau_by_g=tau_by_g, mode="free_widths",
            gamma_mean=args.ordinary_dephasing_rate,
            gamma_sigma=args.ordinary_dephasing_sigma,
            initial_lambda=pair_universal.lambda_strength,
            initial_alpha=pair_universal.parameters[1])
        pair_lr = max(0.0, 2 * (
            pair_universal.negative_log_posterior
            - pair_free.negative_log_posterior))
        heldout.append({
            "conditional_width_fit": asdict(result),
            "pairwise_universal_fit": asdict(pair_universal),
            "pairwise_free_width_fit": asdict(pair_free),
            "pairwise_universality_likelihood_ratio": pair_lr,
        })
        error = float("nan") if result.standard_error is None else result.standard_error
        print(f"{g:<7g} {predicted_sigma:<11.6g} {result.sigma_t:<11.6g} "
              f"{result.sigma_t/tau_by_g[g]:<11.6g} {error:<11.6g} "
              f"{result.prediction_likelihood_ratio:.6g}")
        print(f"        pairwise LR with calibration uncertainty (df=1): {pair_lr:.6g}")

    universal = fit_response(
        experiment, settings, counts, tau_by_g=tau_by_g,
        mode="universal_alpha",
        gamma_mean=args.ordinary_dephasing_rate,
        gamma_sigma=args.ordinary_dephasing_sigma,
    )
    free = fit_response(
        experiment, settings, counts, tau_by_g=tau_by_g,
        mode="free_widths",
        gamma_mean=args.ordinary_dephasing_rate,
        gamma_sigma=args.ordinary_dephasing_sigma,
        initial_lambda=universal.lambda_strength,
        initial_alpha=universal.parameters[1],
    )
    universality_lr = max(
        0.0, 2 * (universal.negative_log_posterior
                  - free.negative_log_posterior))
    print("\nAll-coupling consistency")
    print(f"  universal alpha_hat = {universal.parameters[1]:.6g}")
    print("  free alpha hats     =", {
        g: round(value, 6) for g, value in free.alpha_by_g(tau_by_g).items()})
    print(f"  LR(universal vs free widths, df=2) = {universality_lr:.6g}")

    curve_times = np.asarray(fractions) * tau_by_g[1.0]
    curve = excess_dephasing(
        curve_times, calibration.lambda_strength, calibration_sigma)
    if args.json_output:
        payload = {
            "status": "synthetic_response_not_empirical_measurement",
            "settings": vars(args) | {"json_output": None},
            "experiment": asdict(experiment),
            "tau_by_g": tau_by_g,
            "true_sigma_by_g": sigma_by_g,
            "calibration": asdict(calibration),
            "calibrated_alpha": calibrated_alpha,
            "heldout": heldout,
            "joint_universal_fit": asdict(universal),
            "joint_free_width_fit": asdict(free),
            "universality_likelihood_ratio": universality_lr,
            "calibration_response_curve": {
                "time": curve_times,
                "excess_dephasing": curve,
                "normalized": curve / max(calibration.lambda_strength, 1e-15),
            },
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
