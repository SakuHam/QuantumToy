"""Profile sigma_T from an exact spatial detector distribution."""

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

from analysis.spatial_response import (
    SpatialResponseExperiment,
    compare_normalization_modes,
    fit_cross_grid_response,
    fit_spatial_response,
    fixed_normalization_scales,
    spatial_convergence,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end sigma_T profile in the measurement-guided spatial model")
    parser.add_argument("--true-sigma", type=float, default=0.1)
    parser.add_argument(
        "--candidate-sigmas", type=float, nargs="+",
        default=[0.06, 0.08, 0.1, 0.12, 0.15])
    parser.add_argument("--response-strength", type=float, default=1.0)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--horizon-sigmas", type=float, default=4.0)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-cross-grid", action="store_true")
    parser.add_argument("--skip-normalization-comparison", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    experiment = SpatialResponseExperiment(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        duration=args.duration,
        response_strength=args.response_strength,
        horizon_sigmas=args.horizon_sigmas,
    )
    if args.skip_normalization_comparison:
        profile = fit_spatial_response(
            experiment, args.true_sigma, args.candidate_sigmas, shots=args.shots)
        normalization_profiles = {experiment.normalization_mode: profile}
    else:
        normalization_profiles = compare_normalization_modes(
            experiment, args.true_sigma, args.candidate_sigmas, shots=args.shots)
        profile = normalization_profiles["per_run_max"]
    convergence = ({} if args.skip_convergence else
                   spatial_convergence(experiment, args.true_sigma))
    cross_grid = (None if args.skip_cross_grid else fit_cross_grid_response(
        experiment, args.true_sigma, args.candidate_sigmas))

    print("Spatial sigma_T profile")
    print(f"  injected sigma_T        = {profile.true_sigma_t:.6g}")
    print(f"  recovered grid sigma_T  = {profile.best_sigma_t:.6g}")
    print(f"  detector mass           = {profile.detector_mass:.6g}")
    print(f"  TV(signal, baseline)    = {profile.baseline_total_variation:.6g}")
    print(f"  null state max error    = {profile.null_state_max_error:.3e}")
    print(f"  null detector max error = {profile.null_distribution_max_error:.3e}")
    print("\nCandidate profile")
    print("sigma_T       KL             expected_deviance")
    for sigma, kl, deviance in zip(
            profile.candidate_sigmas, profile.kl_divergence,
            profile.expected_deviance):
        print(f"{sigma:<13.6g} {kl:<14.6g} {deviance:.6g}")
    if convergence:
        print("\nConvergence: total-variation distance from reference")
        for name, value in convergence.items():
            print(f"  {name:<24} {value:.6g}")
    if cross_grid is not None:
        print("\nCross-grid recovery")
        print(f"  generated on             = {cross_grid.reference_grid}")
        print(f"  fitted on                = {cross_grid.fit_grid}")
        print(f"  raw-distribution sigma_T = {cross_grid.raw_best_sigma_t:.6g}")
        print(f"  excess-response sigma_T  = {cross_grid.best_sigma_t:.6g}")
    if len(normalization_profiles) > 1:
        scales = fixed_normalization_scales(
            experiment.__class__(**{
                **asdict(experiment), "normalization_mode": "fixed"}))
        print("\nNormalization comparison")
        print(f"  fixed effect scale       = {scales.effect:.6g}")
        print(f"  fixed overlap scale      = {scales.overlap:.6g}")
        print("mode             best_sigma    TV(signal,null)  max_deviance")
        for mode, result in normalization_profiles.items():
            print(f"{mode:<16} {result.best_sigma_t:<13.6g} "
                  f"{result.baseline_total_variation:<16.6g} "
                  f"{np.max(result.expected_deviance):.6g}")

    if args.json_output:
        payload = {
            "status": "synthetic_spatial_response_not_empirical_measurement",
            "experiment": asdict(experiment),
            "profile": asdict(profile),
            "convergence_total_variation": convergence,
            "cross_grid_recovery": (
                None if cross_grid is None else asdict(cross_grid)),
            "normalization_profiles": {
                mode: asdict(result)
                for mode, result in normalization_profiles.items()
            },
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, default=_json_value, indent=2) + "\n",
            encoding="utf-8")
        print(f"\nWrote {args.json_output}")


if __name__ == "__main__":
    main()
