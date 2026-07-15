from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from theories.thick_front_entanglement import (
    MM_TO_M,
    run_trf_history_velocity_matrix,
    run_trf_lock_distance_sweep,
    run_trf_lock_velocity_sweep,
    run_trf_no_signalling_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal TRF no-signalling diagnostic for entangled spin settings."
    )
    parser.add_argument("--lambda-signal", type=float, default=0.0)
    parser.add_argument(
        "--history-velocity-mode",
        choices=["constant", "past_superluminal", "symmetric"],
        default="constant",
    )
    parser.add_argument("--history-v-future-factor", type=float, default=0.5)
    parser.add_argument("--history-v-present-factor", type=float, default=1.0)
    parser.add_argument("--history-v-past-factor", type=float, default=2.0)
    parser.add_argument("--distance-m", type=float, default=0.0)
    parser.add_argument("--distance-mm", type=float, default=None, help="Entanglement separation in megameters.")
    parser.add_argument("--history-lock-length-m", type=float, default=float("inf"))
    parser.add_argument("--history-lock-length-mm", type=float, default=None, help="Base L_lock in megameters.")
    parser.add_argument("--history-lock-tau-s", type=float, default=None, help="If set, use L_lock=c*tau.")
    parser.add_argument(
        "--history-lock-decay-mode",
        choices=["none", "exp", "soft_power"],
        default="none",
    )
    parser.add_argument("--history-lock-decay-power", type=float, default=2.0)
    parser.add_argument(
        "--lock-length-mode",
        choices=["constant", "proper_time_dilated", "lorentz_contracted", "anisotropic"],
        default="constant",
    )
    parser.add_argument("--relative-velocity-fraction-c", type=float, default=0.0)
    parser.add_argument("--lock-anisotropy-eta", type=float, default=1.0)
    parser.add_argument("--lock-direction-cos-theta", type=float, default=1.0)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run A-D comparison: constant/safe, past_superluminal/safe, constant/forbidden, past_superluminal/forbidden.",
    )
    parser.add_argument("--distance-sweep", action="store_true")
    parser.add_argument("--velocity-sweep", action="store_true")
    parser.add_argument("--csv-output", default=None, help="Optional CSV output path for sweep rows.")
    parser.add_argument(
        "--lambda-signal-forbidden",
        type=float,
        default=0.4,
        help="Forbidden lambda used for matrix cases C and D.",
    )
    parser.add_argument("--a0", type=float, default=0.0, help="Alice setting for ensemble 0, radians")
    parser.add_argument(
        "--a1",
        type=float,
        default=float(np.pi / 3.0),
        help="Alice setting for ensemble 1, radians",
    )
    parser.add_argument(
        "--bob",
        type=float,
        default=float(np.pi / 5.0),
        help="Fixed Bob setting, radians",
    )
    parser.add_argument("--n-trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    distance_m = float(args.distance_m)
    if args.distance_mm is not None:
        distance_m = float(args.distance_mm) * MM_TO_M

    history_lock_length_m = float(args.history_lock_length_m)
    if args.history_lock_length_mm is not None:
        history_lock_length_m = float(args.history_lock_length_mm) * MM_TO_M

    if args.velocity_sweep:
        result = run_trf_lock_velocity_sweep(
            base_lock_length_m=history_lock_length_m if np.isfinite(history_lock_length_m) else 300.0 * MM_TO_M,
            lock_anisotropy_eta=float(args.lock_anisotropy_eta),
        )
        print("TRF finite-lock velocity sweep")
        for row in result["velocity_sweep"]:
            print(
                f"  beta={row['clamped_beta']:.3f}, "
                f"mode={row['lock_length_mode']}, "
                f"cos(theta)={row['lock_direction_cos_theta']:.3f}, "
                f"gamma={row['gamma']:.6f}, "
                f"L_eff={row['effective_lock_length_Mm']:.3f} Mm"
            )
        if args.csv_output:
            write_csv(args.csv_output, result["velocity_sweep"])
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.distance_sweep:
        sweep_tau_s = args.history_lock_tau_s
        if sweep_tau_s is None and not np.isfinite(history_lock_length_m):
            sweep_tau_s = 1.0
        result = run_trf_lock_distance_sweep(
            lambda_signal_safe=0.0,
            lambda_signal_forbidden=float(args.lambda_signal_forbidden),
            history_velocity_mode=str(args.history_velocity_mode),
            history_lock_tau_s=sweep_tau_s,
            history_lock_length_m=history_lock_length_m,
            history_lock_decay_power=float(args.history_lock_decay_power),
            n_trials=int(args.n_trials),
            rng_seed=int(args.seed),
        )
        print("TRF finite-lock distance sweep")
        for row in result["distance_sweep"]:
            print(
                f"  {row['case']}: "
                f"d={row['distance_Mm']:.1f} Mm, "
                f"A={row['lock_attenuation']:.6f}, "
                f"Delta_signal_exact={row['Delta_signal_exact']:.6f}, "
                f"Delta_corr_exact={row['Delta_correlation_exact']:.6f}"
            )
        if args.csv_output:
            write_csv(args.csv_output, result["distance_sweep"])
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.matrix:
        result = run_trf_history_velocity_matrix(
            lambda_signal_safe=float(args.lambda_signal),
            lambda_signal_forbidden=float(args.lambda_signal_forbidden),
            history_v_future_factor=float(args.history_v_future_factor),
            history_v_present_factor=float(args.history_v_present_factor),
            history_v_past_factor=float(args.history_v_past_factor),
            alice_setting_a0=float(args.a0),
            alice_setting_a1=float(args.a1),
            bob_setting=float(args.bob),
            n_trials=int(args.n_trials),
            rng_seed=int(args.seed),
        )

        print("TRF history-velocity no-signalling matrix")
        for name, case in result["cases"].items():
            hist = case["history_locking"]
            print(
                f"  {name}: "
                f"mode={hist['history_velocity_mode']}, "
                f"lambda={case['lambda_signal']:.6g}, "
                f"effective_lambda={case['effective_lambda_signal']:.6g}, "
                f"Delta_signal_exact={case['Delta_signal_exact']:.6f}, "
                f"past/future={hist['past_future_bias_ratio']:.3f}, "
                f"avg_spread={hist['average_trf_spread_radius']:.3f}"
            )
        print()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    result = run_trf_no_signalling_diagnostic(
        lambda_signal=float(args.lambda_signal),
        history_velocity_mode=str(args.history_velocity_mode),
        history_v_future_factor=float(args.history_v_future_factor),
        history_v_present_factor=float(args.history_v_present_factor),
        history_v_past_factor=float(args.history_v_past_factor),
        distance_m=distance_m,
        history_lock_length_m=history_lock_length_m,
        history_lock_tau_s=args.history_lock_tau_s,
        history_lock_decay_mode=str(args.history_lock_decay_mode),
        history_lock_decay_power=float(args.history_lock_decay_power),
        lock_length_mode=str(args.lock_length_mode),
        relative_velocity_fraction_c=float(args.relative_velocity_fraction_c),
        lock_anisotropy_eta=float(args.lock_anisotropy_eta),
        lock_direction_cos_theta=float(args.lock_direction_cos_theta),
        alice_setting_a0=float(args.a0),
        alice_setting_a1=float(args.a1),
        bob_setting=float(args.bob),
        n_trials=int(args.n_trials),
        rng_seed=int(args.seed),
    )

    print("TRF no-signalling diagnostic")
    print(f"  lambda_signal              : {result['lambda_signal']:.6g}")
    print(f"  effective_lambda_signal    : {result['effective_lambda_signal']:.6g}")
    print(f"  history_velocity_mode      : {result['history_locking']['history_velocity_mode']}")
    print(f"  avg TRF spread radius      : {result['history_locking']['average_trf_spread_radius']:.6f}")
    print(f"  past/future bias ratio     : {result['history_locking']['past_future_bias_ratio']:.6f}")
    print(f"  distance                   : {result['finite_history_lock']['distance_Mm']:.6f} Mm")
    print(f"  lock_attenuation           : {result['finite_history_lock']['lock_attenuation']:.6f}")
    print(f"  effective L_lock           : {result['finite_history_lock']['effective_lock_length_m'] / MM_TO_M:.6f} Mm")
    print(f"  Alice settings             : a0={result['alice_setting_a0']:.6f}, a1={result['alice_setting_a1']:.6f}")
    print(f"  Bob setting                : {result['bob_setting']:.6f}")
    print(f"  n_trials                   : {result['n_trials']}")
    print(f"  P_B_plus_given_a0          : {result['P_B_plus_given_a0']:.6f}")
    print(f"  P_B_plus_given_a1          : {result['P_B_plus_given_a1']:.6f}")
    print(f"  Delta_signal               : {result['Delta_signal']:.6f}")
    print(f"  P_B_plus_given_a0_exact    : {result['P_B_plus_given_a0_exact']:.6f}")
    print(f"  P_B_plus_given_a1_exact    : {result['P_B_plus_given_a1_exact']:.6f}")
    print(f"  Delta_signal_exact         : {result['Delta_signal_exact']:.6f}")
    print()
    print(json.dumps(result, indent=2, sort_keys=True))

    return 0


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
