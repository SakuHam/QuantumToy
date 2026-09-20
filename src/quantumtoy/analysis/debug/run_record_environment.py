"""Run the declared weak-probe and environment-record experiment."""

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

from analysis.record_environment import RecordExperiment, simulate_record_formation


def _json_value(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Exact two-path weak-probe/environment record sweep")
    parser.add_argument("--g", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                        help="Environment coupling strengths")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--probe-strength", type=float, default=0.4)
    parser.add_argument("--information-deficit", type=float, default=0.1)
    parser.add_argument("--required-copies", type=int, default=3)
    parser.add_argument("--coherence-tolerance", type=float, default=0.05)
    parser.add_argument("--hold-time", type=float, default=0.5)
    parser.add_argument("--json-output", type=Path,
                        help="Optional path for protocol, time series, and full joint tables")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not np.isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt must be finite and positive")
    if any(not np.isfinite(g) or g < 0 for g in args.g):
        parser.error("Every --g value must be finite and nonnegative")

    experiment = RecordExperiment(
        probe_strength=args.probe_strength,
        information_deficit=args.information_deficit,
        required_copies=args.required_copies,
        coherence_tolerance=args.coherence_tolerance,
        hold_time=args.hold_time,
    )
    times = np.arange(0, args.duration + args.dt / 2, args.dt)
    runs = [simulate_record_formation(experiment, g, times) for g in args.g]

    print("g       tau_stab  confirmed  R_final  coherence_final  I_readout_min  joint_sum")
    for run in runs:
        window = run.stabilization
        latency = "unresolved" if window is None else f"{window.latency:.6g}"
        confirmed = "unresolved" if window is None else f"{window.confirmed_at:.6g}"
        print(f"{run.g:<7g} {latency:<9} {confirmed:<10} {run.redundancy[-1]:<8d} "
              f"{run.coherence[-1]:<16.6g} {run.readout_information[-1].min():<14.6g} "
              f"{run.joint[-1].sum():.12g}")

    if args.json_output:
        payload = {
            "model": "quantum_reference_only",
            "record_units": "nats",
            "time_units": "declared laboratory time unit",
            "axes": {
                "joint": ["time", "probe(+,-)", "fragment_record",
                          "detector(+,-,no_click)"],
                "fragment_record": "fixed-width binary y_1...y_N; 0=+X, 1=-X",
                "readout_information": ["time", "fragment"],
                "system_density": ["time", "path_row", "path_column"],
            },
            "experiment": asdict(experiment),
            "runs": [{
                "g": run.g,
                "times": run.times,
                "joint": run.joint,
                "system_density": run.system_density,
                "readout_information": run.readout_information,
                "holevo_information": run.holevo_information,
                "all_fragments_information": run.all_fragments_information,
                "coherence": run.coherence,
                "redundancy": run.redundancy,
                "stable": run.stable,
                "stabilization": None if run.stabilization is None
                    else asdict(run.stabilization),
            } for run in runs],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, default=_json_value, indent=2) + "\n",
                                    encoding="utf-8")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
