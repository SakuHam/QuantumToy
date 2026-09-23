"""Compare nested fit grids on identical synthetic control and test records."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from analysis import spatial_detector_inference as inference
from analysis.debug.run_spatial_detector_inference_study import (
    FIT_LAMBDAS, FIT_SIGMAS, JITTER_GRID, TRUE_RESPONSE,
    _calibration_experiments, _json_value,
)
from analysis.spatial_effect_measurement import double_slit_effect_experiment


_LIBRARIES = None


def _initialize(coarse, refined):
    global _LIBRARIES
    _LIBRARIES = (coarse, refined)


def _compare(record):
    first, second, difference = inference.compare_spatial_detector_grids(
        *_LIBRARIES, record["calibration_counts"], record["test_counts"])
    # Surface arrays are unnecessary for the replicate log; retain fitted
    # parameters, null/best optimizer diagnostics, and common-point checks.
    return record | {
        "coarse": asdict(inference._replicate_record(first, record["seed"])),
        "refined": asdict(inference._replicate_record(second, record["seed"])),
        "difference": difference,
    }


def _midpoints(grid):
    grid = np.asarray(grid, dtype=float)
    return np.sort(np.concatenate([grid, (grid[:-1] + grid[1:]) / 2]))


def _summaries(records):
    summaries = []
    for strength in sorted({record["true_lambda_strength"] for record in records}):
        rows = [r for r in records if r["true_lambda_strength"] == strength]
        summary = {"true_lambda_strength": strength, "repetitions": len(rows)}
        changes = np.array([r["difference"]["delta_likelihood_ratio"] for r in rows])
        summary["delta_lr_median"] = float(np.median(changes))
        summary["delta_lr_max_abs"] = float(np.max(np.abs(changes)))
        summary["changed_parameter_pairs"] = sum(
            (r["coarse"]["best_sigma_t"], r["coarse"]["best_lambda_strength"])
            != (r["refined"]["best_sigma_t"], r["refined"]["best_lambda_strength"])
            for r in rows)
        summary["max_abs_null_nll_difference"] = max(
            abs(r["difference"]["delta_null_nll"]) for r in rows)
        summary["max_abs_common_profile_nll_difference"] = max(
            r["difference"]["max_abs_common_profile_nll_difference"] for r in rows)
        summary["nested_minimum_violations"] = sum(
            r["difference"]["nested_minimum_violation"] for r in rows)
        for label in ("coarse", "refined"):
            fits = [r[label] for r in rows]
            diagnostics = [f["diagnostics"] for f in fits]
            summary[label] = {
                "successful_best_fits": sum(d["best"]["success"] for d in diagnostics),
                "successful_null_fits": sum(d["null"]["success"] for d in diagnostics),
                "failed_profile_optimizations": sum(d["failed_profile_fits"] for d in diagnostics),
                "total_profile_optimizations": sum(d["profile_fits"] for d in diagnostics),
                "failed_calibration_optimizations": sum(
                    not c["success"] for d in diagnostics for c in d["calibration"]),
                "nonfinite_profile_optimizations": sum(d["nonfinite_profile_fits"] for d in diagnostics),
                "lr_min_median_max": [float(v) for v in np.quantile(
                    [f["likelihood_ratio"] for f in fits], [0, 0.5, 1])],
                "boundary_hits": {key: sum(d["best_boundaries"][key] in ("lower", "upper")
                                          for d in diagnostics)
                                  for key in diagnostics[0]["best_boundaries"]},
                "unidentified_width_fits": sum(not d["sigma_identified_at_best"] for d in diagnostics),
            }
        summaries.append(summary)
    return summaries


def _plot(records, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for col, strength in enumerate((0.0, 1.0)):
        rows = [r for r in records if r["true_lambda_strength"] == strength]
        coarse = np.array([r["coarse"]["likelihood_ratio"] for r in rows])
        refined = np.array([r["refined"]["likelihood_ratio"] for r in rows])
        ax = axes[0, col]
        low, high = min(coarse.min(), refined.min()), max(coarse.max(), refined.max())
        if low == high:
            high = low + 1
        ax.plot([low, high], [low, high], "--", color="gray", label="equal LR")
        ax.scatter(coarse, refined, color="#2166ac")
        ax.set(xlabel="Current grid LR", ylabel="Refined grid LR",
               title=f"lambda = {strength:g}: paired likelihood ratios")
        ax.legend()
        ax = axes[1, col]
        for i, row in enumerate(rows):
            ax.plot([row["coarse"]["best_sigma_t"], row["refined"]["best_sigma_t"]],
                    [row["coarse"]["best_lambda_strength"], row["refined"]["best_lambda_strength"]],
                    color="#a0a0a0", linewidth=1)
        for label, marker, color in (("coarse", "o", "#2166ac"), ("refined", "+", "#b33a3a")):
            ax.scatter([r[label]["best_sigma_t"] for r in rows],
                       [r[label]["best_lambda_strength"] for r in rows],
                       marker=marker, color=color, label=label)
        ax.set(xlabel="Fitted sigma_T", ylabel="Fitted lambda",
               title="Null width is unidentifiable" if strength == 0 else "Paired parameter estimates")
        ax.legend()
    fig.suptitle("Same counts, nested fit grids — diagnostic comparison, no recalibrated test")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=10, help="Trials per null/signal scenario")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--calibration-shots", type=int, default=50_000, help="Shots per control (five controls)")
    parser.add_argument("--test-shots", type=int, default=100_000, help="Total over the two test settings")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json-output", type=Path, default=Path("spatial_detector_grid_comparison.json"))
    parser.add_argument("--plot-output", type=Path, default=Path("spatial_detector_grid_comparison.png"))
    args = parser.parse_args()
    if (args.repetitions < 1 or args.grid_size < 24 or args.workers < 1
            or args.calibration_shots < 1 or args.test_shots < 2 or args.test_shots % 2):
        parser.error("Use positive repetitions/workers/shots, grid-size >=24 and even test-shots")
    first = double_slit_effect_experiment(nx=args.grid_size, ny=args.grid_size)
    second = replace(first, detector_x=0, detector_width=0.5, reference_time=13 / 30)
    controls = _calibration_experiments(first)
    coarse_sigmas, coarse_lambdas = FIT_SIGMAS, FIT_LAMBDAS
    refined_sigmas, refined_lambdas = _midpoints(FIT_SIGMAS), _midpoints(FIT_LAMBDAS)
    settings = dict(vars(args), json_output=str(args.json_output), plot_output=str(args.plot_output))
    code_paths = (Path(__file__), Path(inference.__file__))
    metadata = {
        "status": "paired_grid_diagnostic_not_false_positive_calibration",
        "settings": settings,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "true_response": asdict(TRUE_RESPONSE),
        "calibration_experiments": [asdict(e) for e in controls],
        "test_experiments": [asdict(first), asdict(second)],
        "grids": {"coarse_sigmas": coarse_sigmas, "coarse_lambdas": coarse_lambdas,
                  "refined_sigmas": refined_sigmas, "refined_lambdas": refined_lambdas,
                  "timing_jitters": JITTER_GRID},
        "interpretation": "Only sigma/lambda grid density changes; truth and geometry stay fixed. "
            "Each pair uses identical saved count arrays. Detection is disabled, not calibrated. "
            "The fitted null width is not a physical estimate. No threshold is selected from this run.",
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.json_output.with_suffix(".inputs.json")
    log = args.json_output.with_suffix(".jsonl")
    # Generate each record once, before fitting either grid. Saving counts also
    # permits exact replay when random-number implementations change.
    records = []
    for index, sequence in enumerate(np.random.SeedSequence(args.seed).spawn(2 * args.repetitions)):
        strength = 0.0 if index < args.repetitions else 1.0
        seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        records.append({
            "replicate": index % args.repetitions, "seed": seed,
            "true_sigma_t": 0.2, "true_lambda_strength": strength,
            "calibration_counts": inference.simulate_realistic_counts(
                controls, 0.2, 0, TRUE_RESPONSE, args.calibration_shots, rng=rng),
            "test_counts": inference.simulate_realistic_counts(
                (first, second), 0.2, strength, TRUE_RESPONSE,
                [args.test_shots // 2] * 2, rng=rng),
        })
    manifest.write_text(json.dumps(metadata | {"records": records}, default=_json_value, indent=2) + "\n")
    libraries = []
    for label, sigmas, lambdas in (("coarse", coarse_sigmas, coarse_lambdas),
                                   ("refined", refined_sigmas, refined_lambdas)):
        print(f"Building {label} library: {len(sigmas)} x {len(lambdas)} x {len(JITTER_GRID)}", flush=True)
        libraries.append(inference.build_spatial_detector_inference_library(
            controls, (first, second), calibration_sigma_t=0.2, calibration_lambda_strength=0,
            candidate_sigmas=sigmas, candidate_lambdas=lambdas, candidate_timing_jitters=JITTER_GRID))
    completed = []
    def consume(results):
        with log.open("w") as stream:
            for record in results:
                completed.append(record)
                stream.write(json.dumps(record, default=_json_value) + "\n")
                stream.flush()
                print(f"Completed {len(completed)}/{len(records)}: lambda={record['true_lambda_strength']:g}, "
                      f"delta LR={record['difference']['delta_likelihood_ratio']:.6g}", flush=True)
    if args.workers == 1:
        _initialize(*libraries)
        consume(map(_compare, records))
    else:
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize,
                                 initargs=tuple(libraries), mp_context=context) as pool:
            consume(pool.map(_compare, records, chunksize=1))
    summaries = _summaries(completed)
    args.json_output.write_text(json.dumps(metadata | {"summaries": summaries, "records": completed},
                                          default=_json_value, indent=2) + "\n")
    _plot(completed, args.plot_output)
    print(json.dumps(summaries, indent=2))
    print(f"Results: {args.json_output}; plot: {args.plot_output}")


if __name__ == "__main__":
    main()
