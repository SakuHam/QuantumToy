from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]

MAIN_PATH = ROOT_DIR / "main.py"
if not MAIN_PATH.exists():
    raise RuntimeError(
        f"main.py not found at expected location:\n"
        f"  {MAIN_PATH}\n"
        f"ROOT_DIR was resolved to:\n"
        f"  {ROOT_DIR}\n"
        f"Check project layout."
    )

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from file.run_io import load_run_bundle  # noqa: E402


BASELINE_SUMMARY = ROOT_DIR / "sweep_runs" / "posthoc_trf_seed_sweep" / "summary.jsonl"
BASELINE_OUT_ROOT = ROOT_DIR / "sweep_runs" / "posthoc_trf_seed_sweep"
OUT_ROOT = ROOT_DIR / "sweep_runs" / "posthoc_trf_velocity_sweep"

DEFAULT_MAX_WORKERS = 3
DEFAULT_TOP_K = 5
DEFAULT_K0X_VALUES = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

COMMON_ENV: dict[str, str] = {
    "BATCH_FAST_MODE": "False",
    "BREAK_ON_DETECTOR_CLICK": "True",
    "ENABLE_BOHMIAN_OVERLAY": "False",
    "SAVE_MP4": "False",
    "SAVE_COMPLEX_STATE_FRAMES": "True",
    "THEORY_NAME": "schrodinger",
    "DETECTOR_NAME": "emergent",
}

EXPERIMENT_ENV: dict[str, str] = {
    # Optional fixed overrides for all velocity runs.
}


@dataclass(frozen=True)
class VelocityCase:
    case_name: str
    base_case_name: str
    base_seed: int

    k0x_value: float
    sigma_mode: str
    sigma_fixed_value: float | None

    base_click_side: str | None
    base_trf_side: str | None
    base_dominance: float | None
    base_ratio: float | None
    base_ref_time: float | None
    base_k0x: float | None
    base_sigma_init: float | None

    overrides: dict[str, str]


def safe_case_name(s: str) -> str:
    return s.replace(".", "p").replace("-", "m")


def append_jsonl(path: Path, data: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_summary_latest_per_case(path: Path) -> list[dict]:
    latest_by_case: dict[str, dict] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            case_name = row.get("case_name")
            if case_name is None:
                continue
            latest_by_case[case_name] = row

    return list(latest_by_case.values())


def is_successful_summary(data: dict[str, object] | None) -> bool:
    if not data:
        return False
    return int(data.get("returncode", 999999)) == 0


def build_skip_result(case: VelocityCase, existing_summary: dict[str, object]) -> dict[str, object]:
    result = dict(existing_summary)
    result["case_name"] = case.case_name
    result["skipped"] = True
    result["skip_reason"] = "existing_successful_summary"
    return result


def remove_stale_outputs(case_prefix: Path) -> list[str]:
    stale_files = [
        case_prefix.with_suffix(".npz"),
        case_prefix.with_suffix(".json"),
        case_prefix.with_name(case_prefix.name + "_meta.json"),
        case_prefix.with_name(case_prefix.name + "_flux_summary.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.json"),
        case_prefix.with_name(case_prefix.name + "_pseudo_clicks.jsonl"),
        case_prefix.with_name(case_prefix.name + "_summary.json"),
        case_prefix.with_name(case_prefix.name + "_detector.json"),
        case_prefix.with_name(case_prefix.name + ".mp4"),
        case_prefix.with_name(case_prefix.name + ".log"),
    ]

    removed: list[str] = []
    for path in stale_files:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def case_dir_for(case: VelocityCase) -> Path:
    return OUT_ROOT / case.base_case_name / case.case_name


def case_prefix_for(case: VelocityCase) -> Path:
    return case_dir_for(case) / case.case_name


def case_summary_path(case: VelocityCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}_summary.json"


def case_log_path(case: VelocityCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}.log"


def _safe_get(d: dict | None, key: str, default=None):
    if d is None:
        return default
    return d.get(key, default)


def estimate_sigma_init_from_cfg(cfg_dict: dict) -> float:
    theory_name = str(cfg_dict.get("THEORY_NAME", "")).lower()

    k0x = float(cfg_dict.get("k0x", 5.0))
    m_mass = float(cfg_dict.get("m_mass", 1.0))
    hbar = float(cfg_dict.get("hbar", 1.0))
    c_light = float(cfg_dict.get("c_light", 1.0))
    screen_center_x = float(cfg_dict.get("screen_center_x", 10.0))
    barrier_center_x = float(cfg_dict.get("barrier_center_x", 0.0))

    is_dirac_like = "dirac" in theory_name

    if is_dirac_like:
        p0 = hbar * k0x
        mc2 = m_mass * c_light**2
        E0 = float(np.sqrt((c_light * p0) ** 2 + mc2**2))
        v_est = float((c_light**2 * p0) / (E0 + 1e-30))
    else:
        v_est = float(k0x / (m_mass + 1e-30))

    L_gap = float(screen_center_x - barrier_center_x)
    t_gap = L_gap / (abs(v_est) + 1e-12)
    sigma_init = 0.60 * t_gap
    return float(sigma_init)


def extract_posthoc_summary_from_bundle(npz_path: Path) -> dict[str, object]:
    bundle = load_run_bundle(npz_path)

    x_click = bundle.get("x_click")
    y_click = bundle.get("y_click")
    t_det = bundle.get("t_det")
    idx_det = bundle.get("idx_det")
    detector_clicked = bundle.get("detector_clicked")
    cfg_dict = bundle["meta"]["config"]

    if y_click is None:
        click_side = None
    else:
        click_side = "upper" if float(y_click) > 0.0 else "lower"

    trf = bundle.get("posthoc_trf_info")
    wl = bundle.get("posthoc_worldline_info")

    trf_valid = _safe_get(trf, "valid", None)
    trf_chosen_side = _safe_get(trf, "chosen_side", None)

    wl_used = _safe_get(wl, "used", False)
    wl_seed_side = _safe_get(wl, "seed_side", None)
    wl_seed_x = _safe_get(wl, "seed_x", None)
    wl_seed_y = _safe_get(wl, "seed_y", None)

    posthoc_matches_click = (
        bool(trf_valid)
        and (click_side is not None)
        and (trf_chosen_side is not None)
        and (str(trf_chosen_side) == str(click_side))
    )

    worldline_seed_matches_click = (
        bool(wl_used)
        and (click_side is not None)
        and (wl_seed_side is not None)
        and (str(wl_seed_side) == str(click_side))
    )

    posthoc_base_rho = bundle.get("posthoc_base_rho")
    posthoc_selected_rho = bundle.get("posthoc_selected_rho")

    result = {
        "cfg_k0x": float(cfg_dict.get("k0x", np.nan)),
        "cfg_sigma_init_estimate": estimate_sigma_init_from_cfg(cfg_dict),

        "clicked": bool(detector_clicked) if detector_clicked is not None else None,
        "click_time": None if t_det is None else float(t_det),
        "click_idx": None if idx_det is None else int(idx_det),
        "click_x": None if x_click is None else float(x_click),
        "click_y": None if y_click is None else float(y_click),
        "click_side": click_side,

        "posthoc_trf_valid": None if trf_valid is None else bool(trf_valid),
        "posthoc_trf_ref_idx": _safe_get(trf, "ref_idx", None),
        "posthoc_trf_ref_time": _safe_get(trf, "ref_time", None),
        "posthoc_trf_chosen_side": trf_chosen_side,
        "posthoc_trf_upper_evidence": _safe_get(trf, "upper_evidence", None),
        "posthoc_trf_lower_evidence": _safe_get(trf, "lower_evidence", None),
        "posthoc_trf_total_evidence": _safe_get(trf, "total_evidence", None),
        "posthoc_trf_abs_margin": _safe_get(trf, "abs_margin", None),
        "posthoc_trf_rel_margin": _safe_get(trf, "rel_margin", None),
        "posthoc_trf_ratio": _safe_get(trf, "ratio", None),
        "posthoc_trf_dominance": _safe_get(trf, "dominance", None),
        "posthoc_trf_adaptive_score": _safe_get(trf, "adaptive_score", None),

        "posthoc_worldline_used": bool(wl_used),
        "posthoc_worldline_seed_side": wl_seed_side,
        "posthoc_worldline_seed_x": wl_seed_x,
        "posthoc_worldline_seed_y": wl_seed_y,

        "posthoc_matches_click": bool(posthoc_matches_click),
        "worldline_seed_matches_click": bool(worldline_seed_matches_click),

        "has_posthoc_base_rho": bool(posthoc_base_rho is not None),
        "has_posthoc_selected_rho": bool(posthoc_selected_rho is not None),
    }

    return result


def infer_seed_from_row(row: dict) -> int:
    overrides = row.get("overrides", {})
    if isinstance(overrides, dict):
        seed_raw = overrides.get("DETECTOR_NOISE_SEED", None)
        if seed_raw is not None:
            return int(seed_raw)

    case_name = str(row["case_name"])
    return int(case_name.split("_")[-1].replace("p", ".").replace("m", "-"))


def build_cases(
    baseline_summary_path: Path,
    top_k: int,
    k0x_values: list[float],
    sigma_mode: str,
) -> list[VelocityCase]:
    rows = load_summary_latest_per_case(baseline_summary_path)

    rows_ok = [
        r for r in rows
        if int(r.get("returncode", 999999)) == 0
        and r.get("posthoc_trf_valid") is True
        and r.get("posthoc_trf_dominance") is not None
    ]

    if not rows_ok:
        raise RuntimeError("No successful valid rows found in baseline summary")

    rows_sorted = sorted(
        rows_ok,
        key=lambda r: float(r.get("posthoc_trf_dominance", 999.0)),
    )

    chosen_rows = rows_sorted[:top_k]
    cases: list[VelocityCase] = []

    for row in chosen_rows:
        base_case_name = str(row["case_name"])
        base_seed = infer_seed_from_row(row)

        base_npz = BASELINE_OUT_ROOT / base_case_name / f"{base_case_name}.npz"
        if not base_npz.exists():
            raise FileNotFoundError(f"Baseline NPZ not found for case {base_case_name}: {base_npz}")

        base_bundle = load_run_bundle(base_npz)
        base_cfg_dict = base_bundle["meta"]["config"]
        base_k0x = float(base_cfg_dict.get("k0x", 5.0))
        base_sigma_init = estimate_sigma_init_from_cfg(base_cfg_dict)

        for k0x_value in k0x_values:
            k0x_tag = safe_case_name(f"{k0x_value:.2f}")
            case_name = f"{base_case_name}__k0x_{k0x_tag}"

            case_overrides = {
                "DETECTOR_NOISE_SEED": str(base_seed),
                "CLICK_RNG_SEED": str(base_seed),
                "k0x": str(float(k0x_value)),
            }

            sigma_fixed_value = None
            if sigma_mode == "fixed_baseline":
                sigma_fixed_value = float(base_sigma_init)
                case_overrides["POSTHOC_TRF_SIGMAT"] = str(sigma_fixed_value)

            case_overrides.update(EXPERIMENT_ENV)

            cases.append(
                VelocityCase(
                    case_name=case_name,
                    base_case_name=base_case_name,
                    base_seed=base_seed,

                    k0x_value=float(k0x_value),
                    sigma_mode=sigma_mode,
                    sigma_fixed_value=sigma_fixed_value,

                    base_click_side=row.get("click_side"),
                    base_trf_side=row.get("posthoc_trf_chosen_side"),
                    base_dominance=None if row.get("posthoc_trf_dominance") is None else float(row["posthoc_trf_dominance"]),
                    base_ratio=None if row.get("posthoc_trf_ratio") is None else float(row["posthoc_trf_ratio"]),
                    base_ref_time=None if row.get("posthoc_trf_ref_time") is None else float(row["posthoc_trf_ref_time"]),
                    base_k0x=base_k0x,
                    base_sigma_init=base_sigma_init,

                    overrides=case_overrides,
                )
            )

    return cases


def run_case(case: VelocityCase, python_bin: str, force_rerun: bool) -> dict[str, object]:
    start = time.time()

    case_dir = case_dir_for(case)
    case_dir.mkdir(parents=True, exist_ok=True)

    case_prefix = case_prefix_for(case)
    log_path = case_log_path(case)
    summary_path = case_summary_path(case)

    existing_summary = read_json(summary_path)
    if (not force_rerun) and is_successful_summary(existing_summary):
        return build_skip_result(case, existing_summary)

    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(case.overrides)
    env["OUTPUT_PREFIX"] = str(case_prefix)

    removed = remove_stale_outputs(case_prefix)

    time.sleep(float(np.random.uniform(0.0, 2.0)))

    cmd = [python_bin, "main.py"]

    header_lines = [
        "=" * 80,
        f"RUN CASE: {case.case_name}",
        "=" * 80,
        f"[BASE CASE] {case.base_case_name}",
        f"[K0X VALUE] {case.k0x_value}",
        f"[SIGMA MODE] {case.sigma_mode}",
        f"[SIGMA FIXED VALUE] {case.sigma_fixed_value}",
        "[ENV OVERRIDES]",
    ]
    all_env_to_print = {**COMMON_ENV, **case.overrides, "OUTPUT_PREFIX": env["OUTPUT_PREFIX"]}
    for k in sorted(all_env_to_print):
        header_lines.append(f"  {k}={all_env_to_print[k]}")
    if removed:
        header_lines.append("[REMOVED STALE]")
        for p in removed:
            header_lines.append(f"  {p}")
    header_lines.append("[CMD]")
    header_lines.append(f"  {' '.join(cmd)}")
    header_lines.append("")

    header = "\n".join(header_lines) + "\n"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.flush()

        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = time.time() - start

    result = {
        "case_name": case.case_name,
        "base_case_name": case.base_case_name,
        "base_seed": int(case.base_seed),

        "k0x_value": float(case.k0x_value),
        "sigma_mode": str(case.sigma_mode),
        "sigma_fixed_value": case.sigma_fixed_value,

        "base_click_side": case.base_click_side,
        "base_trf_side": case.base_trf_side,
        "base_dominance": case.base_dominance,
        "base_ratio": case.base_ratio,
        "base_ref_time": case.base_ref_time,
        "base_k0x": case.base_k0x,
        "base_sigma_init": case.base_sigma_init,

        "returncode": int(proc.returncode),
        "elapsed_sec": elapsed,
        "log_path": str(log_path),
        "output_prefix": str(case_prefix),
        "overrides": case.overrides,
        "skipped": False,
        "likely_oom": int(proc.returncode) == -9,
    }

    npz_path = case_prefix.with_suffix(".npz")
    if proc.returncode == 0 and npz_path.exists():
        try:
            result.update(extract_posthoc_summary_from_bundle(npz_path))

            trf_side = result.get("posthoc_trf_chosen_side")
            wl_side = result.get("posthoc_worldline_seed_side")

            result["flip_vs_base"] = (
                (case.base_trf_side is not None)
                and (trf_side is not None)
                and (str(trf_side) != str(case.base_trf_side))
            )
            result["worldline_flip_vs_base"] = (
                (case.base_trf_side is not None)
                and (wl_side is not None)
                and (str(wl_side) != str(case.base_trf_side))
            )
        except Exception as e:
            result["bundle_parse_error"] = repr(e)
            result["flip_vs_base"] = None
            result["worldline_flip_vs_base"] = None
    else:
        result.update(
            {
                "cfg_k0x": None,
                "cfg_sigma_init_estimate": None,

                "clicked": False,
                "click_time": None,
                "click_idx": None,
                "click_x": None,
                "click_y": None,
                "click_side": None,

                "posthoc_trf_valid": None,
                "posthoc_trf_ref_idx": None,
                "posthoc_trf_ref_time": None,
                "posthoc_trf_chosen_side": None,
                "posthoc_trf_upper_evidence": None,
                "posthoc_trf_lower_evidence": None,
                "posthoc_trf_total_evidence": None,
                "posthoc_trf_abs_margin": None,
                "posthoc_trf_rel_margin": None,
                "posthoc_trf_ratio": None,
                "posthoc_trf_dominance": None,
                "posthoc_trf_adaptive_score": None,

                "posthoc_worldline_used": False,
                "posthoc_worldline_seed_side": None,
                "posthoc_worldline_seed_x": None,
                "posthoc_worldline_seed_y": None,

                "posthoc_matches_click": False,
                "worldline_seed_matches_click": False,

                "has_posthoc_base_rho": False,
                "has_posthoc_selected_rho": False,

                "flip_vs_base": None,
                "worldline_flip_vs_base": None,
            }
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def rebuild_summary_jsonl_from_case_summaries(cases: list[VelocityCase], summary_jsonl: Path) -> int:
    count = 0
    with open(summary_jsonl, "w", encoding="utf-8") as out:
        for case in cases:
            data = read_json(case_summary_path(case))
            if data is None:
                continue
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_float_list(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if not vals:
        raise ValueError("No float values given")
    return vals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=str, default=str(BASELINE_SUMMARY))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument(
        "--k0x-values",
        type=str,
        default=",".join(f"{x:.2f}" for x in DEFAULT_K0X_VALUES),
        help="Comma-separated k0x values, e.g. 3,4,5,6,7,8",
    )
    parser.add_argument(
        "--sigma-mode",
        choices=("auto", "fixed_baseline"),
        default="auto",
        help=(
            "auto: let main.py recompute sigmaT from current k0x. "
            "fixed_baseline: keep POSTHOC_TRF_SIGMAT fixed to baseline sigma_init."
        ),
    )
    parser.add_argument(
        "--base-case-prefix",
        default=None,
        help="Only include baseline cases whose case_name starts with this prefix",
    )
    args = parser.parse_args()

    python_bin = sys.executable or "python3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = Path(args.baseline_summary)
    if not baseline_summary_path.exists():
        raise FileNotFoundError(f"Baseline summary not found: {baseline_summary_path}")

    k0x_values = parse_float_list(args.k0x_values)

    cases = build_cases(
        baseline_summary_path=baseline_summary_path,
        top_k=int(args.top_k),
        k0x_values=k0x_values,
        sigma_mode=str(args.sigma_mode),
    )

    if args.base_case_prefix:
        cases = [c for c in cases if c.base_case_name.startswith(args.base_case_prefix)]

    if not cases:
        print("No cases to run.")
        return 0

    summary_jsonl = OUT_ROOT / "summary.jsonl"
    rebuilt_count = rebuild_summary_jsonl_from_case_summaries(cases, summary_jsonl)

    unique_bases = sorted({c.base_case_name for c in cases})

    print("=" * 80)
    print("Posthoc TRF velocity sweep")
    print("=" * 80)
    print(f"ROOT_DIR         : {ROOT_DIR}")
    print(f"MAIN_PATH        : {MAIN_PATH}")
    print(f"BASELINE_SUMMARY : {baseline_summary_path}")
    print(f"OUT_ROOT         : {OUT_ROOT}")
    print(f"MAX_WORKERS      : {args.max_workers}")
    print(f"TOP_K            : {args.top_k}")
    print(f"N_BASE_CASES     : {len(unique_bases)}")
    print(f"N_CASES          : {len(cases)}")
    print(f"K0X_VALUES       : {k0x_values}")
    print(f"SIGMA_MODE       : {args.sigma_mode}")
    print(f"FORCE_RERUN      : {args.force_rerun}")
    print(f"BASE_CASE_PREFIX : {args.base_case_prefix}")
    print(f"REBUILT_JSONL    : {rebuilt_count}")
    print("")

    completed = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_case = {
            ex.submit(run_case, case, python_bin, args.force_rerun): case
            for case in cases
        }

        for fut in as_completed(future_to_case):
            case = future_to_case[fut]

            try:
                result = fut.result()
            except Exception as e:
                failed += 1
                err = {
                    "case_name": case.case_name,
                    "base_case_name": case.base_case_name,
                    "base_seed": int(case.base_seed),
                    "k0x_value": float(case.k0x_value),
                    "sigma_mode": str(case.sigma_mode),
                    "sigma_fixed_value": case.sigma_fixed_value,
                    "returncode": None,
                    "elapsed_sec": None,
                    "clicked": False,
                    "click_side": None,
                    "posthoc_trf_valid": None,
                    "posthoc_trf_chosen_side": None,
                    "posthoc_matches_click": False,
                    "worldline_seed_matches_click": False,
                    "flip_vs_base": None,
                    "worldline_flip_vs_base": None,
                    "error": repr(e),
                    "overrides": case.overrides,
                    "skipped": False,
                    "likely_oom": False,
                }
                append_jsonl(summary_jsonl, err)
                print(f"[FAIL ] {case.case_name}: {e}")
                continue

            append_jsonl(summary_jsonl, result)

            if result.get("skipped", False):
                skipped += 1
                print(
                    f"[SKIP ] {case.case_name} | "
                    f"k0x={result.get('k0x_value')} | "
                    f"trf={result.get('posthoc_trf_chosen_side')} | "
                    f"dom={result.get('posthoc_trf_dominance')}"
                )
                continue

            completed += 1

            if result["returncode"] != 0:
                failed += 1
                status = "FAIL"
            else:
                status = "DONE"

            oom_part = " | likely_oom=True" if result.get("likely_oom") else ""

            print(
                f"[{status:4}] {case.case_name} | "
                f"k0x={result.get('k0x_value'):.2f} | "
                f"cfg_k0x={result.get('cfg_k0x')} | "
                f"rc={result['returncode']} | "
                f"{result['elapsed_sec']:.2f}s | "
                f"click={result.get('click_side')} | "
                f"trf={result.get('posthoc_trf_chosen_side')} | "
                f"dom={result.get('posthoc_trf_dominance')} | "
                f"flip={result.get('flip_vs_base')} | "
                f"match={result.get('posthoc_matches_click')}"
                f"{oom_part}"
            )

    print("")
    print("=" * 80)
    print("Velocity sweep complete")
    print("=" * 80)
    print(f"Completed new runs : {completed}")
    print(f"Skipped existing   : {skipped}")
    print(f"Failed             : {failed}")
    print(f"Summary            : {summary_jsonl}")
    print(f"Outputs            : {OUT_ROOT}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())