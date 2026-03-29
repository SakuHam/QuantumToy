from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
WORKER_PATH = ROOT_DIR / "worldline_hybrid_single.py"

if not WORKER_PATH.exists():
    raise RuntimeError(f"Worker script not found: {WORKER_PATH}")

OUT_ROOT = ROOT_DIR / "sweep_runs" / "worldline_seed_sweep"
DEFAULT_MAX_WORKERS = 10
POLL_INTERVAL_SEC = 0.20


@dataclass(frozen=True)
class SweepCase:
    case_name: str
    seed: int


@dataclass
class RunningProc:
    case: SweepCase
    proc: subprocess.Popen
    log_file_handle: object
    start_time: float
    log_path: Path
    summary_path: Path


def build_cases(seed_start: int, seed_end: int) -> list[SweepCase]:
    return [SweepCase(case_name=f"seed_{seed}", seed=seed) for seed in range(seed_start, seed_end + 1)]


def case_dir_for(case: SweepCase) -> Path:
    return OUT_ROOT / case.case_name


def case_summary_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}_summary.json"


def case_log_path(case: SweepCase) -> Path:
    return case_dir_for(case) / f"{case.case_name}.log"


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_successful_summary(data: dict | None) -> bool:
    if not data:
        return False
    try:
        return int(data.get("returncode", 999999)) == 0
    except Exception:
        return False


def append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def rebuild_summary_jsonl_from_case_summaries(cases: list[SweepCase], summary_jsonl: Path) -> int:
    count = 0
    with open(summary_jsonl, "w", encoding="utf-8") as out:
        for case in cases:
            data = read_json(case_summary_path(case))
            if data is None:
                continue
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "n/a"

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def progress_line(total: int, started: int, finished: int, running: int, t0: float) -> str:
    elapsed = max(0.0, time.time() - t0)
    pct = 100.0 * finished / total if total > 0 else 0.0

    eta = None
    if finished > 0 and finished < total:
        avg_per_finished = elapsed / finished
        remaining = total - finished
        eta = avg_per_finished * remaining

    return (
        f"[PROGRESS] "
        f"{finished}/{total} done ({pct:.1f}%) | "
        f"{running} running | "
        f"elapsed={format_eta(elapsed)} | "
        f"eta≈{format_eta(eta)}"
    )


def to_bool_or_none(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    try:
        return bool(x)
    except Exception:
        return None


def get_pre_slit_side(row: dict):
    return row.get("pre_slit_chosen_side")


def get_slit_side(row: dict):
    return row.get("slit_pass_chosen_side")


def get_click_side(row: dict):
    return row.get("click_side")


def get_forward_side(row: dict):
    return row.get("forward_guess_chosen_side")


def get_trf_side(row: dict):
    return row.get("posthoc_trf_chosen_side", row.get("posthoc_chosen_side"))


def is_interesting_case(row: dict) -> tuple[bool, str | None]:
    """
    Early-bird osuma ensisijaisesti:
      1) interesting_clean_path_mismatch_case == True
         eli pre_slit == slit_pass != click

    Varalla myös eksplisiittinen rekonstruktio samoista kentistä.
    """
    clean = to_bool_or_none(row.get("interesting_clean_path_mismatch_case"))
    if clean is True:
        pre = get_pre_slit_side(row)
        slit = get_slit_side(row)
        click = get_click_side(row)
        return True, f"clean_path_mismatch:{pre}->{slit}->{click}"

    pre = get_pre_slit_side(row)
    slit = get_slit_side(row)
    click = get_click_side(row)

    if (
        pre in {"upper", "lower"}
        and slit in {"upper", "lower"}
        and click in {"upper", "lower"}
        and pre == slit
        and slit != click
    ):
        return True, f"clean_path_mismatch:{pre}->{slit}->{click}"

    return False, None


def make_skip_result(case: SweepCase, existing_summary: dict) -> dict:
    result = dict(existing_summary)
    result["skipped"] = True
    result["skip_reason"] = "existing_successful_summary"
    return result


def make_error_result(
    case: SweepCase,
    returncode: int | None,
    elapsed_sec: float | None,
    error: str,
    skipped: bool = False,
) -> dict:
    return {
        "case_name": case.case_name,
        "seed": case.seed,
        "returncode": returncode,
        "elapsed_sec": elapsed_sec,
        "clicked": False,
        "error": error,
        "skipped": skipped,
    }


def start_case(case: SweepCase, python_bin: str, force_rerun: bool, save_debug: bool) -> tuple[str, RunningProc | dict]:
    """
    Returns:
      ("skip", result_dict) or
      ("run", RunningProc)
    """
    case_dir = case_dir_for(case)
    case_dir.mkdir(parents=True, exist_ok=True)

    summary_path = case_summary_path(case)
    log_path = case_log_path(case)

    existing_summary = read_json(summary_path)
    if (not force_rerun) and is_successful_summary(existing_summary):
        return "skip", make_skip_result(case, existing_summary)

    cmd = [
        python_bin,
        str(WORKER_PATH),
        "--seed", str(case.seed),
        "--out-dir", str(case_dir),
        "--case-name", case.case_name,
    ]
    if save_debug:
        cmd.append("--save")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"CASE {case.case_name}\n")
        f.write(f"SEED {case.seed}\n")
        f.write("[CMD]\n")
        f.write("  " + " ".join(cmd) + "\n\n")

    log_fh = open(log_path, "a", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT_DIR,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        log_fh.close()
        return "skip", make_error_result(
            case=case,
            returncode=None,
            elapsed_sec=0.0,
            error=f"launcher_popen_failed: {repr(e)}",
            skipped=False,
        )

    return "run", RunningProc(
        case=case,
        proc=proc,
        log_file_handle=log_fh,
        start_time=time.time(),
        log_path=log_path,
        summary_path=summary_path,
    )


def finalize_running_proc(r: RunningProc) -> dict:
    elapsed = float(time.time() - r.start_time)
    returncode = r.proc.poll()

    try:
        r.log_file_handle.flush()
    except Exception:
        pass

    try:
        r.log_file_handle.close()
    except Exception:
        pass

    data = read_json(r.summary_path)
    if data is None:
        data = make_error_result(
            case=r.case,
            returncode=returncode,
            elapsed_sec=elapsed,
            error="summary_json_missing",
            skipped=False,
        )
    else:
        data["elapsed_sec_launcher"] = elapsed
        data["returncode"] = returncode

    return data


def terminate_running_proc(r: RunningProc, reason: str) -> dict:
    if r.proc.poll() is None:
        try:
            r.proc.terminate()
        except Exception:
            pass

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if r.proc.poll() is not None:
                break
            time.sleep(0.05)

        if r.proc.poll() is None:
            try:
                r.proc.kill()
            except Exception:
                pass

            deadline2 = time.time() + 2.0
            while time.time() < deadline2:
                if r.proc.poll() is not None:
                    break
                time.sleep(0.05)

    elapsed = float(time.time() - r.start_time)
    returncode = r.proc.poll()

    try:
        r.log_file_handle.flush()
    except Exception:
        pass

    try:
        r.log_file_handle.close()
    except Exception:
        pass

    data = read_json(r.summary_path)
    if data is None:
        data = make_error_result(
            case=r.case,
            returncode=returncode,
            elapsed_sec=elapsed,
            error=reason,
            skipped=False,
        )
    else:
        data["elapsed_sec_launcher"] = elapsed
        data["returncode"] = returncode
        data["terminated_early"] = True
        data["termination_reason"] = reason

    return data


def compact_result_line(result: dict) -> str:
    seed = result.get("seed")
    rc = result.get("returncode")
    pre = result.get("pre_slit_chosen_side")
    slit = result.get("slit_pass_chosen_side")
    fwd = result.get("forward_guess_chosen_side")
    trf = result.get("posthoc_trf_chosen_side", result.get("posthoc_chosen_side"))
    click = result.get("click_side")
    elapsed = result.get("elapsed_sec")

    parts = [f"seed={seed}"]

    if rc is not None:
        parts.append(f"rc={rc}")
    if pre is not None:
        parts.append(f"pre={pre}")
    if slit is not None:
        parts.append(f"slit={slit}")
    if fwd is not None:
        parts.append(f"fwd={fwd}")
    if trf is not None:
        parts.append(f"trf={trf}")
    if click is not None:
        parts.append(f"click={click}")
    if isinstance(elapsed, (float, int)):
        parts.append(f"{float(elapsed):.2f}s")

    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--case-prefix", default=None)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Pass --save to worker so debug npz files are stored.",
    )
    parser.add_argument(
        "--early-bird",
        action="store_true",
        help="Stop sweep at first clean path mismatch case.",
    )
    args = parser.parse_args()

    python_bin = sys.executable or "python3"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args.seed_start, args.seed_end)
    if args.case_prefix:
        cases = [c for c in cases if c.case_name.startswith(args.case_prefix)]

    if not cases:
        print("No cases to run.")
        return 0

    summary_jsonl = OUT_ROOT / "summary.jsonl"
    early_bird_hit_path = OUT_ROOT / "early_bird_hit.json"

    rebuilt_count = rebuild_summary_jsonl_from_case_summaries(cases, summary_jsonl)

    print("=" * 80)
    print("Worldline seed sweep")
    print("=" * 80)
    print(
        f"cases={len(cases)} | "
        f"max_workers={args.max_workers} | "
        f"force_rerun={args.force_rerun} | "
        f"save={args.save} | "
        f"early_bird={args.early_bird}"
    )
    print(f"summary_jsonl={summary_jsonl}")
    print(f"rebuilt_rows={rebuilt_count}")
    print("")

    total = len(cases)
    t0 = time.time()

    completed = 0
    failed = 0
    skipped = 0
    started = 0

    pending_cases = list(cases)
    running: list[RunningProc] = []

    early_bird_hit: dict | None = None
    stop_launching_new = False

    while pending_cases or running:
        while (not stop_launching_new) and pending_cases and len(running) < args.max_workers:
            case = pending_cases.pop(0)
            kind, payload = start_case(
                case=case,
                python_bin=python_bin,
                force_rerun=args.force_rerun,
                save_debug=args.save,
            )

            if kind == "skip":
                result = payload
                append_jsonl(summary_jsonl, result)
                completed += 1

                if result.get("skipped", False):
                    skipped += 1
                    print(f"[SKIP] {result.get('case_name')} | seed={result.get('seed')}")
                else:
                    failed += 1
                    print(f"[FAIL] {result.get('case_name')} | seed={result.get('seed')} | {result.get('error')}")
                continue

            running.append(payload)
            started += 1

        if not running:
            break

        just_finished: list[RunningProc] = []
        still_running: list[RunningProc] = []

        for r in running:
            if r.proc.poll() is None:
                still_running.append(r)
            else:
                just_finished.append(r)

        running = still_running

        if not just_finished:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        for r in just_finished:
            result = finalize_running_proc(r)
            append_jsonl(summary_jsonl, result)
            completed += 1

            if int(result.get("returncode", 1)) != 0:
                failed += 1
                print(f"[FAIL] {result.get('case_name')} | {compact_result_line(result)}")
            else:
                print(f"[DONE] {result.get('case_name')} | {compact_result_line(result)}")

            hit, reason = is_interesting_case(result)
            if args.early_bird and hit and early_bird_hit is None:
                result["early_bird_reason"] = reason
                result["early_bird_time_sec"] = float(time.time() - t0)
                early_bird_hit = result

                with open(early_bird_hit_path, "w", encoding="utf-8") as f:
                    json.dump(early_bird_hit, f, indent=2, ensure_ascii=False)

                print("")
                print(f"[EARLY-BIRD] hit on {result.get('case_name')} | reason={reason}")
                print(progress_line(total, started, completed, len(running), t0))

                stop_launching_new = True
                pending_cases.clear()

                terminated_results: list[dict] = []
                for rr in running:
                    terminated = terminate_running_proc(rr, reason="terminated_by_early_bird")
                    terminated_results.append(terminated)

                running = []

                for terminated in terminated_results:
                    append_jsonl(summary_jsonl, terminated)
                    completed += 1
                    rc = terminated.get("returncode")
                    if rc not in (0, None):
                        failed += 1
                    print(f"[STOP] {terminated.get('case_name')} | seed={terminated.get('seed')}")

                break

        print(progress_line(total, started, completed, len(running), t0))

    print("")
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    print(f"completed={completed} | skipped={skipped} | failed={failed}")
    print(f"summary={summary_jsonl}")

    if early_bird_hit is not None:
        print(f"early_bird_hit={early_bird_hit_path}")
        print(
            f"interesting_case="
            f"{early_bird_hit.get('case_name')} | "
            f"seed={early_bird_hit.get('seed')} | "
            f"reason={early_bird_hit.get('early_bird_reason')}"
        )

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())