from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT_DIR / "sweep_runs" / "posthoc_trf_seed_sweep" / "summary.jsonl"


def load_summary_latest_per_case(path: Path) -> list[dict]:
    """
    Load JSONL summary and keep only the latest row for each case_name.
    Useful when summary.jsonl contains repeated entries due to reruns.
    """
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


def count_by_key(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        val = row.get(key)
        sval = str(val)
        out[sval] = out.get(sval, 0) + 1
    return out


def fmt_counts(d: dict[str, int], total: int) -> list[str]:
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = []
    for k, v in items:
        frac = 100.0 * v / total if total > 0 else 0.0
        lines.append(f"  {k}: {v} ({frac:.1f}%)")
    return lines


def finite_values(rows: list[dict], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        v = row.get(key)
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if vf == vf and vf not in (float("inf"), float("-inf")):
            vals.append(vf)
    return vals


def bool_true_count(rows: list[dict], key: str) -> int:
    return sum(bool(row.get(key, False)) for row in rows)


def safe_stat_block(vals: list[float], label: str) -> list[str]:
    if not vals:
        return [f"{label}: no data"]
    return [
        f"{label}:",
        f"  n      : {len(vals)}",
        f"  mean   : {mean(vals):.6f}",
        f"  median : {median(vals):.6f}",
        f"  min    : {min(vals):.6f}",
        f"  max    : {max(vals):.6f}",
    ]


def case_sort_key(row: dict):
    name = str(row.get("case_name", ""))
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        default=str(DEFAULT_SUMMARY),
        help="Path to summary.jsonl",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many hardest / mismatch cases to print",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows_all = load_summary_latest_per_case(summary_path)
    rows_all.sort(key=case_sort_key)

    total_rows = len(rows_all)
    skipped_rows = [r for r in rows_all if bool(r.get("skipped", False))]
    run_rows = [r for r in rows_all if not bool(r.get("skipped", False))]
    ok_rows = [r for r in run_rows if int(r.get("returncode", 999999)) == 0]
    fail_rows = [r for r in run_rows if int(r.get("returncode", 999999)) != 0]

    valid_rows = [r for r in ok_rows if r.get("posthoc_trf_valid") is True]
    invalid_rows = [r for r in ok_rows if r.get("posthoc_trf_valid") is not True]

    match_rows = [r for r in ok_rows if bool(r.get("posthoc_matches_click", False))]
    mismatch_rows = [r for r in ok_rows if not bool(r.get("posthoc_matches_click", False))]

    wl_match_rows = [r for r in ok_rows if bool(r.get("worldline_seed_matches_click", False))]
    wl_mismatch_rows = [r for r in ok_rows if not bool(r.get("worldline_seed_matches_click", False))]

    print("=" * 80)
    print("Posthoc / TRF sweep analysis")
    print("=" * 80)
    print(f"Summary file : {summary_path}")
    print(f"Rows total   : {total_rows}")
    print(f"Run rows     : {len(run_rows)}")
    print(f"Skipped      : {len(skipped_rows)}")
    print(f"Successful   : {len(ok_rows)}")
    print(f"Failed       : {len(fail_rows)}")
    print("")

    if ok_rows:
        print("Click side distribution")
        for line in fmt_counts(count_by_key(ok_rows, "click_side"), len(ok_rows)):
            print(line)
        print("")

        print("Posthoc TRF chosen side distribution")
        for line in fmt_counts(count_by_key(ok_rows, "posthoc_trf_chosen_side"), len(ok_rows)):
            print(line)
        print("")

        print("Worldline seed side distribution")
        for line in fmt_counts(count_by_key(ok_rows, "posthoc_worldline_seed_side"), len(ok_rows)):
            print(line)
        print("")

        print("Validity / agreement")
        print(f"  TRF valid                 : {len(valid_rows)} / {len(ok_rows)} ({100.0 * len(valid_rows) / len(ok_rows):.1f}%)")
        print(f"  TRF matches click         : {len(match_rows)} / {len(ok_rows)} ({100.0 * len(match_rows) / len(ok_rows):.1f}%)")
        print(f"  Worldline seed matches    : {len(wl_match_rows)} / {len(ok_rows)} ({100.0 * len(wl_match_rows) / len(ok_rows):.1f}%)")
        print("")

        for line in safe_stat_block(finite_values(valid_rows, "posthoc_trf_dominance"), "Dominance stats"):
            print(line)
        print("")

        for line in safe_stat_block(finite_values(valid_rows, "posthoc_trf_ratio"), "Ratio stats"):
            print(line)
        print("")

        for line in safe_stat_block(finite_values(valid_rows, "posthoc_trf_rel_margin"), "Relative margin stats"):
            print(line)
        print("")

        for line in safe_stat_block(finite_values(valid_rows, "posthoc_trf_adaptive_score"), "Adaptive score stats"):
            print(line)
        print("")

        for line in safe_stat_block(finite_values(valid_rows, "posthoc_trf_ref_time"), "Reference time stats"):
            print(line)
        print("")

        hardest_by_dominance = sorted(
            valid_rows,
            key=lambda r: float(r.get("posthoc_trf_dominance", 999.0)),
        )[: args.top_k]

        print(f"Hardest valid cases by lowest dominance (top {min(args.top_k, len(hardest_by_dominance))})")
        if hardest_by_dominance:
            for row in hardest_by_dominance:
                print(
                    f"  {row.get('case_name')}: "
                    f"click={row.get('click_side')}, "
                    f"trf={row.get('posthoc_trf_chosen_side')}, "
                    f"dominance={row.get('posthoc_trf_dominance')}, "
                    f"ratio={row.get('posthoc_trf_ratio')}, "
                    f"rel_margin={row.get('posthoc_trf_rel_margin')}, "
                    f"ref_time={row.get('posthoc_trf_ref_time')}"
                )
        else:
            print("  none")
        print("")

        print(f"TRF mismatch cases (top {args.top_k})")
        if mismatch_rows:
            for row in mismatch_rows[: args.top_k]:
                print(
                    f"  {row.get('case_name')}: "
                    f"click={row.get('click_side')}, "
                    f"trf={row.get('posthoc_trf_chosen_side')}, "
                    f"valid={row.get('posthoc_trf_valid')}, "
                    f"dominance={row.get('posthoc_trf_dominance')}, "
                    f"ratio={row.get('posthoc_trf_ratio')}, "
                    f"ref_time={row.get('posthoc_trf_ref_time')}"
                )
        else:
            print("  none")
        print("")

        print(f"Worldline mismatch cases (top {args.top_k})")
        if wl_mismatch_rows:
            for row in wl_mismatch_rows[: args.top_k]:
                print(
                    f"  {row.get('case_name')}: "
                    f"click={row.get('click_side')}, "
                    f"wl_used={row.get('posthoc_worldline_used')}, "
                    f"wl_seed={row.get('posthoc_worldline_seed_side')}, "
                    f"trf={row.get('posthoc_trf_chosen_side')}"
                )
        else:
            print("  none")
        print("")

    if fail_rows:
        print("Failed cases")
        for row in fail_rows[: args.top_k]:
            print(
                f"  {row.get('case_name')}: "
                f"rc={row.get('returncode')}, "
                f"likely_oom={row.get('likely_oom')}, "
                f"log={row.get('log_path')}"
            )
        print("")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())