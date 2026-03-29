from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary-jsonl",
        type=str,
        default="sweep_runs/worldline_seed_sweep/summary.jsonl",
        help="Path to summary.jsonl",
    )
    p.add_argument(
        "--show-seeds",
        action="store_true",
        help="Print seed lists for some categories",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many seeds to print in example lists",
    )
    p.add_argument(
        "--low-rel-margin-threshold",
        type=float,
        default=0.01,
        help="Threshold for low posthoc TRF relative margin",
    )
    p.add_argument(
        "--high-ratio-threshold",
        type=float,
        default=1.05,
        help="Threshold for high posthoc TRF ratio",
    )
    p.add_argument(
        "--high-dominance-threshold",
        type=float,
        default=0.60,
        help="Threshold for high posthoc TRF dominance",
    )
    p.add_argument(
        "--high-adaptive-score-threshold",
        type=float,
        default=1e-4,
        help="Threshold for high posthoc TRF adaptive score",
    )
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * n / d:.1f}%"


def print_counter(title: str, counter: Counter, total: int):
    print(title)
    if not counter:
        print("  (none)")
        return
    for k, v in counter.most_common():
        print(f"  {k}: {v} ({pct(v, total)})")


def maybe_print_seed_list(title: str, rows: list[dict], top_n: int):
    seeds = [r.get("seed") for r in rows if r.get("seed") is not None]
    seeds = seeds[:top_n]
    print(f"{title}: {seeds}")


def mean_of(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_of(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return float(vals[mid])
    return float(0.5 * (vals[mid - 1] + vals[mid]))


def to_float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


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


# ------------------------------------------------------------
# Posthoc / TRF getters
# ------------------------------------------------------------
def get_posthoc_side(row: dict):
    return row.get("posthoc_trf_chosen_side", row.get("posthoc_chosen_side"))


def get_posthoc_valid(row: dict):
    x = to_bool_or_none(row.get("posthoc_trf_valid"))
    if x is not None:
        return x

    ratio = get_posthoc_ratio(row)
    rel_margin = get_posthoc_rel_margin(row)
    dominance = get_posthoc_dominance(row)

    if ratio is None and rel_margin is None and dominance is None:
        return None

    if ratio == 0.0 and rel_margin == 0.0 and dominance == 0.0:
        return False

    return True


def get_posthoc_rel_margin(row: dict):
    return to_float_or_none(row.get("posthoc_trf_rel_margin"))


def get_posthoc_ratio(row: dict):
    x = to_float_or_none(row.get("posthoc_trf_ratio"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_posthoc_dominance(row: dict):
    x = to_float_or_none(row.get("posthoc_trf_dominance"))
    if x is None:
        return None
    if x < 0.0 or x > 1.0:
        return None
    return x


def get_posthoc_total_evidence(row: dict):
    x = to_float_or_none(row.get("posthoc_trf_total_evidence"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_posthoc_adaptive_score(row: dict):
    x = to_float_or_none(row.get("posthoc_trf_adaptive_score"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_posthoc_ref_time(row: dict):
    return to_float_or_none(row.get("posthoc_trf_ref_time"))


# ------------------------------------------------------------
# Forward-only guess getters
# ------------------------------------------------------------
def get_forward_guess_valid(row: dict):
    return to_bool_or_none(row.get("forward_guess_valid"))


def get_forward_guess_side(row: dict):
    return row.get("forward_guess_chosen_side")


def get_forward_guess_rel_margin(row: dict):
    return to_float_or_none(row.get("forward_guess_rel_margin"))


def get_forward_guess_ratio(row: dict):
    x = to_float_or_none(row.get("forward_guess_ratio"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_forward_guess_dominance(row: dict):
    x = to_float_or_none(row.get("forward_guess_dominance"))
    if x is None:
        return None
    if x < 0.0 or x > 1.0:
        return None
    return x


def get_forward_guess_total_evidence(row: dict):
    x = to_float_or_none(row.get("forward_guess_total_evidence"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_forward_guess_adaptive_score(row: dict):
    x = to_float_or_none(row.get("forward_guess_adaptive_score"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_forward_guess_ref_time(row: dict):
    return to_float_or_none(row.get("forward_guess_ref_time"))


# ------------------------------------------------------------
# Slit-pass getters
# ------------------------------------------------------------
def get_slit_pass_used(row: dict):
    return to_bool_or_none(row.get("slit_pass_debug_used"))


def get_slit_pass_valid(row: dict):
    x = to_bool_or_none(row.get("slit_pass_valid"))
    if x is not None:
        return x

    side = get_slit_pass_side(row)
    if side in {"upper", "lower"}:
        return True
    return None


def get_slit_pass_side(row: dict):
    return row.get("slit_pass_chosen_side")


def get_slit_pass_ref_time(row: dict):
    return to_float_or_none(row.get("slit_pass_ref_time"))


def get_slit_pass_rel_margin(row: dict):
    return to_float_or_none(row.get("slit_pass_rel_margin"))


def get_slit_pass_ratio(row: dict):
    x = to_float_or_none(row.get("slit_pass_ratio"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_slit_pass_dominance(row: dict):
    x = to_float_or_none(row.get("slit_pass_dominance"))
    if x is None:
        return None
    if x < 0.0 or x > 1.0:
        return None
    return x


def get_slit_pass_total_evidence(row: dict):
    x = to_float_or_none(row.get("slit_pass_total_evidence"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_slit_pass_score(row: dict):
    x = to_float_or_none(row.get("slit_pass_score"))
    if x is None:
        return None
    if x < 0.0:
        return None
    return x


def get_slit_vs_click_different(row: dict):
    x = to_bool_or_none(row.get("slit_vs_click_different"))
    if x is not None:
        return x

    slit = get_slit_pass_side(row)
    click = row.get("click_side")
    if slit is None or click is None:
        return None
    return slit != click


def get_slit_to_click_transition(row: dict):
    x = row.get("slit_to_click_transition")
    if x:
        return x
    slit = get_slit_pass_side(row)
    click = row.get("click_side")
    if slit is None or click is None:
        return None
    return f"{slit}_slit->{click}_click"


def get_slit_vs_trf_different(row: dict):
    slit = get_slit_pass_side(row)
    trf = get_posthoc_side(row)
    if slit is None or trf is None:
        return None
    return slit != trf


def get_slit_vs_forward_different(row: dict):
    slit = get_slit_pass_side(row)
    fwd = get_forward_guess_side(row)
    if slit is None or fwd is None:
        return None
    return slit != fwd


# ------------------------------------------------------------
# Comparison helpers
# ------------------------------------------------------------
def get_forward_vs_trf_different(row: dict):
    x = to_bool_or_none(row.get("forward_vs_trf_different"))
    if x is not None:
        return x

    fwd = get_forward_guess_side(row)
    trf = get_posthoc_side(row)
    if fwd is None or trf is None:
        return None
    return fwd != trf


def get_forward_vs_click_different(row: dict):
    x = to_bool_or_none(row.get("forward_vs_click_different"))
    if x is not None:
        return x

    fwd = get_forward_guess_side(row)
    click = row.get("click_side")
    if fwd is None or click is None:
        return None
    return fwd != click


def get_trf_vs_click_match(row: dict):
    x = to_bool_or_none(row.get("trf_vs_click_match"))
    if x is not None:
        return x

    trf = get_posthoc_side(row)
    click = row.get("click_side")
    if trf is None or click is None:
        return None
    return trf == click


def get_interesting_forward_trf_click_case(row: dict):
    x = to_bool_or_none(row.get("interesting_forward_trf_click_case"))
    if x is not None:
        return x

    diff = get_forward_vs_trf_different(row)
    match = get_trf_vs_click_match(row)
    if diff is None or match is None:
        return None
    return diff and match


# ------------------------------------------------------------
# Buckets
# ------------------------------------------------------------
def get_rel_margin_bucket(x: float | None) -> str | None:
    if x is None:
        return None
    if x < 0.005:
        return "<0.005"
    if x < 0.01:
        return "0.005-0.01"
    if x < 0.02:
        return "0.01-0.02"
    return ">=0.02"


def get_ratio_bucket(x: float | None) -> str | None:
    if x is None:
        return None
    if x < 1.01:
        return "<1.01"
    if x < 1.05:
        return "1.01-1.05"
    if x < 1.20:
        return "1.05-1.20"
    return ">=1.20"


def get_dominance_bucket(x: float | None) -> str | None:
    if x is None:
        return None
    if x < 0.52:
        return "<0.52"
    if x < 0.55:
        return "0.52-0.55"
    if x < 0.60:
        return "0.55-0.60"
    return ">=0.60"


def get_adaptive_score_bucket(x: float | None) -> str | None:
    if x is None:
        return None
    if x < 1e-8:
        return "<1e-8"
    if x < 1e-6:
        return "1e-8-1e-6"
    if x < 1e-4:
        return "1e-6-1e-4"
    return ">=1e-4"


def print_accuracy_line(title: str, rows: list[dict], pred_fn):
    if not rows:
        print(f"{title}: n/a")
        return
    matches = 0
    valid_n = 0
    for r in rows:
        pred = pred_fn(r)
        click = r.get("click_side")
        if pred is None or click is None:
            continue
        valid_n += 1
        if pred == click:
            matches += 1
    if valid_n == 0:
        print(f"{title}: n/a")
        return
    print(f"{title}: {matches} / {valid_n} ({pct(matches, valid_n)})")


def main() -> int:
    args = parse_args()
    path = Path(args.summary_jsonl)

    if not path.exists():
        print(f"summary.jsonl not found: {path}")
        return 1

    rows = load_jsonl(path)
    if not rows:
        print(f"No rows found in: {path}")
        return 1

    ok_rows = [r for r in rows if r.get("returncode") == 0]
    err_rows = [r for r in rows if r.get("returncode") != 0]

    total = len(rows)
    ok_total = len(ok_rows)

    print("=" * 80)
    print("Posthoc / TRF / slit-pass seed sweep analysis")
    print("=" * 80)
    print(f"Summary file : {path}")
    print(f"Rows total   : {total}")
    print(f"Successful   : {len(ok_rows)}")
    print(f"Failed       : {len(err_rows)}")
    print("")

    if ok_total == 0:
        print("No successful rows to analyze.")
        return 0

    # ------------------------------------------------------------
    # Run config glimpse
    # ------------------------------------------------------------
    adaptive_rows = [r for r in ok_rows if r.get("trf_use_adaptive_ref") is not None]
    if adaptive_rows:
        sample = adaptive_rows[0]
        print("TRF run config")
        print(f"  adaptive ref      : {sample.get('trf_use_adaptive_ref')}")
        print(f"  ref t min frac    : {sample.get('trf_ref_t_min_frac')}")
        print(f"  ref t max frac    : {sample.get('trf_ref_t_max_frac')}")
        print(f"  corridor x start  : {sample.get('trf_corridor_x_frac_start')}")
        print(f"  corridor y sigma  : {sample.get('trf_corridor_y_sigma')}")
        print(f"  x weight power    : {sample.get('trf_corridor_x_weight_power')}")
        print("")

    slit_debug_rows = [r for r in ok_rows if get_slit_pass_used(r) is not None]
    if slit_debug_rows:
        print("Slit-pass debug")
        sample = slit_debug_rows[0]
        print(f"  enabled           : {sample.get('slit_pass_debug_used')}")
        print("")

    # ------------------------------------------------------------
    # Row groups
    # ------------------------------------------------------------
    click_counter = Counter(r.get("click_side") for r in ok_rows)

    trf_rows = [r for r in ok_rows if get_posthoc_side(r) in {"upper", "lower"}]
    trf_counter = Counter(get_posthoc_side(r) for r in trf_rows)

    valid_rows = [r for r in trf_rows if get_posthoc_valid(r) is True]
    invalid_rows = [r for r in trf_rows if get_posthoc_valid(r) is False]
    unknown_validity_rows = [r for r in trf_rows if get_posthoc_valid(r) is None]

    valid_counter = Counter(get_posthoc_side(r) for r in valid_rows)
    invalid_counter = Counter(get_posthoc_side(r) for r in invalid_rows)

    fwd_rows = [r for r in ok_rows if get_forward_guess_side(r) in {"upper", "lower"}]
    fwd_valid_rows = [r for r in fwd_rows if get_forward_guess_valid(r) is True]
    fwd_counter = Counter(get_forward_guess_side(r) for r in fwd_rows)

    slit_rows = [r for r in ok_rows if get_slit_pass_side(r) in {"upper", "lower"}]
    slit_valid_rows = [r for r in slit_rows if get_slit_pass_valid(r) is True]
    slit_counter = Counter(get_slit_pass_side(r) for r in slit_rows)

    print_counter("Click side distribution", click_counter, ok_total)
    print("")
    print_counter("Forward-only chosen side distribution", fwd_counter, max(len(fwd_rows), 1))
    print("")
    print_counter("Slit-pass chosen side distribution", slit_counter, max(len(slit_rows), 1))
    print("")
    print_counter("Posthoc TRF chosen side distribution", trf_counter, max(len(trf_rows), 1))
    print("")
    print("TRF validity")
    print(f"  valid   : {len(valid_rows)} / {max(len(trf_rows), 1)} ({pct(len(valid_rows), max(len(trf_rows), 1))})")
    print(f"  invalid : {len(invalid_rows)} / {max(len(trf_rows), 1)} ({pct(len(invalid_rows), max(len(trf_rows), 1))})")
    print(f"  unknown : {len(unknown_validity_rows)} / {max(len(trf_rows), 1)} ({pct(len(unknown_validity_rows), max(len(trf_rows), 1))})")
    print("")
    print_counter("Valid TRF chosen side distribution", valid_counter, max(len(valid_rows), 1))
    print("")
    print_counter("Invalid TRF chosen side distribution", invalid_counter, max(len(invalid_rows), 1))
    print("")

    # ------------------------------------------------------------
    # Agreement
    # ------------------------------------------------------------
    trf_matches_click = [r for r in trf_rows if get_posthoc_side(r) == r.get("click_side")]
    trf_mismatches_click = [r for r in trf_rows if get_posthoc_side(r) != r.get("click_side")]

    valid_matches_click = [r for r in valid_rows if get_posthoc_side(r) == r.get("click_side")]
    valid_mismatches_click = [r for r in valid_rows if get_posthoc_side(r) != r.get("click_side")]

    invalid_matches_click = [r for r in invalid_rows if get_posthoc_side(r) == r.get("click_side")]
    invalid_mismatches_click = [r for r in invalid_rows if get_posthoc_side(r) != r.get("click_side")]

    slit_matches_click = [r for r in slit_rows if get_slit_pass_side(r) == r.get("click_side")]
    slit_mismatches_click = [r for r in slit_rows if get_slit_pass_side(r) != r.get("click_side")]

    print("Forward-only-vs-click agreement")
    print_accuracy_line("  forward guess matches click", fwd_rows, get_forward_guess_side)
    print("")

    print("Slit-pass-vs-click agreement")
    print_accuracy_line("  slit pass matches click", slit_rows, get_slit_pass_side)
    print("")

    print("TRF-vs-click agreement (all rows)")
    print(f"  posthoc TRF matches click   : {len(trf_matches_click)} / {max(len(trf_rows), 1)} ({pct(len(trf_matches_click), max(len(trf_rows), 1))})")
    print(f"  posthoc TRF mismatches click: {len(trf_mismatches_click)} / {max(len(trf_rows), 1)} ({pct(len(trf_mismatches_click), max(len(trf_rows), 1))})")
    print("")

    print("TRF-vs-click agreement (valid rows only)")
    print(f"  valid TRF matches click   : {len(valid_matches_click)} / {max(len(valid_rows), 1)} ({pct(len(valid_matches_click), max(len(valid_rows), 1))})")
    print(f"  valid TRF mismatches click: {len(valid_mismatches_click)} / {max(len(valid_rows), 1)} ({pct(len(valid_mismatches_click), max(len(valid_rows), 1))})")
    print("")

    if invalid_rows:
        print("TRF-vs-click agreement (invalid rows only)")
        print(f"  invalid TRF matches click   : {len(invalid_matches_click)} / {len(invalid_rows)} ({pct(len(invalid_matches_click), len(invalid_rows))})")
        print(f"  invalid TRF mismatches click: {len(invalid_mismatches_click)} / {len(invalid_rows)} ({pct(len(invalid_mismatches_click), len(invalid_rows))})")
        print("")

    slit_to_click = Counter(
        f"{get_slit_pass_side(r)} -> {r.get('click_side')}"
        for r in slit_rows
    )
    trf_to_click = Counter(
        f"{get_posthoc_side(r)} -> {r.get('click_side')}"
        for r in trf_rows
    )
    valid_to_click = Counter(
        f"{get_posthoc_side(r)} -> {r.get('click_side')}"
        for r in valid_rows
    )
    invalid_to_click = Counter(
        f"{get_posthoc_side(r)} -> {r.get('click_side')}"
        for r in invalid_rows
    )

    print_counter("Slit-pass -> click (all rows)", slit_to_click, max(len(slit_rows), 1))
    print("")
    print_counter("Posthoc TRF -> click (all rows)", trf_to_click, max(len(trf_rows), 1))
    print("")
    print_counter("Posthoc TRF -> click (valid rows)", valid_to_click, max(len(valid_rows), 1))
    print("")
    if invalid_rows:
        print_counter("Posthoc TRF -> click (invalid rows)", invalid_to_click, len(invalid_rows))
        print("")

    # ------------------------------------------------------------
    # Forward vs slit vs TRF vs click
    # ------------------------------------------------------------
    comparable_rows = [
        r for r in ok_rows
        if get_forward_guess_side(r) in {"upper", "lower"}
        and get_posthoc_side(r) in {"upper", "lower"}
    ]

    slit_comparable_rows = [
        r for r in ok_rows
        if get_slit_pass_side(r) in {"upper", "lower"}
        and get_posthoc_side(r) in {"upper", "lower"}
    ]

    slit_forward_comparable_rows = [
        r for r in ok_rows
        if get_slit_pass_side(r) in {"upper", "lower"}
        and get_forward_guess_side(r) in {"upper", "lower"}
    ]

    forward_vs_trf_diff_rows = [r for r in comparable_rows if get_forward_vs_trf_different(r) is True]
    forward_vs_trf_same_rows = [r for r in comparable_rows if get_forward_vs_trf_different(r) is False]
    interesting_rows = [r for r in comparable_rows if get_interesting_forward_trf_click_case(r) is True]

    slit_vs_trf_diff_rows = [r for r in slit_comparable_rows if get_slit_vs_trf_different(r) is True]
    slit_vs_trf_same_rows = [r for r in slit_comparable_rows if get_slit_vs_trf_different(r) is False]

    slit_vs_forward_diff_rows = [r for r in slit_forward_comparable_rows if get_slit_vs_forward_different(r) is True]
    slit_vs_forward_same_rows = [r for r in slit_forward_comparable_rows if get_slit_vs_forward_different(r) is False]

    print("Forward vs TRF comparison")
    print(f"  comparable rows         : {len(comparable_rows)}")
    print(f"  forward == TRF          : {len(forward_vs_trf_same_rows)} / {max(len(comparable_rows), 1)} ({pct(len(forward_vs_trf_same_rows), max(len(comparable_rows), 1))})")
    print(f"  forward != TRF          : {len(forward_vs_trf_diff_rows)} / {max(len(comparable_rows), 1)} ({pct(len(forward_vs_trf_diff_rows), max(len(comparable_rows), 1))})")
    print(f"  interesting cases       : {len(interesting_rows)} / {max(len(comparable_rows), 1)} ({pct(len(interesting_rows), max(len(comparable_rows), 1))})")
    print("")

    print("Slit vs TRF comparison")
    print(f"  comparable rows         : {len(slit_comparable_rows)}")
    print(f"  slit == TRF             : {len(slit_vs_trf_same_rows)} / {max(len(slit_comparable_rows), 1)} ({pct(len(slit_vs_trf_same_rows), max(len(slit_comparable_rows), 1))})")
    print(f"  slit != TRF             : {len(slit_vs_trf_diff_rows)} / {max(len(slit_comparable_rows), 1)} ({pct(len(slit_vs_trf_diff_rows), max(len(slit_comparable_rows), 1))})")
    print("")

    print("Slit vs forward comparison")
    print(f"  comparable rows         : {len(slit_forward_comparable_rows)}")
    print(f"  slit == forward         : {len(slit_vs_forward_same_rows)} / {max(len(slit_forward_comparable_rows), 1)} ({pct(len(slit_vs_forward_same_rows), max(len(slit_forward_comparable_rows), 1))})")
    print(f"  slit != forward         : {len(slit_vs_forward_diff_rows)} / {max(len(slit_forward_comparable_rows), 1)} ({pct(len(slit_vs_forward_diff_rows), max(len(slit_forward_comparable_rows), 1))})")
    print("")

    fwd_to_trf = Counter(
        f"{get_forward_guess_side(r)} -> {get_posthoc_side(r)}"
        for r in comparable_rows
    )
    slit_to_trf = Counter(
        f"{get_slit_pass_side(r)} -> {get_posthoc_side(r)}"
        for r in slit_comparable_rows
    )
    slit_to_forward = Counter(
        f"{get_slit_pass_side(r)} -> {get_forward_guess_side(r)}"
        for r in slit_forward_comparable_rows
    )

    print_counter("Forward-only -> TRF", fwd_to_trf, max(len(comparable_rows), 1))
    print("")
    print_counter("Slit-pass -> TRF", slit_to_trf, max(len(slit_comparable_rows), 1))
    print("")
    print_counter("Slit-pass -> forward", slit_to_forward, max(len(slit_forward_comparable_rows), 1))
    print("")

    interesting_to_click = Counter(
        f"{get_forward_guess_side(r)} -> {get_posthoc_side(r)} -> {r.get('click_side')}"
        for r in interesting_rows
    )
    if interesting_rows:
        print_counter("Interesting cases: forward -> TRF -> click", interesting_to_click, len(interesting_rows))
        print("")

    four_stage_rows = [
        r for r in ok_rows
        if get_slit_pass_side(r) in {"upper", "lower"}
        and get_forward_guess_side(r) in {"upper", "lower"}
        and get_posthoc_side(r) in {"upper", "lower"}
        and r.get("click_side") in {"upper", "lower"}
    ]
    four_stage_counter = Counter(
        f"{get_slit_pass_side(r)} -> {get_forward_guess_side(r)} -> {get_posthoc_side(r)} -> {r.get('click_side')}"
        for r in four_stage_rows
    )
    print_counter("Slit -> forward -> TRF -> click", four_stage_counter, max(len(four_stage_rows), 1))
    print("")

    # ------------------------------------------------------------
    # Evidence stats: valid TRF rows
    # ------------------------------------------------------------
    rel_margins = [get_posthoc_rel_margin(r) for r in valid_rows]
    rel_margins = [x for x in rel_margins if x is not None]

    ratios = [get_posthoc_ratio(r) for r in valid_rows]
    ratios = [x for x in ratios if x is not None]

    dominances = [get_posthoc_dominance(r) for r in valid_rows]
    dominances = [x for x in dominances if x is not None]

    total_evidences = [get_posthoc_total_evidence(r) for r in valid_rows]
    total_evidences = [x for x in total_evidences if x is not None]

    adaptive_scores = [get_posthoc_adaptive_score(r) for r in valid_rows]
    adaptive_scores = [x for x in adaptive_scores if x is not None]

    ref_times = [get_posthoc_ref_time(r) for r in valid_rows]
    ref_times = [x for x in ref_times if x is not None]

    if rel_margins or ratios or dominances or total_evidences or adaptive_scores or ref_times:
        print("Posthoc TRF evidence stats (valid rows only)")
        if rel_margins:
            print(f"  rel margin min/avg/median/max : {min(rel_margins):.6f} / {mean_of(rel_margins):.6f} / {median_of(rel_margins):.6f} / {max(rel_margins):.6f}")
        if ratios:
            print(f"  ratio min/avg/median/max      : {min(ratios):.6f} / {mean_of(ratios):.6f} / {median_of(ratios):.6f} / {max(ratios):.6f}")
        if dominances:
            print(f"  dominance min/avg/median/max  : {min(dominances):.6f} / {mean_of(dominances):.6f} / {median_of(dominances):.6f} / {max(dominances):.6f}")
        if total_evidences:
            print(f"  total ev min/avg/median/max   : {min(total_evidences):.6e} / {mean_of(total_evidences):.6e} / {median_of(total_evidences):.6e} / {max(total_evidences):.6e}")
        if adaptive_scores:
            print(f"  adaptive score min/avg/median/max: {min(adaptive_scores):.6e} / {mean_of(adaptive_scores):.6e} / {median_of(adaptive_scores):.6e} / {max(adaptive_scores):.6e}")
        if ref_times:
            print(f"  ref time min/avg/median/max   : {min(ref_times):.6f} / {mean_of(ref_times):.6f} / {median_of(ref_times):.6f} / {max(ref_times):.6f}")
        print("")

    # ------------------------------------------------------------
    # Forward-only evidence stats
    # ------------------------------------------------------------
    fwd_rel_margins = [get_forward_guess_rel_margin(r) for r in fwd_valid_rows]
    fwd_rel_margins = [x for x in fwd_rel_margins if x is not None]

    fwd_ratios = [get_forward_guess_ratio(r) for r in fwd_valid_rows]
    fwd_ratios = [x for x in fwd_ratios if x is not None]

    fwd_dominances = [get_forward_guess_dominance(r) for r in fwd_valid_rows]
    fwd_dominances = [x for x in fwd_dominances if x is not None]

    fwd_total_evidences = [get_forward_guess_total_evidence(r) for r in fwd_valid_rows]
    fwd_total_evidences = [x for x in fwd_total_evidences if x is not None]

    fwd_adaptive_scores = [get_forward_guess_adaptive_score(r) for r in fwd_valid_rows]
    fwd_adaptive_scores = [x for x in fwd_adaptive_scores if x is not None]

    fwd_ref_times = [get_forward_guess_ref_time(r) for r in fwd_valid_rows]
    fwd_ref_times = [x for x in fwd_ref_times if x is not None]

    if fwd_rel_margins or fwd_ratios or fwd_dominances or fwd_total_evidences or fwd_adaptive_scores or fwd_ref_times:
        print("Forward-only evidence stats (valid rows only)")
        if fwd_rel_margins:
            print(f"  rel margin min/avg/median/max : {min(fwd_rel_margins):.6f} / {mean_of(fwd_rel_margins):.6f} / {median_of(fwd_rel_margins):.6f} / {max(fwd_rel_margins):.6f}")
        if fwd_ratios:
            print(f"  ratio min/avg/median/max      : {min(fwd_ratios):.6f} / {mean_of(fwd_ratios):.6f} / {median_of(fwd_ratios):.6f} / {max(fwd_ratios):.6f}")
        if fwd_dominances:
            print(f"  dominance min/avg/median/max  : {min(fwd_dominances):.6f} / {mean_of(fwd_dominances):.6f} / {median_of(fwd_dominances):.6f} / {max(fwd_dominances):.6f}")
        if fwd_total_evidences:
            print(f"  total ev min/avg/median/max   : {min(fwd_total_evidences):.6e} / {mean_of(fwd_total_evidences):.6e} / {median_of(fwd_total_evidences):.6e} / {max(fwd_total_evidences):.6e}")
        if fwd_adaptive_scores:
            print(f"  adaptive score min/avg/median/max: {min(fwd_adaptive_scores):.6e} / {mean_of(fwd_adaptive_scores):.6e} / {median_of(fwd_adaptive_scores):.6e} / {max(fwd_adaptive_scores):.6e}")
        if fwd_ref_times:
            print(f"  ref time min/avg/median/max   : {min(fwd_ref_times):.6f} / {mean_of(fwd_ref_times):.6f} / {median_of(fwd_ref_times):.6f} / {max(fwd_ref_times):.6f}")
        print("")

    # ------------------------------------------------------------
    # Slit-pass evidence stats
    # ------------------------------------------------------------
    slit_rel_margins = [get_slit_pass_rel_margin(r) for r in slit_valid_rows]
    slit_rel_margins = [x for x in slit_rel_margins if x is not None]

    slit_ratios = [get_slit_pass_ratio(r) for r in slit_valid_rows]
    slit_ratios = [x for x in slit_ratios if x is not None]

    slit_dominances = [get_slit_pass_dominance(r) for r in slit_valid_rows]
    slit_dominances = [x for x in slit_dominances if x is not None]

    slit_total_evidences = [get_slit_pass_total_evidence(r) for r in slit_valid_rows]
    slit_total_evidences = [x for x in slit_total_evidences if x is not None]

    slit_scores = [get_slit_pass_score(r) for r in slit_valid_rows]
    slit_scores = [x for x in slit_scores if x is not None]

    slit_ref_times = [get_slit_pass_ref_time(r) for r in slit_valid_rows]
    slit_ref_times = [x for x in slit_ref_times if x is not None]

    if slit_rel_margins or slit_ratios or slit_dominances or slit_total_evidences or slit_scores or slit_ref_times:
        print("Slit-pass evidence stats (valid rows only)")
        if slit_rel_margins:
            print(f"  rel margin min/avg/median/max : {min(slit_rel_margins):.6f} / {mean_of(slit_rel_margins):.6f} / {median_of(slit_rel_margins):.6f} / {max(slit_rel_margins):.6f}")
        if slit_ratios:
            print(f"  ratio min/avg/median/max      : {min(slit_ratios):.6f} / {mean_of(slit_ratios):.6f} / {median_of(slit_ratios):.6f} / {max(slit_ratios):.6f}")
        if slit_dominances:
            print(f"  dominance min/avg/median/max  : {min(slit_dominances):.6f} / {mean_of(slit_dominances):.6f} / {median_of(slit_dominances):.6f} / {max(slit_dominances):.6f}")
        if slit_total_evidences:
            print(f"  total ev min/avg/median/max   : {min(slit_total_evidences):.6e} / {mean_of(slit_total_evidences):.6e} / {median_of(slit_total_evidences):.6e} / {max(slit_total_evidences):.6e}")
        if slit_scores:
            print(f"  score min/avg/median/max      : {min(slit_scores):.6e} / {mean_of(slit_scores):.6e} / {median_of(slit_scores):.6e} / {max(slit_scores):.6e}")
        if slit_ref_times:
            print(f"  ref time min/avg/median/max   : {min(slit_ref_times):.6f} / {mean_of(slit_ref_times):.6f} / {median_of(slit_ref_times):.6f} / {max(slit_ref_times):.6f}")
        print("")

    # ------------------------------------------------------------
    # Accuracy by thresholds
    # ------------------------------------------------------------
    print("Valid-only TRF accuracy by dominance threshold")
    for thr in [0.52, 0.55, 0.60, 0.70, 0.80, 0.90]:
        rows_thr = [r for r in valid_rows if (get_posthoc_dominance(r) or -1.0) >= thr]
        print_accuracy_line(f"  dominance >= {thr:.2f}", rows_thr, get_posthoc_side)
    print("")

    print("Valid-only TRF accuracy by adaptive score threshold")
    for thr in [1e-8, 1e-6, 1e-4, 1e-3]:
        rows_thr = [r for r in valid_rows if (get_posthoc_adaptive_score(r) or -1.0) >= thr]
        print_accuracy_line(f"  adaptive_score >= {thr:.0e}", rows_thr, get_posthoc_side)
    print("")

    print("Forward-only accuracy by dominance threshold")
    for thr in [0.52, 0.55, 0.60, 0.70, 0.80, 0.90]:
        rows_thr = [r for r in fwd_valid_rows if (get_forward_guess_dominance(r) or -1.0) >= thr]
        print_accuracy_line(f"  dominance >= {thr:.2f}", rows_thr, get_forward_guess_side)
    print("")

    # ------------------------------------------------------------
    # Buckets: valid TRF rows
    # ------------------------------------------------------------
    rel_bucket_counter = Counter()
    rel_bucket_to_click = Counter()

    ratio_bucket_counter = Counter()
    ratio_bucket_to_click = Counter()

    dom_bucket_counter = Counter()
    dom_bucket_to_click = Counter()

    adaptive_bucket_counter = Counter()
    adaptive_bucket_to_click = Counter()

    for r in valid_rows:
        match = "match" if get_posthoc_side(r) == r.get("click_side") else "mismatch"

        rel_bucket = get_rel_margin_bucket(get_posthoc_rel_margin(r))
        if rel_bucket is not None:
            rel_bucket_counter[rel_bucket] += 1
            rel_bucket_to_click[f"{rel_bucket} -> {match}"] += 1

        ratio_bucket = get_ratio_bucket(get_posthoc_ratio(r))
        if ratio_bucket is not None:
            ratio_bucket_counter[ratio_bucket] += 1
            ratio_bucket_to_click[f"{ratio_bucket} -> {match}"] += 1

        dom_bucket = get_dominance_bucket(get_posthoc_dominance(r))
        if dom_bucket is not None:
            dom_bucket_counter[dom_bucket] += 1
            dom_bucket_to_click[f"{dom_bucket} -> {match}"] += 1

        adaptive_bucket = get_adaptive_score_bucket(get_posthoc_adaptive_score(r))
        if adaptive_bucket is not None:
            adaptive_bucket_counter[adaptive_bucket] += 1
            adaptive_bucket_to_click[f"{adaptive_bucket} -> {match}"] += 1

    print_counter("TRF rel-margin bucket distribution (valid rows)", rel_bucket_counter, max(len(valid_rows), 1))
    print("")
    print_counter("TRF rel-margin bucket -> click (valid rows)", rel_bucket_to_click, max(len(valid_rows), 1))
    print("")
    print_counter("TRF ratio bucket distribution (valid rows)", ratio_bucket_counter, max(len(valid_rows), 1))
    print("")
    print_counter("TRF ratio bucket -> click (valid rows)", ratio_bucket_to_click, max(len(valid_rows), 1))
    print("")
    print_counter("TRF dominance bucket distribution (valid rows)", dom_bucket_counter, max(len(valid_rows), 1))
    print("")
    print_counter("TRF dominance bucket -> click (valid rows)", dom_bucket_to_click, max(len(valid_rows), 1))
    print("")
    print_counter("TRF adaptive-score bucket distribution (valid rows)", adaptive_bucket_counter, max(len(valid_rows), 1))
    print("")
    print_counter("TRF adaptive-score bucket -> click (valid rows)", adaptive_bucket_to_click, max(len(valid_rows), 1))
    print("")

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------
    elapsed = [to_float_or_none(r.get("elapsed_sec")) for r in ok_rows]
    elapsed = [x for x in elapsed if x is not None]
    if elapsed:
        print("Runtime stats")
        print(f"  min/avg/median/max sec: {min(elapsed):.2f} / {mean_of(elapsed):.2f} / {median_of(elapsed):.2f} / {max(elapsed):.2f}")
        print("")

    # ------------------------------------------------------------
    # Interesting categories
    # ------------------------------------------------------------
    categories: dict[str, list[dict]] = defaultdict(list)

    for r in ok_rows:
        click = r.get("click_side")
        trf_side = get_posthoc_side(r)
        fwd_side = get_forward_guess_side(r)
        slit_side = get_slit_pass_side(r)

        valid = get_posthoc_valid(r)
        rel_margin = get_posthoc_rel_margin(r)
        ratio = get_posthoc_ratio(r)
        dominance = get_posthoc_dominance(r)
        adaptive_score = get_posthoc_adaptive_score(r)

        if valid is True:
            categories["trf_valid"].append(r)
        elif valid is False:
            categories["trf_invalid"].append(r)
        else:
            categories["trf_unknown_validity"].append(r)

        if trf_side is not None:
            if trf_side == click:
                categories["trf_matches_click"].append(r)
            else:
                categories["trf_mismatches_click"].append(r)

        if fwd_side is not None:
            if fwd_side == click:
                categories["forward_matches_click"].append(r)
            else:
                categories["forward_mismatches_click"].append(r)

        if slit_side is not None:
            if slit_side == click:
                categories["slit_matches_click"].append(r)
            else:
                categories["slit_mismatches_click"].append(r)

        if valid is True and trf_side == click:
            categories["trf_valid_and_match"].append(r)
        if valid is True and trf_side != click:
            categories["trf_valid_and_mismatch"].append(r)
        if valid is False and trf_side == click:
            categories["trf_invalid_and_match"].append(r)
        if valid is False and trf_side != click:
            categories["trf_invalid_and_mismatch"].append(r)

        if trf_side == "upper" and click == "upper":
            categories["trf_upper_and_click_upper"].append(r)
        if trf_side == "upper" and click == "lower":
            categories["trf_upper_but_click_lower"].append(r)
        if trf_side == "lower" and click == "lower":
            categories["trf_lower_and_click_lower"].append(r)
        if trf_side == "lower" and click == "upper":
            categories["trf_lower_but_click_upper"].append(r)

        if slit_side == "upper" and click == "upper":
            categories["slit_upper_and_click_upper"].append(r)
        if slit_side == "upper" and click == "lower":
            categories["slit_upper_but_click_lower"].append(r)
        if slit_side == "lower" and click == "lower":
            categories["slit_lower_and_click_lower"].append(r)
        if slit_side == "lower" and click == "upper":
            categories["slit_lower_but_click_upper"].append(r)

        diff = get_forward_vs_trf_different(r)
        if diff is True:
            categories["forward_vs_trf_different"].append(r)
        if diff is False:
            categories["forward_vs_trf_same"].append(r)

        slit_trf_diff = get_slit_vs_trf_different(r)
        if slit_trf_diff is True:
            categories["slit_vs_trf_different"].append(r)
        if slit_trf_diff is False:
            categories["slit_vs_trf_same"].append(r)

        slit_fwd_diff = get_slit_vs_forward_different(r)
        if slit_fwd_diff is True:
            categories["slit_vs_forward_different"].append(r)
        if slit_fwd_diff is False:
            categories["slit_vs_forward_same"].append(r)

        slit_click_diff = get_slit_vs_click_different(r)
        if slit_click_diff is True:
            categories["slit_vs_click_different"].append(r)
        if slit_click_diff is False:
            categories["slit_vs_click_same"].append(r)

        interesting = get_interesting_forward_trf_click_case(r)
        if interesting is True:
            categories["interesting_forward_trf_click_case"].append(r)

        transition = get_slit_to_click_transition(r)
        if transition is not None:
            categories[f"transition_{transition}"].append(r)

        if (
            slit_side in {"upper", "lower"}
            and fwd_side in {"upper", "lower"}
            and trf_side in {"upper", "lower"}
            and click in {"upper", "lower"}
        ):
            categories[f"path_{slit_side}_{fwd_side}_{trf_side}_{click}"].append(r)

        if valid is True:
            if rel_margin is not None and rel_margin < float(args.low_rel_margin_threshold):
                categories["trf_low_rel_margin"].append(r)
            if rel_margin is not None and rel_margin >= float(args.low_rel_margin_threshold):
                categories["trf_not_low_rel_margin"].append(r)

            if ratio is not None and ratio >= float(args.high_ratio_threshold):
                categories["trf_high_ratio"].append(r)
            if ratio is not None and ratio < float(args.high_ratio_threshold):
                categories["trf_not_high_ratio"].append(r)

            if dominance is not None and dominance >= float(args.high_dominance_threshold):
                categories["trf_high_dominance"].append(r)
            if dominance is not None and dominance < float(args.high_dominance_threshold):
                categories["trf_not_high_dominance"].append(r)

            if adaptive_score is not None and adaptive_score >= float(args.high_adaptive_score_threshold):
                categories["trf_high_adaptive_score"].append(r)
            if adaptive_score is not None and adaptive_score < float(args.high_adaptive_score_threshold):
                categories["trf_not_high_adaptive_score"].append(r)

            if (
                rel_margin is not None
                and rel_margin < float(args.low_rel_margin_threshold)
                and trf_side != click
            ):
                categories["trf_low_rel_margin_and_mismatch"].append(r)

            if (
                ratio is not None
                and ratio >= float(args.high_ratio_threshold)
                and trf_side == click
            ):
                categories["trf_high_ratio_and_match"].append(r)

            if (
                dominance is not None
                and dominance >= float(args.high_dominance_threshold)
                and trf_side == click
            ):
                categories["trf_high_dominance_and_match"].append(r)

            if (
                adaptive_score is not None
                and adaptive_score >= float(args.high_adaptive_score_threshold)
                and trf_side == click
            ):
                categories["trf_high_adaptive_score_and_match"].append(r)

            rel_bucket = get_rel_margin_bucket(rel_margin)
            if rel_bucket is not None:
                categories[f"trf_rel_margin_bucket_{rel_bucket}"].append(r)
                if trf_side == click:
                    categories[f"trf_rel_margin_bucket_{rel_bucket}_match"].append(r)
                else:
                    categories[f"trf_rel_margin_bucket_{rel_bucket}_mismatch"].append(r)

            ratio_bucket = get_ratio_bucket(ratio)
            if ratio_bucket is not None:
                categories[f"trf_ratio_bucket_{ratio_bucket}"].append(r)
                if trf_side == click:
                    categories[f"trf_ratio_bucket_{ratio_bucket}_match"].append(r)
                else:
                    categories[f"trf_ratio_bucket_{ratio_bucket}_mismatch"].append(r)

            dom_bucket = get_dominance_bucket(dominance)
            if dom_bucket is not None:
                categories[f"trf_dominance_bucket_{dom_bucket}"].append(r)
                if trf_side == click:
                    categories[f"trf_dominance_bucket_{dom_bucket}_match"].append(r)
                else:
                    categories[f"trf_dominance_bucket_{dom_bucket}_mismatch"].append(r)

            adaptive_bucket = get_adaptive_score_bucket(adaptive_score)
            if adaptive_bucket is not None:
                categories[f"trf_adaptive_score_bucket_{adaptive_bucket}"].append(r)
                if trf_side == click:
                    categories[f"trf_adaptive_score_bucket_{adaptive_bucket}_match"].append(r)
                else:
                    categories[f"trf_adaptive_score_bucket_{adaptive_bucket}_mismatch"].append(r)

    print("Interesting category counts")
    for name in sorted(categories):
        print(f"  {name}: {len(categories[name])} ({pct(len(categories[name]), max(len(ok_rows), 1))})")
    print("")

    # ------------------------------------------------------------
    # Optional seed lists
    # ------------------------------------------------------------
    if args.show_seeds:
        print("Example seed lists")
        print("-" * 80)
        for name in [
            "trf_valid",
            "trf_invalid",
            "trf_valid_and_match",
            "trf_valid_and_mismatch",
            "trf_matches_click",
            "trf_mismatches_click",
            "forward_matches_click",
            "forward_mismatches_click",
            "slit_matches_click",
            "slit_mismatches_click",
            "forward_vs_trf_same",
            "forward_vs_trf_different",
            "slit_vs_trf_same",
            "slit_vs_trf_different",
            "slit_vs_forward_same",
            "slit_vs_forward_different",
            "slit_vs_click_same",
            "slit_vs_click_different",
            "interesting_forward_trf_click_case",
            "trf_upper_and_click_upper",
            "trf_upper_but_click_lower",
            "trf_lower_and_click_lower",
            "trf_lower_but_click_upper",
            "slit_upper_and_click_upper",
            "slit_upper_but_click_lower",
            "slit_lower_and_click_lower",
            "slit_lower_but_click_upper",
            "transition_upper_slit->upper_click",
            "transition_upper_slit->lower_click",
            "transition_lower_slit->lower_click",
            "transition_lower_slit->upper_click",
            "trf_low_rel_margin",
            "trf_low_rel_margin_and_mismatch",
            "trf_high_ratio",
            "trf_high_ratio_and_match",
            "trf_high_dominance",
            "trf_high_dominance_and_match",
            "trf_high_adaptive_score",
            "trf_high_adaptive_score_and_match",
            "trf_rel_margin_bucket_<0.005",
            "trf_rel_margin_bucket_<0.005_match",
            "trf_rel_margin_bucket_<0.005_mismatch",
            "trf_rel_margin_bucket_0.005-0.01",
            "trf_rel_margin_bucket_0.005-0.01_match",
            "trf_rel_margin_bucket_0.005-0.01_mismatch",
            "trf_rel_margin_bucket_0.01-0.02",
            "trf_rel_margin_bucket_0.01-0.02_match",
            "trf_rel_margin_bucket_0.01-0.02_mismatch",
            "trf_rel_margin_bucket_>=0.02",
            "trf_rel_margin_bucket_>=0.02_match",
            "trf_rel_margin_bucket_>=0.02_mismatch",
            "trf_ratio_bucket_<1.01",
            "trf_ratio_bucket_<1.01_match",
            "trf_ratio_bucket_<1.01_mismatch",
            "trf_ratio_bucket_1.01-1.05",
            "trf_ratio_bucket_1.01-1.05_match",
            "trf_ratio_bucket_1.01-1.05_mismatch",
            "trf_ratio_bucket_1.05-1.20",
            "trf_ratio_bucket_1.05-1.20_match",
            "trf_ratio_bucket_1.05-1.20_mismatch",
            "trf_ratio_bucket_>=1.20",
            "trf_ratio_bucket_>=1.20_match",
            "trf_ratio_bucket_>=1.20_mismatch",
            "trf_dominance_bucket_<0.52",
            "trf_dominance_bucket_<0.52_match",
            "trf_dominance_bucket_<0.52_mismatch",
            "trf_dominance_bucket_0.52-0.55",
            "trf_dominance_bucket_0.52-0.55_match",
            "trf_dominance_bucket_0.52-0.55_mismatch",
            "trf_dominance_bucket_0.55-0.60",
            "trf_dominance_bucket_0.55-0.60_match",
            "trf_dominance_bucket_0.55-0.60_mismatch",
            "trf_dominance_bucket_>=0.60",
            "trf_dominance_bucket_>=0.60_match",
            "trf_dominance_bucket_>=0.60_mismatch",
            "trf_adaptive_score_bucket_<1e-8",
            "trf_adaptive_score_bucket_<1e-8_match",
            "trf_adaptive_score_bucket_<1e-8_mismatch",
            "trf_adaptive_score_bucket_1e-8-1e-6",
            "trf_adaptive_score_bucket_1e-8-1e-6_match",
            "trf_adaptive_score_bucket_1e-8-1e-6_mismatch",
            "trf_adaptive_score_bucket_1e-6-1e-4",
            "trf_adaptive_score_bucket_1e-6-1e-4_match",
            "trf_adaptive_score_bucket_1e-6-1e-4_mismatch",
            "trf_adaptive_score_bucket_>=1e-4",
            "trf_adaptive_score_bucket_>=1e-4_match",
            "trf_adaptive_score_bucket_>=1e-4_mismatch",
        ]:
            rows_cat = categories.get(name, [])
            print(f"{name}:")
            maybe_print_seed_list("  seeds", rows_cat, args.top_n)
        print("")

    # ------------------------------------------------------------
    # Failed rows
    # ------------------------------------------------------------
    if err_rows:
        print("Failed rows")
        for r in err_rows[:args.top_n]:
            print(
                f"  seed={r.get('seed')} "
                f"case={r.get('case_name')} "
                f"rc={r.get('returncode')} "
                f"error={r.get('error')}"
            )
        print("")

    # ------------------------------------------------------------
    # Hard cases (valid TRF but mismatch)
    # ------------------------------------------------------------
    hard_cases = [
        r for r in valid_rows
        if get_posthoc_side(r) != r.get("click_side")
    ]

    print("=" * 80)
    print("HARD CASE SEEDS (valid TRF but mismatch)")
    print("=" * 80)

    for r in hard_cases:
        rel = get_posthoc_rel_margin(r)
        dom = get_posthoc_dominance(r)
        ratio = get_posthoc_ratio(r)
        adaptive_score = get_posthoc_adaptive_score(r)
        ref_time = get_posthoc_ref_time(r)
        print(
            f"seed={r.get('seed')} "
            f"click={r.get('click_side')} "
            f"trf={get_posthoc_side(r)} "
            f"rel={rel:.6f} "
            f"dom={dom:.6f} "
            f"ratio={ratio:.6f} "
            f"adaptive={adaptive_score:.6e} "
            f"ref_t={ref_time:.6f}"
        )

    print("")

    hard_path = Path("hard_cases.json")
    with open(hard_path, "w", encoding="utf-8") as f:
        json.dump(hard_cases, f, indent=2)

    print(f"Saved hard cases to: {hard_path}")

    # ------------------------------------------------------------
    # Interesting cases (forward != TRF and TRF == click)
    # ------------------------------------------------------------
    print("=" * 80)
    print("INTERESTING CASE SEEDS (forward != TRF and TRF == click)")
    print("=" * 80)

    for r in interesting_rows:
        fwd_side = get_forward_guess_side(r)
        trf_side = get_posthoc_side(r)
        click = r.get("click_side")
        fwd_dom = get_forward_guess_dominance(r)
        trf_dom = get_posthoc_dominance(r)
        fwd_ref_t = get_forward_guess_ref_time(r)
        trf_ref_t = get_posthoc_ref_time(r)
        print(
            f"seed={r.get('seed')} "
            f"forward={fwd_side} "
            f"trf={trf_side} "
            f"click={click} "
            f"fwd_dom={fwd_dom:.6f} "
            f"trf_dom={trf_dom:.6f} "
            f"fwd_ref_t={fwd_ref_t:.6f} "
            f"trf_ref_t={trf_ref_t:.6f}"
        )

    print("")

    interesting_path = Path("interesting_cases.json")
    with open(interesting_path, "w", encoding="utf-8") as f:
        json.dump(interesting_rows, f, indent=2)

    print(f"Saved interesting cases to: {interesting_path}")

    # ------------------------------------------------------------
    # Slit mismatch cases
    # ------------------------------------------------------------
    slit_mismatch_cases = [
        r for r in slit_rows
        if get_slit_pass_side(r) != r.get("click_side")
    ]

    print("=" * 80)
    print("SLIT MISMATCH CASE SEEDS (slit pass != click)")
    print("=" * 80)

    for r in slit_mismatch_cases:
        slit_side = get_slit_pass_side(r)
        click = r.get("click_side")
        fwd_side = get_forward_guess_side(r)
        trf_side = get_posthoc_side(r)
        slit_dom = get_slit_pass_dominance(r)
        slit_ratio = get_slit_pass_ratio(r)
        slit_ref_t = get_slit_pass_ref_time(r)
        print(
            f"seed={r.get('seed')} "
            f"slit={slit_side} "
            f"click={click} "
            f"forward={fwd_side} "
            f"trf={trf_side} "
            f"slit_dom={slit_dom:.6f} "
            f"slit_ratio={slit_ratio:.6f} "
            f"slit_ref_t={slit_ref_t:.6f}"
        )

    print("")

    slit_mismatch_path = Path("slit_mismatch_cases.json")
    with open(slit_mismatch_path, "w", encoding="utf-8") as f:
        json.dump(slit_mismatch_cases, f, indent=2)

    print(f"Saved slit mismatch cases to: {slit_mismatch_path}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())