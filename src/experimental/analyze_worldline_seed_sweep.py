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


def get_posthoc_side(row: dict):
    return row.get("posthoc_trf_chosen_side", row.get("posthoc_chosen_side"))


def get_posthoc_valid(row: dict):
    x = to_bool_or_none(row.get("posthoc_trf_valid"))
    if x is not None:
        return x

    # fallback heuristic for old rows:
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


def get_posthoc_ref_time(row: dict):
    return to_float_or_none(row.get("posthoc_trf_ref_time"))


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
    print("Posthoc / TRF seed sweep analysis")
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

    print_counter("Click side distribution", click_counter, ok_total)
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

    print_counter("Posthoc TRF -> click (all rows)", trf_to_click, max(len(trf_rows), 1))
    print("")
    print_counter("Posthoc TRF -> click (valid rows)", valid_to_click, max(len(valid_rows), 1))
    print("")
    if invalid_rows:
        print_counter("Posthoc TRF -> click (invalid rows)", invalid_to_click, len(invalid_rows))
        print("")

    # ------------------------------------------------------------
    # Evidence stats: valid rows only
    # ------------------------------------------------------------
    rel_margins = [get_posthoc_rel_margin(r) for r in valid_rows]
    rel_margins = [x for x in rel_margins if x is not None]

    ratios = [get_posthoc_ratio(r) for r in valid_rows]
    ratios = [x for x in ratios if x is not None]

    dominances = [get_posthoc_dominance(r) for r in valid_rows]
    dominances = [x for x in dominances if x is not None]

    ref_times = [get_posthoc_ref_time(r) for r in valid_rows]
    ref_times = [x for x in ref_times if x is not None]

    if rel_margins or ratios or dominances or ref_times:
        print("Posthoc TRF evidence stats (valid rows only)")
        if rel_margins:
            print(f"  rel margin min/avg/median/max : {min(rel_margins):.6f} / {mean_of(rel_margins):.6f} / {median_of(rel_margins):.6f} / {max(rel_margins):.6f}")
        if ratios:
            print(f"  ratio min/avg/median/max      : {min(ratios):.6f} / {mean_of(ratios):.6f} / {median_of(ratios):.6f} / {max(ratios):.6f}")
        if dominances:
            print(f"  dominance min/avg/median/max  : {min(dominances):.6f} / {mean_of(dominances):.6f} / {median_of(dominances):.6f} / {max(dominances):.6f}")
        if ref_times:
            print(f"  ref time min/avg/median/max   : {min(ref_times):.6f} / {mean_of(ref_times):.6f} / {median_of(ref_times):.6f} / {max(ref_times):.6f}")
        print("")

    # ------------------------------------------------------------
    # Invalid stats
    # ------------------------------------------------------------
    if invalid_rows:
        invalid_rel = [get_posthoc_rel_margin(r) for r in invalid_rows]
        invalid_rel = [x for x in invalid_rel if x is not None]
        invalid_ratio = [get_posthoc_ratio(r) for r in invalid_rows]
        invalid_ratio = [x for x in invalid_ratio if x is not None]
        invalid_dom = [get_posthoc_dominance(r) for r in invalid_rows]
        invalid_dom = [x for x in invalid_dom if x is not None]

        print("Invalid TRF evidence stats")
        if invalid_rel:
            print(f"  rel margin min/avg/median/max : {min(invalid_rel):.6f} / {mean_of(invalid_rel):.6f} / {median_of(invalid_rel):.6f} / {max(invalid_rel):.6f}")
        if invalid_ratio:
            print(f"  ratio min/avg/median/max      : {min(invalid_ratio):.6f} / {mean_of(invalid_ratio):.6f} / {median_of(invalid_ratio):.6f} / {max(invalid_ratio):.6f}")
        if invalid_dom:
            print(f"  dominance min/avg/median/max  : {min(invalid_dom):.6f} / {mean_of(invalid_dom):.6f} / {median_of(invalid_dom):.6f} / {max(invalid_dom):.6f}")
        print("")

    # ------------------------------------------------------------
    # Buckets: valid rows only
    # ------------------------------------------------------------
    rel_bucket_counter = Counter()
    rel_bucket_to_click = Counter()

    ratio_bucket_counter = Counter()
    ratio_bucket_to_click = Counter()

    dom_bucket_counter = Counter()
    dom_bucket_to_click = Counter()

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

    for r in trf_rows:
        click = r.get("click_side")
        trf_side = get_posthoc_side(r)
        valid = get_posthoc_valid(r)
        rel_margin = get_posthoc_rel_margin(r)
        ratio = get_posthoc_ratio(r)
        dominance = get_posthoc_dominance(r)

        if valid is True:
            categories["trf_valid"].append(r)
        elif valid is False:
            categories["trf_invalid"].append(r)
        else:
            categories["trf_unknown_validity"].append(r)

        if trf_side == click:
            categories["trf_matches_click"].append(r)
        else:
            categories["trf_mismatches_click"].append(r)

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

    print("Interesting category counts")
    for name in sorted(categories):
        print(f"  {name}: {len(categories[name])} ({pct(len(categories[name]), max(len(trf_rows), 1))})")
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
            "trf_invalid_and_match",
            "trf_invalid_and_mismatch",
            "trf_matches_click",
            "trf_mismatches_click",
            "trf_upper_and_click_upper",
            "trf_upper_but_click_lower",
            "trf_lower_and_click_lower",
            "trf_lower_but_click_upper",
            "trf_low_rel_margin",
            "trf_low_rel_margin_and_mismatch",
            "trf_high_ratio",
            "trf_high_ratio_and_match",
            "trf_high_dominance",
            "trf_high_dominance_and_match",
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

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())