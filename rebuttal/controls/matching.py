from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, write_json, write_jsonl
from rebuttal.controls.surface_audit import audit_record
from rebuttal.identifiability.feature_schema import canonical_error_name


MATCH_FIELDS = ("rejected_tokens", "token_edit_distance", "error_node_depth")
BALANCE_FIELDS = ("op", "dag_depth", "rejected_tokens", "token_edit_distance", "error_node_depth")


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _distance(left: dict[str, Any], right: dict[str, Any], scales: dict[str, float]) -> float:
    total = 0.0
    for field in MATCH_FIELDS:
        a, b = _number(left.get(field)), _number(right.get(field))
        if a is None or b is None:
            total += 1.0
        else:
            total += abs(a - b) / scales[field]
    return total


def _smd(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left_std = pstdev(left) if len(left) > 1 else 0.0
    right_std = pstdev(right) if len(right) > 1 else 0.0
    pooled = math.sqrt((left_std**2 + right_std**2) / 2)
    return (mean(left) - mean(right)) / pooled if pooled else 0.0


def match_rows(
    source: list[dict[str, Any]], error_a: str, error_b: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decorated = [(row, audit_record(row)) for row in source]
    selected = [(row, meta) for row, meta in decorated if meta["error_type"] in {error_a, error_b}]
    scales: dict[str, float] = {}
    for field in MATCH_FIELDS:
        values = [_number(meta.get(field)) for _row, meta in selected]
        present = [value for value in values if value is not None]
        scales[field] = pstdev(present) if len(present) > 1 and pstdev(present) > 0 else 1.0

    buckets: dict[tuple[Any, ...], dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = defaultdict(
        lambda: {error_a: [], error_b: []}
    )
    for row, meta in selected:
        key = (meta.get("op"), meta.get("dag_depth"), meta.get("final_answer_correct"))
        buckets[key][str(meta["error_type"])].append((row, meta))

    matches: list[dict[str, Any]] = []
    paired_meta: dict[str, list[dict[str, Any]]] = {error_a: [], error_b: []}
    match_index = 0
    for key in sorted(buckets, key=str):
        left = buckets[key][error_a]
        available = list(buckets[key][error_b])
        for left_row, left_meta in left:
            if not available:
                break
            best_index = min(range(len(available)), key=lambda index: _distance(left_meta, available[index][1], scales))
            right_row, right_meta = available.pop(best_index)
            match_id = f"match-{match_index:06d}"
            match_index += 1
            for role, row, meta in (("a", left_row, left_meta), ("b", right_row, right_meta)):
                packed = dict(row)
                packed.update({"match_id": match_id, "match_role": role, "matching_bucket": list(key)})
                matches.append(packed)
                paired_meta[error_a if role == "a" else error_b].append(meta)

    balance: list[dict[str, Any]] = []
    for field in BALANCE_FIELDS:
        left_values = [_number(row.get(field)) for row in paired_meta[error_a]]
        right_values = [_number(row.get(field)) for row in paired_meta[error_b]]
        left = [value for value in left_values if value is not None]
        right = [value for value in right_values if value is not None]
        value = _smd(left, right)
        balance.append(
            {
                "metric": field,
                "mean_a": mean(left) if left else None,
                "mean_b": mean(right) if right else None,
                "standardized_mean_difference": value,
                "balanced_below_0_1": value is not None and abs(value) < 0.1,
            }
        )
    return matches, balance


def run(args: argparse.Namespace) -> dict[str, Any]:
    error_a, error_b = canonical_error_name(args.error_a), canonical_error_name(args.error_b)
    source = load_records(args.data, args.max_samples)
    matches, balance = match_rows(source, error_a, error_b)
    report = {
        "error_a": error_a,
        "error_b": error_b,
        "n_source": len(source),
        "n_pairs": len(matches) // 2,
        "all_balance_below_0_1": all(row["balanced_below_0_1"] for row in balance if row["standardized_mean_difference"] is not None),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    output = ensure_output_dir(args.out, args.overwrite)
    write_jsonl(output / "matched_pairs.jsonl", matches)
    with (output / "matching_balance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balance[0]) if balance else [])
        if balance:
            writer.writeheader()
            writer.writerows(balance)
    write_json(output / "matching_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact-bucket plus nearest-neighbor preference matching.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--error_a", required=True)
    parser.add_argument("--error_b", required=True)
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
