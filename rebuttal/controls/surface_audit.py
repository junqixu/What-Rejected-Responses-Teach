from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, stable_prompt_id, write_json
from rebuttal.identifiability.feature_schema import canonical_error_name
from rebuttal.taxonomy.base import parse_trace


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
NUMERIC_FIELDS = (
    "prompt_tokens",
    "chosen_tokens",
    "rejected_tokens",
    "length_delta",
    "normalized_length_delta",
    "token_edit_distance",
    "step_count_delta",
    "numeric_delta",
    "error_node_depth",
    "num_nodes_changed",
    "num_edges_changed",
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _edit_distance(left: list[str], right: list[str]) -> int:
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    edits = 0
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "equal":
            edits += max(left_end - left_start, right_end - right_start)
    return edits


def _last_number(text: str) -> float | None:
    values = NUMBER.findall(text.replace(",", ""))
    return float(values[-1]) if values else None


def _label(row: dict[str, Any]) -> str:
    labels = row.get("error_labels")
    if isinstance(labels, list) and labels:
        return "+".join(sorted(canonical_error_name(str(value)) for value in labels))
    if row.get("error_type"):
        return canonical_error_name(str(row["error_type"]))
    return "unknown"


def audit_record(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt", ""))
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    prompt_tokens, chosen_tokens, rejected_tokens = _tokens(prompt), _tokens(chosen), _tokens(rejected)
    chosen_number, rejected_number = _last_number(chosen), _last_number(rejected)
    numeric_delta = None
    if chosen_number is not None and rejected_number is not None:
        numeric_delta = rejected_number - chosen_number
    return {
        "prompt_id": stable_prompt_id(row),
        "error_type": _label(row),
        "prompt_tokens": len(prompt_tokens),
        "chosen_tokens": len(chosen_tokens),
        "rejected_tokens": len(rejected_tokens),
        "length_delta": len(rejected_tokens) - len(chosen_tokens),
        "normalized_length_delta": (len(rejected_tokens) - len(chosen_tokens)) / max(len(chosen_tokens), 1),
        "token_edit_distance": _edit_distance(chosen_tokens, rejected_tokens),
        "step_count_delta": len(parse_trace(rejected).steps) - len(parse_trace(chosen).steps),
        "final_answer_correct": chosen_number is not None and chosen_number == rejected_number,
        "numeric_delta": numeric_delta,
        "error_node_depth": row.get("error_node_depth"),
        "num_nodes_changed": row.get("num_nodes_changed"),
        "num_edges_changed": row.get("num_edges_changed"),
        "op": row.get("op"),
        "dag_depth": row.get("dag_depth", row.get("d")),
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["error_type"])].append(row)
    output: list[dict[str, Any]] = []
    for error_type, group in sorted(groups.items()):
        for field in NUMERIC_FIELDS:
            values = [float(row[field]) for row in group if isinstance(row.get(field), (int, float))]
            if not values:
                continue
            deviation = pstdev(values) if len(values) > 1 else 0.0
            output.append(
                {
                    "error_type": error_type,
                    "metric": field,
                    "n": len(values),
                    "mean": mean(values),
                    "std": deviation,
                    "q25": _quantile(values, 0.25),
                    "median": median(values),
                    "q75": _quantile(values, 0.75),
                    "standardized_mean": mean(values) / deviation if deviation > 0 else None,
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = load_records(args.data, args.max_samples)
    audited = [audit_record(row) for row in source]
    summary_rows = summarize(audited)
    report = {
        "n_samples": len(audited),
        "error_types": sorted({row["error_type"] for row in audited}),
        "tokenizer": "unicode_word_punctuation_v1",
        "note": "Token counts are deterministic surface proxies, not model-tokenizer lengths.",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    output = ensure_output_dir(args.out, args.overwrite)
    _write_csv(output / "surface_records.csv", audited)
    _write_csv(output / "surface_statistics.csv", summary_rows)
    write_json(output / "surface_audit.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit surface-form confounds in preference data.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
