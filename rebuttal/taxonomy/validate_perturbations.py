from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, prompt_split_leakage, sha256_text, write_json, write_jsonl
from rebuttal.generation.deepseek_eight_types import validate_candidate
from rebuttal.identifiability.feature_schema import canonical_error_name
from rebuttal.taxonomy.base import parse_trace


def validate(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    prompt_types: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        reasons: list[str] = []
        chosen, rejected = row.get("chosen"), row.get("rejected")
        record_id = str(row.get("id") or "")
        if record_id:
            if record_id in seen_ids:
                reasons.append("duplicate_record_id")
            seen_ids.add(record_id)
        if not isinstance(chosen, str) or not chosen.strip():
            reasons.append("empty_chosen")
        if not isinstance(rejected, str) or not rejected.strip():
            reasons.append("empty_rejected")
        if isinstance(chosen, str) and isinstance(rejected, str) and chosen.strip() == rejected.strip():
            reasons.append("identical_pair")
        labels = row.get("error_labels") or ([row["error_type"]] if row.get("error_type") else [])
        try:
            canonical = [canonical_error_name(str(label)) for label in labels]
            counts.update(canonical)
            if len(canonical) == 1 and isinstance(chosen, str) and isinstance(rejected, str):
                structural_valid, structural_reasons = validate_candidate(canonical[0], chosen, rejected)
                if not structural_valid:
                    reasons.extend(f"structural:{reason}" for reason in structural_reasons)
                prompt_id = str(row.get("prompt_id") or row.get("prompt") or index)
                prompt_types.setdefault(prompt_id, set()).add(canonical[0])
        except ValueError as error:
            reasons.append(str(error))
        if isinstance(chosen, str) and not parse_trace(chosen).steps:
            reasons.append("unparseable_chosen")
        if isinstance(rejected, str) and not parse_trace(rejected).steps:
            reasons.append("unparseable_rejected")
        for field, text in (("chosen_hash", chosen), ("rejected_hash", rejected)):
            if row.get(field) and isinstance(text, str) and row[field] != sha256_text(text):
                reasons.append(f"invalid_{field}")
        if reasons:
            failures.append({"index": index, "id": row.get("id"), "reasons": reasons})
    leakage = prompt_split_leakage(rows)
    expected_types = set(counts)
    complete_prompt_count = sum(1 for types in prompt_types.values() if types == expected_types)
    return {
        "n_samples": len(rows),
        "n_failed": len(failures),
        "n_prompts": len(prompt_types),
        "complete_prompt_count": complete_prompt_count,
        "error_exposures": dict(sorted(counts.items())),
        "prompt_split_leakage": leakage,
        "valid": not failures and not leakage,
    }, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated structural preference records.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    rows = load_records(args.data, args.max_samples)
    report, failures = validate(rows)
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    output = ensure_output_dir(args.out, args.overwrite)
    write_json(output / "validation_report.json", report)
    write_jsonl(output / "validation_failures.jsonl", failures)


if __name__ == "__main__":
    main()
