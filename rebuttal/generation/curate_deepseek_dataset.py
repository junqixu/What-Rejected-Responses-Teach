from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    iter_records,
    prompt_split_leakage,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from rebuttal.generation.deepseek_eight_types import GENERATOR_VERSION, validate_candidate
from rebuttal.identifiability.feature_schema import ERROR_FEATURES, canonical_error_name


def curate_type_rows(
    rows: Iterable[Mapping[str, Any]],
    expected_type: str,
    expected_split: str | None = None,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    expected_type = canonical_error_name(expected_type)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reasons_total: Counter[str] = Counter()
    seen_prompt_ids: set[str] = set()
    for index, source in enumerate(rows):
        if max_samples is not None and index >= max_samples:
            break
        row = dict(source)
        reasons: list[str] = []
        prompt_id = str(row.get("prompt_id") or "")
        chosen = row.get("chosen")
        rejected_text = row.get("rejected")
        try:
            row_type = canonical_error_name(str(row.get("error_type") or ""))
        except ValueError:
            row_type = ""
            reasons.append("invalid_error_type")
        if row_type and row_type != expected_type:
            reasons.append("wrong_error_type_file")
        if not prompt_id:
            reasons.append("missing_prompt_id")
        elif prompt_id in seen_prompt_ids:
            reasons.append("duplicate_prompt_id")
        if expected_split is not None and str(row.get("split")) != expected_split:
            reasons.append("wrong_split")
        if not isinstance(chosen, str) or not isinstance(rejected_text, str):
            reasons.append("missing_pair_text")
        else:
            if row.get("chosen_hash") != sha256_text(chosen):
                reasons.append("invalid_chosen_hash")
            if row.get("rejected_hash") != sha256_text(rejected_text):
                reasons.append("invalid_rejected_hash")
            valid, structural_reasons = validate_candidate(expected_type, chosen, rejected_text)
            if not valid:
                reasons.extend(structural_reasons)
        if reasons:
            unique_reasons = sorted(set(reasons))
            reasons_total.update(unique_reasons)
            rejected.append(
                {
                    "index": index,
                    "id": row.get("id"),
                    "prompt_id": prompt_id or None,
                    "error_type": expected_type,
                    "reasons": unique_reasons,
                    "source_generator_version": row.get("generator_version"),
                }
            )
            continue
        seen_prompt_ids.add(prompt_id)
        row["validation_version"] = GENERATOR_VERSION
        kept.append(row)
    kept.sort(key=lambda item: str(item.get("prompt_id")))
    return kept, rejected, reasons_total


def curate(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_dir)
    error_types = [canonical_error_name(value) for value in args.error_types.split(",") if value.strip()]
    if len(error_types) != len(set(error_types)):
        raise ValueError("error_types contains duplicates")
    by_type: dict[str, list[dict[str, Any]]] = {}
    all_rejections: list[dict[str, Any]] = []
    rejection_reasons: Counter[str] = Counter()
    input_counts: dict[str, int] = {}
    for error_type in error_types:
        path = source / f"{error_type}.jsonl"
        rows = list(iter_records(path)) if path.exists() else []
        input_counts[error_type] = len(rows)
        kept, rejected, reasons = curate_type_rows(
            rows,
            error_type,
            expected_split=args.split,
            max_samples=args.max_samples,
        )
        by_type[error_type] = kept
        all_rejections.extend(rejected)
        rejection_reasons.update(reasons)

    complete_prompt_ids = (
        set.intersection(*[{str(row["prompt_id"]) for row in by_type[name]} for name in error_types])
        if error_types
        else set()
    )
    all_rows = sorted(
        (row for rows in by_type.values() for row in rows),
        key=lambda row: (str(row.get("prompt_id")), str(row.get("error_type"))),
    )
    report = {
        "curator_version": GENERATOR_VERSION,
        "source_dir": str(source),
        "split": args.split,
        "error_types": error_types,
        "input_counts": input_counts,
        "kept_counts": {name: len(by_type[name]) for name in error_types},
        "rejected_count": len(all_rejections),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "complete_prompt_count": len(complete_prompt_ids),
        "balanced_record_count": len(complete_prompt_ids) * len(error_types),
        "prompt_split_leakage": prompt_split_leakage(all_rows),
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report

    output = ensure_output_dir(args.out, args.overwrite)
    output_hashes: dict[str, str] = {}
    for error_type in error_types:
        path = output / f"{error_type}.jsonl"
        write_jsonl(path, by_type[error_type])
        output_hashes[path.name] = sha256_file(path)
    write_jsonl(output / "all_types.jsonl", all_rows)
    output_hashes["all_types.jsonl"] = sha256_file(output / "all_types.jsonl")
    balanced = output / "balanced"
    balanced.mkdir(parents=True, exist_ok=True)
    balanced_rows: list[dict[str, Any]] = []
    for error_type in error_types:
        rows = [row for row in by_type[error_type] if str(row["prompt_id"]) in complete_prompt_ids]
        path = balanced / f"{error_type}.jsonl"
        write_jsonl(path, rows)
        output_hashes[str(path.relative_to(output))] = sha256_file(path)
        balanced_rows.extend(rows)
    balanced_rows.sort(key=lambda row: (str(row.get("prompt_id")), str(row.get("error_type"))))
    write_jsonl(balanced / "all_types.jsonl", balanced_rows)
    output_hashes[str((balanced / "all_types.jsonl").relative_to(output))] = sha256_file(
        balanced / "all_types.jsonl"
    )
    write_jsonl(output / "curation_rejections.jsonl", all_rejections)
    output_hashes["curation_rejections.jsonl"] = sha256_file(output / "curation_rejections.jsonl")
    source_manifest = source / "dataset_manifest.json"
    if source_manifest.exists():
        report["source_manifest_hash"] = sha256_file(source_manifest)
    report["output_hashes"] = output_hashes
    write_json(output / "curation_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revalidate and version an existing DeepSeek dataset without overwriting it."
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--error_types", default=",".join(ERROR_FEATURES))
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    report = curate(args)
    if not args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
