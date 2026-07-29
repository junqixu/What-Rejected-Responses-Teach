from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rebuttal.common import add_common_cli_args, configure_logging, parse_bool, write_json
from rebuttal.generation.deepseek_eight_types import (
    _append,
    _finish,
    _load_done,
    _record,
    _select_records,
    _source_prompt_id,
    validate_candidate,
)
from rebuttal.identifiability.feature_schema import ERROR_FEATURES, canonical_error_name
from rebuttal.taxonomy.base import apply_perturbation


DETERMINISTIC_GENERATOR_VERSION = "deterministic-graph-edit-v3"


def run(args: argparse.Namespace) -> dict[str, object]:
    error_types = [canonical_error_name(value) for value in args.error_types.split(",") if value.strip()]
    if len(error_types) != len(set(error_types)):
        raise ValueError("error_types contains duplicates")
    rows, eligible, duplicates = _select_records(args.input, args.max_samples, args.seed, args.op)
    report: dict[str, object] = {
        "generator_version": DETERMINISTIC_GENERATOR_VERSION,
        "eligible_source_rows": eligible,
        "duplicate_source_prompts_skipped": duplicates,
        "selected_rows": len(rows),
        "requested_error_types": error_types,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError("output is not empty; use --resume true or a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    done = _load_done(output, error_types) if args.resume else {name: set() for name in error_types}
    generated = Counter()
    failures = Counter()
    for source in rows:
        chosen = str(source.get("solution") or source.get("chosen") or source.get("gold_trace") or "")
        prompt_id = _source_prompt_id(source)
        for error_type in error_types:
            if prompt_id in done[error_type]:
                continue
            result = apply_perturbation(chosen, error_type)
            if result is None:
                failures[f"{error_type}:inapplicable"] += 1
                _append(
                    output / "deterministic_failures.jsonl",
                    {"prompt_id": prompt_id, "error_type": error_type, "reasons": ["inapplicable"]},
                )
                continue
            valid, reasons = validate_candidate(error_type, chosen, result.text)
            if not valid:
                failures.update(f"{error_type}:{reason}" for reason in reasons)
                _append(
                    output / "deterministic_failures.jsonl",
                    {"prompt_id": prompt_id, "error_type": error_type, "reasons": reasons},
                )
                continue
            packed = _record(
                source,
                error_type,
                result.text,
                json.dumps(result.metadata, ensure_ascii=False, sort_keys=True),
                args,
            )
            packed["generator_version"] = DETERMINISTIC_GENERATOR_VERSION
            packed["generator_model"] = "deterministic"
            _append(output / f"{error_type}.jsonl", packed)
            done[error_type].add(prompt_id)
            generated[error_type] += 1
    manifest = _finish(output, list(ERROR_FEATURES), args.input, args, Counter())
    manifest.update(
        {
            "deterministic_generator_version": DETERMINISTIC_GENERATOR_VERSION,
            "deterministic_records_this_run": dict(generated),
            "deterministic_failures_this_run": dict(failures),
        }
    )
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill structurally valid preference records with deterministic graph edits."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--error_types", default=",".join(ERROR_FEATURES))
    parser.add_argument("--op", type=int, default=10)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--model", default="deterministic")
    parser.add_argument("--base_url", default="offline")
    parser.add_argument("--resume", type=parse_bool, default=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    report = run(args)
    if not args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
