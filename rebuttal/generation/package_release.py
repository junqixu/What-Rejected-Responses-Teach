from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    iter_records,
    sha256_file,
    write_json,
    write_jsonl,
)
from rebuttal.identifiability.feature_schema import ERROR_FEATURES, canonical_error_name
from rebuttal.taxonomy.validate_perturbations import validate


RELEASE_VERSION = "rebuttal-eight-axis-v1"
SPLITS = ("train", "validation", "test")


def validate_complete_split(
    rows: Iterable[Mapping[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    packed = [dict(row) for row in rows]
    problems: list[str] = []
    expected = set(ERROR_FEATURES)
    prompt_types: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(packed):
        if str(row.get("split")) != split:
            problems.append(f"row {index} has split={row.get('split')!r}, expected {split!r}")
        prompt_id = str(row.get("prompt_id") or "")
        if not prompt_id:
            problems.append(f"row {index} has no prompt_id")
            continue
        try:
            error_type = canonical_error_name(str(row.get("error_type")))
        except ValueError as error:
            problems.append(f"row {index}: {error}")
            continue
        prompt_types[prompt_id][error_type] += 1
    for prompt_id, counts in sorted(prompt_types.items()):
        if set(counts) != expected:
            problems.append(
                f"prompt {prompt_id} has types {sorted(counts)}, expected {sorted(expected)}"
            )
        duplicates = sorted(name for name, count in counts.items() if count != 1)
        if duplicates:
            problems.append(f"prompt {prompt_id} has duplicate types {duplicates}")
    report, failures = validate(packed)
    if not report["valid"]:
        problems.append(f"strict structural validation failed for {len(failures)} records")
    return packed, problems


def _source_by_prompt(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_records(path):
        prompt_id = str(row.get("prompt_id") or "")
        if not prompt_id:
            raise ValueError(f"{path} contains a source row without prompt_id")
        if prompt_id in result:
            raise ValueError(f"{path} contains duplicate prompt_id {prompt_id}")
        result[prompt_id] = row
    return result


def _select_prompts(rows: list[dict[str, Any]], max_prompts: int | None) -> list[dict[str, Any]]:
    if max_prompts is None:
        return rows
    selected = sorted({str(row["prompt_id"]) for row in rows})[:max_prompts]
    selected_set = set(selected)
    return [row for row in rows if str(row["prompt_id"]) in selected_set]


def package(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    source_root = Path(args.source_root)
    packaged: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, Any] = {}
    all_problems: list[str] = []
    for split in SPLITS:
        data_path = dataset_root / split / "balanced" / "all_types.jsonl"
        source_path = source_root / f"{split}_clean.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        rows, problems = validate_complete_split(iter_records(data_path), split)
        rows = _select_prompts(rows, args.max_samples)
        sources = _source_by_prompt(source_path)
        enriched: list[dict[str, Any]] = []
        missing_sources: list[str] = []
        for row in rows:
            prompt_id = str(row["prompt_id"])
            source = sources.get(prompt_id)
            if source is None:
                missing_sources.append(prompt_id)
                continue
            row.update(
                {
                    "dag_family_id": source.get("dag_family_id") or source.get("source_id") or source.get("id"),
                    "source_id": source.get("source_id") or source.get("id"),
                    "release_version": RELEASE_VERSION,
                }
            )
            enriched.append(row)
        if missing_sources:
            problems.append(f"{len(missing_sources)} records have no clean source metadata")
        packaged[split] = sorted(
            enriched,
            key=lambda row: (str(row["prompt_id"]), str(row["error_type"])),
        )
        prompt_ids = {str(row["prompt_id"]) for row in enriched}
        families = {str(row["dag_family_id"]) for row in enriched}
        split_reports[split] = {
            "records": len(enriched),
            "prompts": len(prompt_ids),
            "dag_families": len(families),
            "error_type_counts": dict(sorted(Counter(str(row["error_type"]) for row in enriched).items())),
            "generator_versions": dict(
                sorted(Counter(str(row.get("generator_version", "unknown")) for row in enriched).items())
            ),
            "generator_models": dict(
                sorted(Counter(str(row.get("generator_model", "unknown")) for row in enriched).items())
            ),
            "generator_seeds": dict(
                sorted(Counter(str(row.get("seed", "unknown")) for row in enriched).items())
            ),
            "source_file": str(source_path),
            "source_file_hash": sha256_file(source_path),
            "input_file": str(data_path),
            "input_file_hash": sha256_file(data_path),
            "problems": problems,
        }
        minimum = {
            "train": args.min_train_prompts,
            "validation": args.min_validation_prompts,
            "test": args.min_test_prompts,
        }[split]
        if len(prompt_ids) < minimum:
            problems.append(f"only {len(prompt_ids)} complete prompts; minimum is {minimum}")
            split_reports[split]["problems"] = problems
        all_problems.extend(f"{split}: {problem}" for problem in problems)

    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            left_prompts = {str(row["prompt_id"]) for row in packaged[left]}
            right_prompts = {str(row["prompt_id"]) for row in packaged[right]}
            left_families = {str(row["dag_family_id"]) for row in packaged[left]}
            right_families = {str(row["dag_family_id"]) for row in packaged[right]}
            overlaps[f"{left}_{right}_prompts"] = len(left_prompts & right_prompts)
            overlaps[f"{left}_{right}_families"] = len(left_families & right_families)
    if any(overlaps.values()):
        all_problems.append(f"cross-split overlap detected: {overlaps}")
    manifest: dict[str, Any] = {
        "release_version": RELEASE_VERSION,
        "valid": not all_problems,
        "splits": split_reports,
        "cross_split_overlaps": overlaps,
        "problems": all_problems,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest
    if all_problems:
        raise ValueError("release validation failed:\n" + "\n".join(all_problems[:50]))
    output = ensure_output_dir(args.out, args.overwrite)
    output_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        write_jsonl(path, packaged[split])
        output_hashes[path.name] = sha256_file(path)
    manifest["output_hashes"] = output_hashes
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the compact, strict train/validation/test dataset committed to the release repository."
    )
    parser.add_argument("--dataset_root", default="data/rebuttal_v3/deepseek")
    parser.add_argument("--source_root", default="data/rebuttal_v2/source")
    parser.add_argument("--out", default="data/rebuttal_release")
    parser.add_argument("--min_train_prompts", type=int, default=400)
    parser.add_argument("--min_validation_prompts", type=int, default=80)
    parser.add_argument("--min_test_prompts", type=int, default=200)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    manifest = package(args)
    if not args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
