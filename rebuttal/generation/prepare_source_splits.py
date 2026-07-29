from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    iter_records,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)


SPLITS = ("train", "validation", "test")


def prompt_id(row: Mapping[str, Any]) -> str:
    problem = str(row.get("problem", "")).strip()
    question = str(row.get("question", "")).strip()
    return sha256_text(problem + "\n" + question)[:24]


def family_id(row: Mapping[str, Any]) -> str:
    value = row.get("dag_family_id", row.get("id"))
    return str(value) if value is not None else f"prompt:{prompt_id(row)}"


def assign_split(family: str, seed: int, train_ratio: float, validation_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}:{family}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def prepare(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if args.train_ratio <= 0 or args.validation_ratio <= 0 or args.train_ratio + args.validation_ratio >= 1:
        raise ValueError("ratios must be positive and leave a positive test ratio")
    targets = {
        "train": args.train_samples,
        "validation": args.validation_samples,
        "test": args.test_samples,
    }
    reservoirs: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    eligible_counts = {split: 0 for split in SPLITS}
    family_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    seen_prompts: set[str] = set()
    duplicate_prompts = 0
    rngs = {split: random.Random(args.seed + index * 1009) for index, split in enumerate(SPLITS)}

    for row in iter_records(args.input):
        try:
            if int(row.get("op")) != args.op:
                continue
        except (TypeError, ValueError):
            continue
        if not str(row.get("problem", "")).strip() or not str(row.get("solution", row.get("chosen", ""))).strip():
            continue
        current_prompt = prompt_id(row)
        if current_prompt in seen_prompts:
            duplicate_prompts += 1
            continue
        seen_prompts.add(current_prompt)
        current_family = family_id(row)
        split = assign_split(current_family, args.seed, args.train_ratio, args.validation_ratio)
        family_sets[split].add(current_family)
        eligible_counts[split] += 1
        packed = dict(row)
        packed["source_id"] = row.get("id")
        packed["prompt_id"] = current_prompt
        packed["dag_family_id"] = current_family
        packed["split"] = split

        target = targets[split]
        reservoir = reservoirs[split]
        if target is None or len(reservoir) < target:
            reservoir.append(packed)
        else:
            replacement = rngs[split].randrange(eligible_counts[split])
            if replacement < target:
                reservoir[replacement] = packed

    for split in SPLITS:
        reservoirs[split].sort(key=lambda row: str(row["prompt_id"]))
    prompt_sets = {split: {str(row["prompt_id"]) for row in reservoirs[split]} for split in SPLITS}
    selected_families = {split: {str(row["dag_family_id"]) for row in reservoirs[split]} for split in SPLITS}
    overlaps = {
        "train_validation_prompts": len(prompt_sets["train"] & prompt_sets["validation"]),
        "train_test_prompts": len(prompt_sets["train"] & prompt_sets["test"]),
        "validation_test_prompts": len(prompt_sets["validation"] & prompt_sets["test"]),
        "train_validation_families": len(selected_families["train"] & selected_families["validation"]),
        "train_test_families": len(selected_families["train"] & selected_families["test"]),
        "validation_test_families": len(selected_families["validation"] & selected_families["test"]),
    }
    report = {
        "source_files": args.input,
        "op_filter": args.op,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": 1.0 - args.train_ratio - args.validation_ratio,
        },
        "eligible_prompt_counts": eligible_counts,
        "eligible_family_counts": {split: len(family_sets[split]) for split in SPLITS},
        "selected_prompt_counts": {split: len(reservoirs[split]) for split in SPLITS},
        "selected_family_counts": {split: len(selected_families[split]) for split in SPLITS},
        "duplicate_prompt_texts_skipped": duplicate_prompts,
        "overlaps": overlaps,
        "valid_no_leakage": all(value == 0 for value in overlaps.values()),
    }
    return reservoirs, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    splits, report = prepare(args)
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return report
    output = ensure_output_dir(args.out, args.overwrite)
    hashes: dict[str, str] = {}
    for split in SPLITS:
        path = output / f"{split}_clean.jsonl"
        write_jsonl(path, splits[split])
        hashes[path.name] = sha256_file(path)
    report["output_hashes"] = hashes
    write_json(output / "source_split_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare prompt- and DAG-family-disjoint clean OP source pools.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--op", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--validation_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=500)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
