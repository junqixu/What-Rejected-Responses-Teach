from __future__ import annotations

import argparse
import json
import logging
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from rebuttal.common import (
    GENERATOR_VERSION,
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    load_records,
    sha256_file,
    sha256_text,
    stable_prompt_id,
    write_json,
    write_jsonl,
)
from rebuttal.identifiability.feature_schema import ERROR_FEATURES, canonical_error_name
from rebuttal.taxonomy.base import PerturbationResult, apply_perturbation, parse_trace, validate_perturbation


LOGGER = logging.getLogger(__name__)
REGIMES = ("confounded", "orthogonal_size", "orthogonal_exposure", "factorial", "single_a", "single_b")


def _last_number(text: str) -> str | None:
    values = re.findall(r"-?\d+(?:\.\d+)?", text)
    return values[-1] if values else None


def _token_edit_distance(left: str, right: str) -> int:
    matcher = SequenceMatcher(a=left.split(), b=right.split(), autojunk=False)
    equal = sum(block.size for block in matcher.get_matching_blocks())
    return max(len(left.split()), len(right.split())) - equal


def _error_vector(labels: list[str]) -> dict[str, int]:
    selected = set(labels)
    return {name: int(name in selected) for name in ERROR_FEATURES}


def _extract_clean(row: dict[str, Any]) -> str:
    for key in ("chosen", "solution", "gold_trace"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    raise ValueError("row has no chosen, solution, or gold_trace string")


def _make_record(
    row: dict[str, Any],
    rejected: str,
    labels: list[str],
    regime: str,
    results: list[PerturbationResult],
    seed: int,
) -> dict[str, Any]:
    chosen = _extract_clean(row)
    prompt_id = stable_prompt_id(row)
    before = parse_trace(chosen)
    after = parse_trace(rejected)
    changed_indices = [index for result in results for index in result.changed_step_indices]
    depth = min(changed_indices) / max(len(before.steps) - 1, 1) if changed_indices else None
    return {
        "id": f"{prompt_id}:{regime}:{'+'.join(labels)}",
        "prompt_id": prompt_id,
        "prompt": row.get("prompt") or row.get("problem") or "",
        "chosen": chosen,
        "rejected": rejected,
        "regime": regime,
        "error_labels": labels,
        "error_vector": _error_vector(labels),
        "op": row.get("op"),
        "dag_depth": row.get("dag_depth", row.get("op")),
        "error_node_depth": depth,
        "num_nodes_changed": abs(after.node_count - before.node_count),
        "num_edges_changed": abs(after.edge_count - before.edge_count),
        "chosen_tokens": len(chosen.split()),
        "rejected_tokens": len(rejected.split()),
        "token_edit_distance": _token_edit_distance(chosen, rejected),
        "final_answer_correct": _last_number(chosen) == _last_number(rejected),
        "seed": seed,
        "source_hash": sha256_text(chosen),
        "chosen_hash": sha256_text(chosen),
        "rejected_hash": sha256_text(rejected),
        "generator_version": GENERATOR_VERSION,
        "split": row.get("split", "train"),
        "dag_metadata": {
            "before_nodes": before.node_count,
            "before_edges": before.edge_count,
            "after_nodes": after.node_count,
            "after_edges": after.edge_count,
            "edits": [result.to_dict() for result in results],
        },
    }


def _variants(row: dict[str, Any], error_a: str, error_b: str) -> tuple[dict[str, tuple[str, list[PerturbationResult]]], list[str]]:
    clean = _extract_clean(row)
    failures: list[str] = []
    first_a = apply_perturbation(clean, error_a)
    first_b = apply_perturbation(clean, error_b)
    variants: dict[str, tuple[str, list[PerturbationResult]]] = {}
    for key, result in (("a", first_a), ("b", first_b)):
        if result is None:
            failures.append(f"{key}_not_applicable")
            continue
        valid, reasons = validate_perturbation(clean, result)
        if not valid:
            failures.extend(f"{key}_{reason}" for reason in reasons)
            continue
        variants[key] = (result.text, [result])

    ab_failure: str | None = None
    if first_a is not None:
        second_b = apply_perturbation(first_a.text, error_b)
        if second_b is not None:
            valid, reasons = validate_perturbation(first_a.text, second_b)
            if valid:
                variants["ab"] = (second_b.text, [first_a, second_b])
            else:
                ab_failure = ",".join(reasons)
        else:
            ab_failure = "second_edit_not_applicable"
    if "ab" not in variants and first_b is not None:
        second_a = apply_perturbation(first_b.text, error_a)
        if second_a is not None:
            valid, reasons = validate_perturbation(first_b.text, second_a)
            if valid:
                variants["ab"] = (second_a.text, [first_b, second_a])
            else:
                ab_failure = ",".join(reasons)
    if "ab" not in variants:
        failures.append(f"ab_{ab_failure or 'composition_not_applicable'}")
    return variants, failures


def generate_regimes(
    rows: list[dict[str, Any]], error_a: str, error_b: str, regimes: list[str], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(stable_prompt_id(row), row)
    ordered = list(unique.values())
    rng.shuffle(ordered)

    output = {regime: [] for regime in regimes}
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        variants, reasons = _variants(row, error_a, error_b)
        prompt_id = stable_prompt_id(row)
        if reasons:
            failures.append({"prompt_id": prompt_id, "reasons": reasons})
        complete = all(key in variants for key in ("a", "b", "ab"))
        if "confounded" in output and complete:
            text, edits = variants["ab"]
            output["confounded"].append(_make_record(row, text, [error_a, error_b], "confounded_ab", edits, seed))
        if "orthogonal_size" in output and complete:
            key = "a" if index % 2 == 0 else "b"
            if key in variants:
                label = error_a if key == "a" else error_b
                text, edits = variants[key]
                output["orthogonal_size"].append(_make_record(row, text, [label], "orthogonal_size", edits, seed))
        if "orthogonal_exposure" in output and complete:
            for key, label in (("a", error_a), ("b", error_b)):
                if key in variants:
                    text, edits = variants[key]
                    output["orthogonal_exposure"].append(
                        _make_record(row, text, [label], "orthogonal_exposure", edits, seed)
                    )
        if "factorial" in output and complete:
            selector = index % 5
            key = "a" if selector < 2 else "b" if selector < 4 else "ab"
            if key in variants:
                labels = [error_a] if key == "a" else [error_b] if key == "b" else [error_a, error_b]
                text, edits = variants[key]
                output["factorial"].append(_make_record(row, text, labels, "factorial", edits, seed))
        for regime, key, label in (("single_a", "a", error_a), ("single_b", "b", error_b)):
            if regime in output and key in variants:
                text, edits = variants[key]
                output[regime].append(_make_record(row, text, [label], regime, edits, seed))
    return output, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    error_a = canonical_error_name(args.error_a)
    error_b = canonical_error_name(args.error_b)
    if error_a == error_b:
        raise ValueError("error_a and error_b must differ")
    regimes = [value.strip() for value in args.regimes.split(",") if value.strip()]
    unknown = sorted(set(regimes) - set(REGIMES))
    if unknown:
        raise ValueError(f"unsupported regimes: {unknown}")
    rows = load_records(args.base_data, args.max_samples)
    generated, failures = generate_regimes(rows, error_a, error_b, regimes, args.seed)
    summary: dict[str, Any] = {
        "error_a": error_a,
        "error_b": error_b,
        "seed": args.seed,
        "generator_version": GENERATOR_VERSION,
        "source_files": [str(Path(path)) for path in args.base_data],
        "source_rows": len(rows),
        "output_rows": {name: len(values) for name, values in generated.items()},
        "failed_prompts": len(failures),
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

    output = ensure_output_dir(args.out, args.overwrite)
    output_hashes: dict[str, str] = {}
    for regime, values in generated.items():
        path = output / f"{regime}.jsonl"
        write_jsonl(path, values)
        output_hashes[path.name] = sha256_file(path)
    write_jsonl(output / "generation_failures.jsonl", failures)
    summary["output_hashes"] = output_hashes
    write_json(output / "dataset_manifest.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate confounded and orthogonal structural preference regimes.")
    parser.add_argument("--base_data", nargs="+", required=True)
    parser.add_argument("--error_a", required=True)
    parser.add_argument("--error_b", required=True)
    parser.add_argument("--regimes", default="confounded,orthogonal_size,orthogonal_exposure,factorial")
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
