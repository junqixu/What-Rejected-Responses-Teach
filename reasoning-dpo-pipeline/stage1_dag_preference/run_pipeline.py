#!/usr/bin/env python3
"""
Stage 1: Main Preference Data Generation Pipeline
================================================
End-to-end pipeline that:
  1. Loads raw math-CoT solutions (.json / .jsonl)
  2. Parses each solution into a ReasoningDAG
  3. Applies four corruption strategies (computer_error, dependency_mismatch,
     missing_nodes, disorder) to produce (chosen, rejected) pairs
  4. Runs multi-stage quality filtering (schema, similarity, length, signal)
  5. Writes per-type and mixed preference datasets + a QC report

Usage::

    python -m stage1_dag_preference.run_pipeline \
        --input_path data/raw_solutions.json \
        --output_dir ./outputs/dag_preference \
        --num_per_type 1000 \
        --seed 42 \
        --error_types computer_error,dependency_mismatch,missing_nodes,disorder \
        --min_chosen_len 50 \
        --max_rejected_chosen_ratio 0.98

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import json
import copy
import random
import hashlib
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

from dag_parser import (
    DAGParser,
    DAGCorruptor,
    ReasoningDAG,
    load_json_or_jsonl,
    save_jsonl,
)
from dag_corruptor import CORRUPTION_TYPES


# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------

@dataclass
class QCConfig:
    min_chosen_len: int = 50
    min_rejected_len: int = 30
    max_rejected_chosen_ratio: float = 0.98  # missing_nodes must be shorter
    min_rejected_chosen_ratio: float = 0.40  # others must not be too short
    min_similarity: float = 0.35
    max_similarity: float = 0.995
    bad_markers: Tuple[str, ...] = (
        "here is the json",
        "computation_error",
        "dependency_mismatch",
        "missing_nodes",
        "error type",
        "rejected solution",
        "chosen solution",
    )


@dataclass
class QCMetrics:
    total_processed: int = 0
    schema_failed: int = 0
    identical: int = 0
    too_short: int = 0
    length_ratio_fail: int = 0
    similarity_fail: int = 0
    corruption_failed: int = 0
    bad_marker: int = 0
    passed: int = 0
    by_error_type: Dict[str, int] = field(default_factory=dict)


def safe_strip(x) -> str:
    return "" if x is None else str(x).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def check_similarity(chosen: str, rejected: str, error_type: str, cfg: QCConfig) -> Tuple[bool, str]:
    """Check string-level quality filters."""
    chosen = safe_strip(chosen)
    rejected = safe_strip(rejected)

    if not rejected:
        return False, "empty_rejected"
    if rejected == chosen:
        return False, "identical"
    if len(rejected) < cfg.min_rejected_len:
        return False, "too_short"
    if len(chosen) < cfg.min_chosen_len:
        return False, "chosen_too_short"

    sim = similarity(chosen, rejected)
    if sim > cfg.max_similarity:
        return False, "too_similar"
    if sim < cfg.min_similarity:
        return False, "too_different"

    ratio = len(rejected) / max(len(chosen), 1)

    if error_type == "missing_nodes":
        if ratio > cfg.max_rejected_chosen_ratio:
            return False, "missing_nodes_not_shorter"
    else:
        if ratio < cfg.min_rejected_chosen_ratio:
            return False, f"{error_type}_too_short"

    lower = rejected.lower()
    for marker in cfg.bad_markers:
        if marker in lower:
            return False, "bad_marker"

    return True, "ok"


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------

def build_prompt_from_record(record: dict) -> str:
    """
    Format the problem/question as a ChatML prompt for Qwen2.
    """
    problem = safe_strip(record.get("problem", ""))
    question = safe_strip(record.get("question", ""))

    if question:
        user_content = f"Problem:\n{problem}\n\nQuestion:\n{question}"
    else:
        user_content = f"Problem:\n{problem}"

    return (
        "<|im_start|>user\n"
        f"{user_content}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def stable_item_id(record: dict) -> str:
    if record.get("id") and str(record["id"]).strip():
        return str(record["id"]).strip()
    base = "\n".join([
        str(record.get("problem", "")).strip(),
        str(record.get("question", "")).strip(),
        str(record.get("solution", "")).strip(),
    ])
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def construct_one_sample(
    record: dict,
    error_type: str,
    parser: DAGParser,
    qc_cfg: QCConfig,
    pair_idx: int,
) -> Tuple[Optional[dict], str]:
    """
    Parse a solution, apply one corruption strategy, and quality-filter.

    Returns
    -------
    sample : dict or None
        The DPO sample, or None if it fails QC.
    reason : str
        Human-readable discard reason.
    """
    solution = safe_strip(record.get("solution", ""))
    if not solution:
        return None, "no_solution"

    # Parse
    try:
        dag = parser.parse(solution)
    except Exception:
        return None, "parse_failed"

    # Corrupt
    corruptor = DAGCorruptor(dag)
    rejected = corruptor.corrupt(error_type)
    if rejected is None:
        return None, "corruption_failed"

    rejected = safe_strip(rejected)
    chosen = solution.strip()

    if rejected.strip() == chosen.strip():
        return None, "identical_after_strip"

    # Quality checks
    ok, reason = check_similarity(chosen, rejected, error_type, qc_cfg)
    if not ok:
        return None, reason

    raw_id = str(record.get("id", f"unknown_{pair_idx}"))

    sample = {
        "pair_id": f"{raw_id}__{error_type}__{pair_idx}",
        "id": raw_id,
        "template": record.get("template"),
        "op": record.get("op"),
        "mode": record.get("mode"),
        "length": record.get("length"),
        "d": record.get("d"),
        "error_type": error_type,
        "prompt": build_prompt_from_record(record),
        "chosen": chosen,
        "rejected": rejected,
        "source_problem": safe_strip(record.get("problem")),
        "source_question": safe_strip(record.get("question")),
        "dag": dag.to_dict(),
    }
    return sample, "ok"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    input_data: List[dict],
    error_types: List[str],
    num_per_type: int,
    qc_cfg: QCConfig,
    seed: int = 42,
) -> Tuple[List[dict], Dict[str, int], Counter]:
    """
    Build a DAG-based preference dataset.

    Returns
    -------
    rows : list of DPO samples
    counts : {error_type: n_passed}
    failures : Counter[error_type: {reason: n}]
    """
    random.seed(seed)
    parser = DAGParser()

    rows: List[dict] = []
    counts: Dict[str, int] = {e: 0 for e in error_types}
    failures: Counter = Counter()
    pair_idx = 0

    # Shuffle once for fair sampling
    data = copy.deepcopy(input_data)
    random.shuffle(data)

    for error_type in error_types:
        seen_ids: set = set()
        for record in data:
            if counts[error_type] >= num_per_type:
                break

            record_id = stable_item_id(record)
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)

            sample, reason = construct_one_sample(
                record=record,
                error_type=error_type,
                parser=parser,
                qc_cfg=qc_cfg,
                pair_idx=pair_idx,
            )

            if sample is None:
                failures[f"{error_type}:{reason}"] += 1
                continue

            rows.append(sample)
            counts[error_type] += 1
            pair_idx += 1

    return rows, counts, failures


def print_sample_diffs(rows: List[dict], n: int = 3) -> None:
    """Print side-by-side chosen/rejected diffs for inspection."""
    import difflib

    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["error_type"], []).append(r)

    print("\n" + "=" * 70)
    print("SAMPLE DIFFS")
    print("=" * 70)

    for error_type, group in by_type.items():
        print(f"\n### {error_type}")
        for r in group[:n]:
            chosen = r["chosen"]
            rejected = r["rejected"]
            sm = SequenceMatcher(None, chosen, rejected)
            printed = False
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"  [{tag.upper()}] pair_id={r['pair_id']}")
                    print(f"  CHOSEN : {chosen[max(0,i1-50):i2+100]}")
                    print(f"  REJECT : {rejected[max(0,j1-50):j2+100]}")
                    printed = True
                    break
            if not printed:
                print(f"  (no visible diff) pair_id={r['pair_id']}")


# ---------------------------------------------------------------------------
# QC report
# ---------------------------------------------------------------------------

def generate_qc_report(
    rows: List[dict],
    counts: Dict[str, int],
    failures: Counter,
    output_path: str,
    args: argparse.Namespace,
) -> None:
    report: Dict = {
        "command": " ".join(["python", "-m", __name__]) + " " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items()
        ),
        "config": asdict(args) if hasattr(args, "__dict__") else vars(args),
        "counts": counts,
        "total_passed": len(rows),
        "failure_detail": dict(failures),
        "per_type_stats": {},
    }

    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["error_type"], []).append(r)

    for et, group in by_type.items():
        chosen_lens = [len(r["chosen"]) for r in group]
        rejected_lens = [len(r["rejected"]) for r in group]
        ratios = [len(r["rejected"]) / max(len(r["chosen"]), 1) for r in group]
        report["per_type_stats"][et] = {
            "n": len(group),
            "avg_chosen_len": round(sum(chosen_lens) / len(chosen_lens), 1),
            "avg_rejected_len": round(sum(rejected_lens) / len(rejected_lens), 1),
            "avg_ratio": round(sum(ratios) / len(ratios), 3),
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 — DAG-based DPO preference data generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_path", required=True, help=".json or .jsonl source data")
    p.add_argument("--output_dir", required=True, help="Output directory")
    p.add_argument("--error_types", default="computer_error,dependency_mismatch,missing_nodes,disorder")
    p.add_argument("--num_per_type", type=int, default=500, help="Target samples per error type")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_chosen_len", type=int, default=50)
    p.add_argument("--min_rejected_len", type=int, default=30)
    p.add_argument("--max_rejected_chosen_ratio", type=float, default=0.98)
    p.add_argument("--min_rejected_chosen_ratio", type=float, default=0.40)
    p.add_argument("--print_sample_diffs", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 1 — DAG Preference Data Generation")
    print(f"  Input  : {args.input_path}")
    print(f"  Output : {args.output_dir}")
    print(f"  Error types: {args.error_types}")
    print(f"  Per type : {args.num_per_type}")
    print("=" * 60)

    # Load data
    data = load_json_or_jsonl(args.input_path)
    print(f"Loaded {len(data)} raw records.")

    error_types = [e.strip() for e in args.error_types.split(",") if e.strip()]

    qc_cfg = QCConfig(
        min_chosen_len=args.min_chosen_len,
        min_rejected_len=args.min_rejected_len,
        max_rejected_chosen_ratio=args.max_rejected_chosen_ratio,
        min_rejected_chosen_ratio=args.min_rejected_chosen_ratio,
    )

    rows, counts, failures = build_dataset(
        input_data=data,
        error_types=error_types,
        num_per_type=args.num_per_type,
        qc_cfg=qc_cfg,
        seed=args.seed,
    )

    # Shuffle and save per-type
    random.shuffle(rows)
    by_type: Dict[str, List[dict]] = {}
    for r in rows:
        by_type.setdefault(r["error_type"], []).append(r)

    mixed_out = []
    for et in error_types:
        path = os.path.join(args.output_dir, f"dpo_{et}.jsonl")
        save_jsonl(by_type.get(et, []), path)
        print(f"  {et}: {counts[et]} samples → {path}")
        mixed_out.extend(by_type.get(et, []))

    mixed_path = os.path.join(args.output_dir, "dpo_mixed.jsonl")
    random.shuffle(mixed_out)
    save_jsonl(mixed_out, mixed_path)
    print(f"  mixed: {len(mixed_out)} samples → {mixed_path}")

    # QC report
    qc_path = os.path.join(args.output_dir, "qc_report.json")
    generate_qc_report(rows, counts, failures, qc_path, args)
    print(f"\nQC report: {qc_path}")

    if args.print_sample_diffs:
        print_sample_diffs(rows, n=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
