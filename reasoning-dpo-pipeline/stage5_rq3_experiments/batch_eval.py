#!/usr/bin/env python3
"""
Stage 5: RQ3 Batch Evaluation Pipeline
========================================
Runs the evaluation engine (Stage 4) across multiple DPO model checkpoints
in parallel or sequentially, then aggregates results into structured comparison
reports (Markdown + CSV).

Research Questions addressed:
  RQ1: Does DPO improve over SFT on in-distribution problems?
  RQ2: Does DPO improve over SFT on OOD (out-of-distribution) problems?
  RQ3: How does the type of error in the DPO data affect downstream performance?

Two experiment groups:
  - **Exp 1**: Pairwise mix — equal proportions of two error types per dataset
  - **Exp 2**: Proportional injection — 1% or 10% error injection into a base dataset

Usage
-----
    # Evaluate all models in RQ3 Exp 1
    python -m stage5_rq3_experiments.batch_eval \
        --exp exp1 \
        --eval_script stage4_evaluation.vllm_eval.run_eval \
        --output_base ./results/rq3_exp1

    # Skip inference, only aggregate existing results
    python -m stage5_rq3_experiments.batch_eval \
        --exp all \
        --skip_eval \
        --output_base ./results/rq3_all

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import json
import time
import subprocess
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Exp 1: Two-type mix, 200 samples each type
EXP1_MODELS: Dict[str, str] = {
    "compute_error_missing_200each":  "path/to/exp1/compute_error_missing_200each",
    "disorder_compute_error_200each": "path/to/exp1/disorder_compute_error_200each",
    "disorder_mismatch_200each":    "path/to/exp1/disorder_mismatch_200each",
    "disorder_missing_200each":     "path/to/exp1/disorder_missing_200each",
    "mismatch_compute_error_200each": "path/to/exp1/mismatch_compute_error_200each",
    "mismatch_missing_200each":     "path/to/exp1/mismatch_missing_200each",
}

# Exp 2: 1% / 10% injection
EXP2_MODELS: Dict[str, str] = {
    "compute_error_10pct_disorder_total400":   "path/to/exp2/compute_error_10pct_disorder_total400",
    "compute_error_10pct_missing_total400":    "path/to/exp2/compute_error_10pct_missing_total400",
    "compute_error_1pct_disorder_total400":    "path/to/exp2/compute_error_1pct_disorder_total400",
    "compute_error_1pct_missing_total400":     "path/to/exp2/compute_error_1pct_missing_total400",
    "disorder_10pct_missing_total400":         "path/to/exp2/disorder_10pct_missing_total400",
    "disorder_1pct_missing_total400":          "path/to/exp2/disorder_1pct_missing_total400",
    "mismatch_10pct_disorder_total400":        "path/to/exp2/mismatch_10pct_disorder_total400",
    "mismatch_10pct_missing_total400":         "path/to/exp2/mismatch_10pct_missing_total400",
    "mismatch_1pct_disorder_total400":         "path/to/exp2/mismatch_1pct_disorder_total400",
    "mismatch_1pct_missing_total400":          "path/to/exp2/mismatch_1pct_missing_total400",
    "missing_10pct_disorder_total400":         "path/to/exp2/missing_10pct_disorder_total400",
    "missing_1pct_disorder_total400":          "path/to/exp2/missing_1pct_disorder_total400",
}

BASE_EVAL_SCRIPT = "python -m stage4_evaluation.vllm_eval.run_eval"
DEFAULT_DATA_PATH = "./data/half/validation.json"
TOKENIZER_PATH = "Qwen/Qwen2-0.5B"


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def load_summary(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(summary: dict) -> dict:
    """Flatten the summary JSON into a simple dict for CSV export."""
    results = {}
    metrics = summary.get("metrics", {})

    overall = metrics.get("overall", {})
    for key, value in overall.items():
        if isinstance(value, (int, float)):
            results[f"overall_{key}"] = value

    if "generalization_decay_rate" in overall:
        results["generalization_decay_rate"] = overall["generalization_decay_rate"]

    return results


def aggregate_exp(
    exp_name: str,
    models: Dict[str, str],
    output_base: str,
) -> Dict[str, dict]:
    """
    Load existing evaluation summaries for all models in an experiment group.
    Returns {model_name: {"model_path": ..., "metrics": flattened_metrics, "raw": raw_summary}}
    """
    results = {}
    for model_name, model_path in models.items():
        output_dir = os.path.join(output_base, exp_name, model_name)
        summary_path = os.path.join(output_dir, "summary_by_op.json")
        summary = load_summary(summary_path)
        if summary:
            results[model_name] = {
                "model_path": model_path,
                "metrics": extract_metrics(summary),
                "raw": summary,
            }
    return results


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_markdown_report(
    exp1_results: Dict[str, dict],
    exp2_results: Dict[str, dict],
    output_path: str,
) -> None:
    """Generate a comprehensive Markdown comparison report."""
    lines = [
        "# RQ3 Evaluation Results Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # ---- Overall table ----
    def overall_table(results: Dict[str, dict], title: str) -> List[str]:
        out = [f"## {title}", ""]
        out.append("| Model | pass@1 | pass@128 | reason@1 | reason@128 | gen_decay |")
        out.append("|------ |------- |--------- |--------- |----------- |---------- |")
        for name, data in sorted(results.items()):
            m = data.get("metrics", {})
            p1 = m.get("overall_answer-pass@1", 0)
            p128 = m.get("overall_answer-pass@128", 0)
            r1 = m.get("overall_reasoning-chain-pass@1", 0)
            r128 = m.get("overall_reasoning-chain-pass@128", 0)
            gd = m.get("generalization_decay_rate", None)
            gd_str = f"{gd:.2%}" if gd is not None else "N/A"
            out.append(f"| {name} | {p1:.2%} | {p128:.2%} | {r1:.2%} | {r128:.2%} | {gd_str} |")
        out.append("")
        return out

    if exp1_results:
        lines += overall_table(exp1_results, "Experiment 1: Two-Type Mix (200 each)")

    if exp2_results:
        lines += overall_table(exp2_results, "Experiment 2: Proportional Injection")

    # ---- Per-bucket detail ----
    def bucket_table(results: Dict[str, dict], title: str) -> List[str]:
        out = [f"## {title} — Per-Bucket Detail", ""]
        out.append("| Model | Bucket | pass@1 | pass@128 | reason@1 | reason@128 |")
        out.append("|------ |------- |------- |--------- |--------- |----------- |")
        for name, data in sorted(results.items()):
            raw = data.get("raw", {})
            by_bucket = raw.get("metrics", {}).get("by_bucket", {})
            for bucket in sorted(by_bucket.keys()):
                b = by_bucket[bucket]
                p1 = b.get("answer-pass@1", 0)
                p128 = b.get("answer-pass@128", 0)
                r1 = b.get("reasoning-chain-pass@1", 0)
                r128 = b.get("reasoning-chain-pass@128", 0)
                out.append(f"| {name} | {bucket} | {p1:.2%} | {p128:.2%} | {r1:.2%} | {r128:.2%} |")
        out.append("")
        return out

    if exp1_results:
        lines += bucket_table(exp1_results, "Exp 1 Per-Bucket")
    if exp2_results:
        lines += bucket_table(exp2_results, "Exp 2 Per-Bucket")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(
    exp1_results: Dict[str, dict],
    exp2_results: Dict[str, dict],
    output_dir: str,
) -> None:
    """Export aggregated metrics as CSV files."""
    import csv

    all_results = {}
    for name, data in exp1_results.items():
        all_results[f"exp1_{name}"] = data
    for name, data in exp2_results.items():
        all_results[f"exp2_{name}"] = data

    if not all_results:
        return

    # Collect all metric keys
    metric_keys = set()
    for data in all_results.values():
        for k in data.get("metrics", {}).keys():
            metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    # Overall CSV
    csv_path = os.path.join(output_dir, "rq3_overall_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "exp_group"] + metric_keys,
            extrasaction="ignore",
        )
        writer.writeheader()
        for full_name, data in sorted(all_results.items()):
            row = {"model": full_name, "exp_group": full_name.split("_", 1)[0]}
            row.update(data.get("metrics", {}))
            writer.writerow(row)
    print(f"  Overall CSV: {csv_path}")

    # Per-bucket CSV
    bucket_csv_path = os.path.join(output_dir, "rq3_bucket_metrics.csv")
    rows = []
    for full_name, data in sorted(all_results.items()):
        raw = data.get("raw", {})
        by_bucket = raw.get("metrics", {}).get("by_bucket", {})
        for bucket, bdata in sorted(by_bucket.items()):
            row = {
                "model": full_name,
                "exp_group": full_name.split("_", 1)[0],
                "bucket": bucket,
            }
            for k, v in bdata.items():
                if isinstance(v, (int, float)):
                    row[k] = v
            rows.append(row)

    if rows:
        all_keys = sorted(rows[0].keys())
        with open(bucket_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Bucket CSV: {bucket_csv_path}")


# ---------------------------------------------------------------------------
# Run evaluation for one model
# ---------------------------------------------------------------------------

def run_evaluation(
    model_name: str,
    model_path: str,
    output_dir: str,
    data_path: str,
    tokenizer_path: str,
    eval_script: str,
    skip_eval: bool,
) -> bool:
    """Run evaluation for one model. Returns True on success."""
    os.makedirs(output_dir, exist_ok=True)

    if skip_eval:
        summary_path = os.path.join(output_dir, "summary_by_op.json")
        if os.path.exists(summary_path):
            print(f"  [SKIP] {model_name} (already exists)")
            return True
        print(f"  [WARN] {model_name} — no existing results, skipping")
        return False

    print(f"  [EVAL] {model_name}: {model_path}")
    cmd = [
        "python", "-m", "stage4_evaluation.vllm_eval.run_eval",
        "--model_path", model_path,
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--tokenizer_path", tokenizer_path,
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  [OK] {model_name} ({elapsed:.0f}s)")
        return True
    else:
        print(f"  [FAIL] {model_name} ({elapsed:.0f}s)")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 5 — RQ3 Batch Evaluation")
    p.add_argument("--exp", choices=["exp1", "exp2", "all"], default="all",
                   help="Which experiment group to evaluate")
    p.add_argument("--output_base", default="./results/rq3",
                   help="Base output directory")
    p.add_argument("--eval_script", default=BASE_EVAL_SCRIPT)
    p.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    p.add_argument("--tokenizer_path", default=TOKENIZER_PATH)
    p.add_argument("--skip_eval", action="store_true",
                   help="Skip inference, only aggregate existing results")
    p.add_argument("--report_only", action="store_true",
                   help="Only generate reports from existing summaries")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_base, exist_ok=True)

    exp1_results: Dict[str, dict] = {}
    exp2_results: Dict[str, dict] = {}

    # ---- Exp 1 ----
    if args.exp in ["exp1", "all"]:
        models = EXP1_MODELS
        for model_name, model_path in models.items():
            output_dir = os.path.join(args.output_base, "exp1", model_name)
            run_evaluation(
                model_name, model_path, output_dir,
                args.data_path, args.tokenizer_path,
                args.eval_script, args.skip_eval or args.report_only,
            )
        exp1_results = aggregate_exp("exp1", EXP1_MODELS, args.output_base)

    # ---- Exp 2 ----
    if args.exp in ["exp2", "all"]:
        models = EXP2_MODELS
        for model_name, model_path in models.items():
            output_dir = os.path.join(args.output_base, "exp2", model_name)
            run_evaluation(
                model_name, model_path, output_dir,
                args.data_path, args.tokenizer_path,
                args.eval_script, args.skip_eval or args.report_only,
            )
        exp2_results = aggregate_exp("exp2", EXP2_MODELS, args.output_base)

    # ---- Reports ----
    if exp1_results or exp2_results:
        md_path = os.path.join(args.output_base, "rq3_summary_report.md")
        generate_markdown_report(exp1_results, exp2_results, md_path)
        print(f"\nMarkdown report: {md_path}")

        export_csv(exp1_results, exp2_results, args.output_base)

        json_path = os.path.join(args.output_base, "rq3_summary_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"exp1": exp1_results, "exp2": exp2_results},
                f, ensure_ascii=False, indent=2,
            )
        print(f"JSON summary: {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
