#!/usr/bin/env python3
"""
Evaluation Analysis Utilities
============================
Utilities for loading, normalising, and exporting evaluation results from
multiple model checkpoints into structured tables and comparative reports.

Functions
---------
- load_and_merge_results()  : load all summary_by_op.json files from a directory tree
- normalise_model_names()   : apply a renaming map (e.g. "dpo_full_disorder_414" → "disorder")
- build_overall_table()    : pivot to a model × metric table (CSV + pandas)
- build_bucket_table()      : per-bucket breakdown
- build_op_table()         : per-OP breakdown
- compute_gap_table()      : answer-pass@k − reasoning-chain-pass@k
- compute_retention_table() : retention rate of harder buckets vs op_2_10
- export_all()             : write all tables as CSV + Markdown

Usage
-----
    python -m stage4_evaluation.analysis \
        --results_dir ./results \
        --output_dir ./results/tables \
        --rename_map configs/model_rename.json

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# Model name normalisation
# ---------------------------------------------------------------------------

DEFAULT_RENAME_MAP: Dict[str, str] = {
    "Qwen_Qwen2-0.5B": "base",
    "qwen2_0.5b_sft_op10_checkpoint-10339": "sft",
    "dpo_full_disorder_414": "disorder",
    "dpo_full_final_compute_error_1": "compute_error",
    "dpo_full_missing_nodes_414": "missing_nodes",
    "dpo_full_mismatch_414": "dependency_mismatch",
}

MODEL_ORDER = ["base", "sft", "disorder", "compute_error", "missing_nodes", "dependency_mismatch"]
BUCKET_ORDER = ["op_2_10", "op_11_14", "op_15_20"]


def normalise_model_name(raw: str, rename_map: Optional[Dict[str, str]] = None) -> str:
    """Apply renaming rules to a raw model name."""
    raw = str(raw)
    if rename_map and raw in rename_map:
        return rename_map[raw]
    s = raw.lower()
    for key, value in DEFAULT_RENAME_MAP.items():
        if key.lower() in s or s in key.lower():
            return value
    return raw


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

@dataclass
class LoadedResult:
    model_name: str
    model_path: str
    summary: dict
    k_list: List[int]
    num_test_samples: int


def load_summary_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_summary_files(root: Path) -> Dict[str, Path]:
    """Find all summary_by_op.json files under root."""
    result = {}
    for p in root.rglob("summary_by_op.json"):
        rel = p.parent.relative_to(root)
        result[str(rel)] = p
    return result


def load_results(
    results_dir: str,
    rename_map: Optional[Dict[str, str]] = None,
) -> List[LoadedResult]:
    """Load all evaluation summaries recursively."""
    root = Path(results_dir)
    files = find_summary_files(root)
    results: List[LoadedResult] = []

    for rel_path, summary_path in sorted(files.items()):
        summary = load_summary_json(summary_path)
        if not summary:
            continue

        model_path = summary.get("model_path", "")
        model_name = normalise_model_name(
            os.path.basename(model_path) or os.path.basename(str(rel_path)),
            rename_map,
        )

        k_list_raw = summary.get("k_list", [1, 2, 4, 8, 16, 32, 64, 128])
        k_list = [int(x) for x in k_list_raw]

        results.append(LoadedResult(
            model_name=model_name,
            model_path=model_path,
            summary=summary,
            k_list=k_list,
            num_test_samples=summary.get("num_test_samples", 0),
        ))

    return results


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_overall_table(results: List[LoadedResult]) -> pd.DataFrame:
    """Pivot: rows = models, cols = overall metric names."""
    rows = []
    for r in results:
        m = r.summary.get("metrics", {}).get("overall", {})
        row = {"model": r.model_name, "num_test_samples": r.num_test_samples}
        for k in r.k_list:
            for prefix in ["answer-pass", "reasoning-chain-pass"]:
                key = f"{prefix}@{k}"
                row[f"overall_{key}"] = m.get(key)
        for key in ["reasoning-chain-accuracy", "generalization-decay-rate",
                    "generalization_decay_rate"]:
            if key in m:
                row[f"overall_{key}"] = m[key]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def build_bucket_table(results: List[LoadedResult]) -> pd.DataFrame:
    """Per-model, per-bucket metrics."""
    rows = []
    for r in results:
        by_bucket = r.summary.get("metrics", {}).get("by_bucket", {})
        for bucket, bdata in by_bucket.items():
            row = {"model": r.model_name, "bucket": bucket}
            for k in r.k_list:
                for prefix in ["answer-pass", "reasoning-chain-pass"]:
                    key = f"{prefix}@{k}"
                    row[f"{bucket}_{key}"] = bdata.get(key)
            row[f"{bucket}_num_samples"] = bdata.get("num_samples")
            rows.append(row)

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df["bucket"] = pd.Categorical(df["bucket"], categories=BUCKET_ORDER, ordered=True)
    return df.sort_values(["model", "bucket"]).reset_index(drop=True)


def build_op_table(results: List[LoadedResult]) -> pd.DataFrame:
    """Per-model, per-OP metrics."""
    rows = []
    for r in results:
        by_op = r.summary.get("metrics", {}).get("by_op", {})
        for op_key, odata in sorted(by_op.items()):
            try:
                op_int = int(op_key)
            except ValueError:
                op_int = 999
            row = {"model": r.model_name, "op": op_int}
            for k in r.k_list:
                for prefix in ["answer-pass", "reasoning-chain-pass"]:
                    key = f"{prefix}@{k}"
                    row[f"op{op_int}_{key}"] = odata.get(key)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(["model", "op"]).reset_index(drop=True)


def build_gap_table(overall_df: pd.DataFrame) -> pd.DataFrame:
    """Compute answer-pass@k − reasoning-chain-pass@k."""
    rows = []
    for _, row in overall_df.iterrows():
        model = row["model"]
        row_data = {"model": model}
        for k in [1, 8, 32, 128]:
            ans = row.get(f"overall_answer-pass@{k}")
            chn = row.get(f"overall_reasoning-chain-pass@{k}")
            if ans is not None and chn is not None:
                row_data[f"gap@{k}"] = float(ans) - float(chn)
        rows.append(row_data)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_all(
    results: List[LoadedResult],
    output_dir: str,
    rename_map: Optional[Dict[str, str]] = None,
) -> None:
    """Write all tables as CSV and a combined Markdown report."""
    os.makedirs(output_dir, exist_ok=True)

    overall_df = build_overall_table(results)
    overall_df.to_csv(
        os.path.join(output_dir, "table_overall.csv"), index=False, encoding="utf-8-sig",
    )

    bucket_df = build_bucket_table(results)
    bucket_df.to_csv(
        os.path.join(output_dir, "table_bucket.csv"), index=False, encoding="utf-8-sig",
    )

    op_df = build_op_table(results)
    op_df.to_csv(
        os.path.join(output_dir, "table_op.csv"), index=False, encoding="utf-8-sig",
    )

    gap_df = build_gap_table(overall_df)
    gap_df.to_csv(
        os.path.join(output_dir, "table_gap.csv"), index=False, encoding="utf-8-sig",
    )

    # Markdown summary
    md_lines = ["# Evaluation Results Summary\n"]
    md_lines.append(f"Models: {', '.join(sorted(overall_df['model'].unique()))}\n")

    md_lines.append("## Overall pass@128\n")
    cols = [c for c in overall_df.columns if "pass@128" in c and "reasoning" not in c]
    if cols:
        md_lines.append(overall_df[["model"] + cols].to_markdown(index=False))

    md_lines.append("\n## Generalization Decay Rate\n")
    gd_col = [c for c in overall_df.columns if "generalization" in c]
    if gd_col:
        md_lines.append(overall_df[["model"] + gd_col].to_markdown(index=False))

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Exported tables to: {output_dir}")
    print(f"  table_overall.csv ({len(overall_df)} rows)")
    print(f"  table_bucket.csv  ({len(bucket_df)} rows)")
    print(f"  table_op.csv      ({len(op_df)} rows)")
    print(f"  table_gap.csv     ({len(gap_df)} rows)")
    print(f"  summary.md")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — Evaluation analysis utilities")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rename_map", default="",
                   help="Path to JSON with {raw_name: clean_name} mapping")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rename_map = None
    if args.rename_map and os.path.exists(args.rename_map):
        with open(args.rename_map) as f:
            rename_map = json.load(f)

    results = load_results(args.results_dir, rename_map)
    print(f"Loaded {len(results)} evaluation results.")
    export_all(results, args.output_dir, rename_map)


if __name__ == "__main__":
    main()
