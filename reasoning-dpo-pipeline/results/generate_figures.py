#!/usr/bin/env python3
"""
Thesis Figure Generator
======================
Generates all figures and summary tables for the thesis from the
evaluation results CSV files produced by the analysis utilities.

Figures generated
----------------
1. Heatmaps: answer-pass@k and reasoning-chain-pass@k for overall performance
2. Line plots: pass@k curves by OP bucket
3. Bar charts: answer-reasoning gap@32 across buckets
4. Retention charts: how well does each model generalise to harder OP buckets?
5. Reasoning boundary plots: pass@1 and pass@32 vs task difficulty (OP value)
6. Smoothed boundary plots: 3-op rolling average for cleaner boundary visualization

Usage
-----
    python -m results.generate_figures \
        --input_csv ./results/tables/merged_all_details.csv \
        --output_dir ./results/thesis_figures \
        --format png,pdf

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ORDER = ["base", "sft", "disorder", "compute_error", "missing_nodes", "dependency_mismatch"]
MAIN_MODELS = ["sft", "compute_error", "missing_nodes", "dependency_mismatch"]
BUCKET_ORDER = ["op_2_10", "op_11_14", "op_15_20"]
MAIN_KS = [1, 8, 32, 128]
ALL_KS = [1, 2, 4, 8, 16, 32, 64, 128]

MODEL_DISPLAY_NAMES = {
    "base": "Base (Qwen2-0.5B)",
    "sft": "SFT",
    "disorder": "DPO-Disorder",
    "compute_error": "DPO-ComputeError",
    "missing_nodes": "DPO-MissingNodes",
    "dependency_mismatch": "DPO-DepMismatch",
}

COLOR_PALETTE = {
    "base": "#cccccc",
    "sft": "#2196F3",
    "disorder": "#4CAF50",
    "compute_error": "#FF9800",
    "missing_nodes": "#F44336",
    "dependency_mismatch": "#9C27B0",
}

MARKERS = {
    "base": "X",
    "sft": "o",
    "disorder": "s",
    "compute_error": "^",
    "missing_nodes": "D",
    "dependency_mismatch": "p",
}

BUCKET_TITLES = {
    "op_2_10": "Low difficulty (op 2–10)",
    "op_11_14": "Medium difficulty (op 11–14)",
    "op_15_20": "High difficulty (op 15–20)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pct_formatter(x: float, pos: int) -> str:
    return f"{x * 100:.0f}%"


def load_normalise_csv(csv_path: str) -> pd.DataFrame:
    """Load and normalise a merged results CSV."""
    df = pd.read_csv(csv_path)

    if "model_name" in df.columns:
        df["model"] = df["model_name"].apply(
            lambda x: _normalise_one(str(x))
        )
    elif "model" in df.columns:
        df["model"] = df["model"].apply(lambda x: _normalise_one(str(x)))

    numeric_cols = [c for c in df.columns
                    if any(k in c for k in ["pass@", "reasoning-chain", "decay", "gap"])]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _normalise_one(raw: str) -> str:
    raw = str(raw)
    s = raw.lower()
    for key in ["compute_error", "compute", "missing_nodes", "missing", "disorder",
                "mismatch", "dep", "sft", "qwen"]:
        if key in s:
            if "compute" in s:
                return "compute_error"
            if "missing" in s:
                return "missing_nodes"
            if "mismatch" in s or "dep" in s:
                return "dependency_mismatch"
            if "disorder" in s:
                return "disorder"
            if "sft" in s:
                return "sft"
            if "qwen" in s or "base" in s:
                return "base"
    return raw


def get_overall(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["evaluation_type"] == "overall"].copy()
    cats = [m for m in MODEL_ORDER if m in sub["model"].values]
    sub["model"] = pd.Categorical(sub["model"], categories=cats, ordered=True)
    return sub.sort_values("model").reset_index(drop=True)


def get_bucket(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["evaluation_type"] == "bucket"].copy()
    sub["bucket"] = pd.Categorical(sub["evaluation_target"], categories=BUCKET_ORDER, ordered=True)
    cats = [m for m in MODEL_ORDER if m in sub["model"].values]
    sub["model"] = pd.Categorical(sub["model"], categories=cats, ordered=True)
    return sub.sort_values(["bucket", "model"]).reset_index(drop=True)


def get_op(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["evaluation_type"] == "op"].copy()
    sub["op"] = pd.to_numeric(sub["evaluation_target"], errors="coerce")
    cats = [m for m in MODEL_ORDER if m in sub["model"].values]
    sub["model"] = pd.Categorical(sub["model"], categories=cats, ordered=True)
    return sub.sort_values(["model", "op"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1: Heatmaps
# ---------------------------------------------------------------------------

def draw_heatmap(
    matrix: pd.DataFrame,
    title: str,
    out_path: str,
    fmt: str = ".3f",
    figsize: tuple = (8, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title(title, fontsize=11)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overall_heatmaps(overall_df: pd.DataFrame, output_dir: str) -> None:
    answer_cols = [f"answer-pass@{k}" for k in MAIN_KS]
    reason_cols = [f"reasoning-chain-pass@{k}" for k in MAIN_KS]

    available_answer = [c for c in answer_cols if c in overall_df.columns]
    available_reason = [c for c in reason_cols if c in overall_df.columns]

    if available_answer:
        m = overall_df.set_index("model")[available_answer].copy()
        m.columns = [f"@{k}" for k in MAIN_KS if f"answer-pass@{k}" in available_answer]
        draw_heatmap(
            m, "Overall Answer-pass@k",
            os.path.join(output_dir, "fig01_overall_answer_heatmap.png"),
        )

    if available_reason:
        m = overall_df.set_index("model")[available_reason].copy()
        m.columns = [f"@{k}" for k in MAIN_KS if f"reasoning-chain-pass@{k}" in available_reason]
        draw_heatmap(
            m, "Overall Reasoning-chain-pass@k",
            os.path.join(output_dir, "fig02_overall_reasoning_heatmap.png"),
        )


# ---------------------------------------------------------------------------
# Figure 2: Pass@k curves by bucket
# ---------------------------------------------------------------------------

def plot_bucket_curves(bucket_df: pd.DataFrame, output_dir: str) -> None:
    for bucket in BUCKET_ORDER:
        sub = bucket_df[bucket_df["bucket"] == bucket]
        if sub.empty:
            continue

        for metric, label in [
            ("answer-pass@", "Answer-pass"),
            ("reasoning-chain-pass@", "Reasoning-chain-pass"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for model in MODEL_ORDER:
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                ys = []
                for k in MAIN_KS:
                    col = f"{metric}{k}"
                    if col in row.columns:
                        v = row[col].values[0]
                        ys.append(float(v) if pd.notna(v) else 0)
                    else:
                        ys.append(np.nan)
                ax.plot(MAIN_KS, ys, marker=MARKERS.get(model, "o"),
                        linewidth=2, markersize=6,
                        color=COLOR_PALETTE.get(model, "#000"),
                        label=MODEL_DISPLAY_NAMES.get(model, model))

            ax.set_xscale("log", base=2)
            ax.set_xlabel("k")
            ax.set_ylabel(label)
            ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
            ax.set_ylim(0, 1.05)
            ax.set_xticks(MAIN_KS)
            ax.grid(True, alpha=0.25)
            ax.legend(ncol=2, fontsize=8)
            ax.set_title(f"{label} — {BUCKET_TITLES.get(bucket, bucket)}")
            fig.tight_layout()

            slug = metric.replace("@", "").replace("-", "_")
            fig.savefig(
                os.path.join(output_dir, f"fig_{slug}_{bucket}.png"),
                dpi=220, bbox_inches="tight",
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Gap analysis
# ---------------------------------------------------------------------------

def plot_gap_analysis(overall_df: pd.DataFrame, bucket_df: pd.DataFrame, output_dir: str) -> None:
    # Overall gap line plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in MODEL_ORDER:
        row = overall_df[overall_df["model"] == model]
        if row.empty:
            continue
        ys = []
        for k in MAIN_KS:
            ans_col = f"answer-pass@{k}"
            chn_col = f"reasoning-chain-pass@{k}"
            if ans_col in row.columns and chn_col in row.columns:
                a = row[ans_col].values[0]
                c = row[chn_col].values[0]
                ys.append(float(a) - float(c) if (pd.notna(a) and pd.notna(c)) else np.nan)
            else:
                ys.append(np.nan)
        ax.plot(MAIN_KS, ys, marker=MARKERS.get(model, "o"),
                linewidth=2, color=COLOR_PALETTE.get(model, "#000"),
                label=MODEL_DISPLAY_NAMES.get(model, model))

    ax.set_xscale("log", base=2)
    ax.set_xlabel("k")
    ax.set_ylabel("Gap (answer − reasoning)")
    ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
    ax.set_xticks(MAIN_KS)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Answer–Reasoning Gap vs k")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_gap_overall.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Gap@32 by bucket bar chart
    buckets_in_data = [b for b in BUCKET_ORDER if b in bucket_df["bucket"].values]
    if buckets_in_data:
        x = list(range(len(MODEL_ORDER)))
        width = 0.25
        n_buckets = len(buckets_in_data)

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, bucket in enumerate(buckets_in_data):
            ys = []
            for model in MODEL_ORDER:
                row = bucket_df[(bucket_df["model"] == model) & (bucket_df["bucket"] == bucket)]
                if row.empty:
                    ys.append(0.0)
                else:
                    a = row[f"answer-pass@32"].values[0]
                    c = row[f"reasoning-chain-pass@32"].values[0]
                    ys.append(float(a) - float(c) if (pd.notna(a) and pd.notna(c)) else 0.0)

            offset = (i - n_buckets / 2 + 0.5) * width
            ax.bar([v + offset for v in x], ys, width=width,
                   label=BUCKET_TITLES.get(bucket, bucket), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER], rotation=20)
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_ylabel("Gap@32")
        ax.set_title("Answer–Reasoning Gap@32 by Bucket")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig_gap32_by_bucket.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Generalization / retention
# ---------------------------------------------------------------------------

def plot_retention(
    bucket_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Retention rate = harder bucket score / easy bucket score."""
    buckets = ["op_2_10", "op_11_14", "op_15_20"]

    for metric_base, metric_label in [
        ("answer-pass@", "Answer"),
        ("reasoning-chain-pass@", "Reasoning"),
    ]:
        for k in [1, 32, 128]:
            metric_col = f"{metric_base}{k}"
            retention_rows = []
            for model in MODEL_ORDER:
                anchor_vals = bucket_df[
                    (bucket_df["model"] == model) &
                    (bucket_df["bucket"] == "op_2_10")
                ][metric_col]
                anchor = float(anchor_vals.values[0]) if not anchor_vals.empty and pd.notna(anchor_vals.values[0]) else None

                for hard_bucket in ["op_11_14", "op_15_20"]:
                    hard_vals = bucket_df[
                        (bucket_df["model"] == model) &
                        (bucket_df["bucket"] == hard_bucket)
                    ][metric_col]
                    hard = float(hard_vals.values[0]) if not hard_vals.empty and pd.notna(hard_vals.values[0]) else None

                    retention = hard / anchor if (anchor and hard is not None and anchor > 0) else np.nan
                    retention_rows.append({
                        "model": MODEL_DISPLAY_NAMES.get(model, model),
                        "bucket": BUCKET_TITLES.get(hard_bucket, hard_bucket),
                        "retention": retention,
                    })

            if not retention_rows:
                continue
            ret_df = pd.DataFrame(retention_rows)
            pivot = ret_df.pivot(index="model", columns="bucket", values="retention")
            pivot = pivot.reindex([MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER
                                    if MODEL_DISPLAY_NAMES.get(m, m) in pivot.index])

            fig, ax = plt.subplots(figsize=(8, 4.5))
            pivot.plot(kind="bar", ax=ax, width=0.7, alpha=0.85)
            ax.set_title(f"{metric_label} Retention@{k} (vs op_2_10)")
            ax.set_ylabel("Retention rate")
            ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
            ax.set_ylim(0, 1.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2, axis="y")
            fig.tight_layout()
            slug = metric_base.replace("@", "").replace("-", "_")
            fig.savefig(
                os.path.join(output_dir, f"fig_retention_{slug}_{k}.png"),
                dpi=220, bbox_inches="tight",
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Reasoning boundary plots
# ---------------------------------------------------------------------------

def plot_reasoning_boundary(op_df: pd.DataFrame, output_dir: str) -> None:
    """pass@k vs OP value (task difficulty) for multiple models."""
    boundary_df = op_df[op_df["model"].isin(MODEL_ORDER)].copy()
    ops = sorted(boundary_df["op"].dropna().unique())

    if len(ops) == 0:
        return

    for metric, label in [
        ("answer-pass@1", "Answer-pass@1"),
        ("answer-pass@32", "Answer-pass@32"),
        ("reasoning-chain-pass@1", "Reasoning-chain-pass@1"),
        ("reasoning-chain-pass@32", "Reasoning-chain-pass@32"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))

        # Shade OOD regions
        ax.axvspan(10.5, 14.5, alpha=0.07, label="OOD-edge (op 11–14)")
        ax.axvspan(14.5, 20.5, alpha=0.04, label="OOD-hard (op 15–20)")
        ax.axvline(14.5, linestyle="--", linewidth=1.2, color="gray")

        for model in MODEL_ORDER:
            sub = boundary_df[boundary_df["model"] == model].sort_values("op")
            if sub.empty or metric not in sub.columns:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            ax.plot(sub["op"], vals, marker=MARKERS.get(model, "o"),
                    linewidth=2, markersize=5,
                    color=COLOR_PALETTE.get(model, "#000"),
                    label=MODEL_DISPLAY_NAMES.get(model, model))

        ax.set_xlabel("Task difficulty (number of operations, op)")
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_xticks(list(range(int(min(ops)), int(max(ops)) + 1)))
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=2, fontsize=8, frameon=True)
        ax.set_title(f"{label} vs Task Difficulty")

        slug = metric.replace("@", "_").replace("-", "_")
        fig.savefig(
            os.path.join(output_dir, f"fig_boundary_{slug}.png"),
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)

        # Smoothed version (3-op rolling average)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axvspan(10.5, 14.5, alpha=0.07)
        ax.axvspan(14.5, 20.5, alpha=0.04)
        ax.axvline(14.5, linestyle="--", linewidth=1.2, color="gray")

        for model in MAIN_MODELS:
            sub = boundary_df[boundary_df["model"] == model].sort_values("op")
            if sub.empty or metric not in sub.columns:
                continue
            vals = pd.to_numeric(sub[metric], errors="coerce")
            smooth = vals.rolling(3, center=True, min_periods=1).mean()
            ax.plot(sub["op"], smooth, marker=MARKERS.get(model, "o"),
                    linewidth=2, markersize=5,
                    color=COLOR_PALETTE.get(model, "#000"),
                    label=MODEL_DISPLAY_NAMES.get(model, model))

        ax.set_xlabel("Task difficulty (op)")
        ax.set_ylabel(label + " (3-op MA)")
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_xticks(list(range(int(min(ops)), int(max(ops)) + 1)))
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=2, fontsize=8, frameon=True)
        ax.set_title(f"{label} (3-op rolling average)")
        slug = metric.replace("@", "_").replace("-", "_")
        fig.savefig(
            os.path.join(output_dir, f"fig_boundary_smooth_{slug}.png"),
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: Generalization decay rate bar
# ---------------------------------------------------------------------------

def plot_decay_bar(overall_df: pd.DataFrame, output_dir: str) -> None:
    for col in ["generalization-decay-rate", "generalization_decay_rate"]:
        if col not in overall_df.columns:
            continue
        vals = overall_df.set_index("model").reindex(MODEL_ORDER)[col]
        vals = pd.to_numeric(vals, errors="coerce").fillna(0)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [COLOR_PALETTE.get(m, "#888") for m in MODEL_ORDER if m in vals.index]
        filtered_vals = [vals[m] for m in MODEL_ORDER if m in vals.index]
        labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER if m in vals.index]

        ax.bar(labels, filtered_vals, color=colors, alpha=0.85)
        ax.set_ylabel("Generalisation Decay Rate")
        ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
        ax.set_title("Generalisation Decay Rate by Model (vs op_2_10)")
        ax.set_xticklabels(labels, rotation=20)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        slug = col.replace("-", "_").replace("rate", "").replace("__", "_")
        fig.savefig(
            os.path.join(output_dir, f"fig_decay_rate_{slug}.png"),
            dpi=220, bbox_inches="tight",
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Thesis Figure Generator")
    p.add_argument("--input_csv", default="./results/tables/merged_all_details.csv")
    p.add_argument("--output_dir", default="./results/thesis_figures")
    p.add_argument("--format", default="png", help="Output format: png, pdf, or both")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    if not os.path.exists(args.input_csv):
        print(f"[WARN] Input CSV not found: {args.input_csv}")
        print("Generating sample data for demonstration...")
        # Generate a demo CSV for testing
        import numpy as np
        rng = np.random.default_rng(42)
        rows = []
        for model in MODEL_ORDER:
            for bucket in ["overall"] + BUCKET_ORDER:
                row = {"model": model, "evaluation_type": "bucket" if bucket != "overall" else "overall",
                       "evaluation_target": bucket}
                for k in ALL_KS:
                    base = rng.uniform(0.3, 0.8)
                    row[f"answer-pass@{k}"] = base + rng.uniform(-0.05, 0.05)
                    row[f"reasoning-chain-pass@{k}"] = base * 0.7 + rng.uniform(-0.05, 0.05)
                row["num_samples"] = 100
                rows.append(row)
            for op in range(2, 21):
                row = {"model": model, "evaluation_type": "op",
                       "evaluation_target": str(op), "op": op}
                for k in ALL_KS:
                    base = max(0.1, 0.9 - op * 0.04 + rng.uniform(-0.03, 0.03))
                    row[f"answer-pass@{k}"] = base + rng.uniform(-0.05, 0.05)
                    row[f"reasoning-chain-pass@{k}"] = base * 0.7 + rng.uniform(-0.05, 0.05)
                rows.append(row)
        pd.DataFrame(rows).to_csv(args.input_csv, index=False)
        print(f"[INFO] Demo CSV written to: {args.input_csv}")

    df = load_normalise_csv(args.input_csv)

    overall_df = get_overall(df)
    bucket_df = get_bucket(df)
    op_df = get_op(df)

    print(f"  Overall rows   : {len(overall_df)}")
    print(f"  Bucket rows    : {len(bucket_df)}")
    print(f"  OP rows        : {len(op_df)}")

    plot_overall_heatmaps(overall_df, args.output_dir)
    plot_bucket_curves(bucket_df, args.output_dir)
    plot_gap_analysis(overall_df, bucket_df, args.output_dir)
    plot_retention(bucket_df, args.output_dir)
    plot_reasoning_boundary(op_df, args.output_dir)
    plot_decay_bar(overall_df, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
