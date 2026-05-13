#!/usr/bin/env python3
"""
Utility: Merge evaluation CSVs from multiple model runs into one file.
Usage:
    python scripts/merge_eval_csvs.py \
        --dirs results/exp1 results/exp2 results/exp3 \
        --output results/tables/merged_all_details.csv
"""

import argparse, os, json, csv
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def flatten_summary(summary, model_name, model_path):
    rows = []
    metrics = summary.get("metrics", {})

    def add_row(eval_type, target, d):
        row = {
            "model_name": model_name,
            "model_path": model_path,
            "evaluation_type": eval_type,
            "evaluation_target": target,
            "num_samples": d.get("num_samples", 0),
        }
        for k, v in d.items():
            if k == "num_samples":
                continue
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

    # Overall
    if "overall" in metrics:
        add_row("overall", "overall", metrics["overall"])

    # Buckets
    for bucket, bdata in metrics.get("by_bucket", {}).items():
        add_row("bucket", bucket, bdata)

    # Individual OP
    for op_key, odata in metrics.get("by_op", {}).items():
        add_row("op", op_key, odata)

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    all_rows = []
    for d in args.dirs:
        root = Path(d)
        for summary_file in root.rglob("summary_by_op.json"):
            rel = summary_file.parent.relative_to(root)
            summary = load_json(summary_file)
            model_path = summary.get("model_path", str(rel))
            model_name = os.path.basename(model_path)
            rows = flatten_summary(summary, model_name, model_path)
            all_rows.extend(rows)

    if not all_rows:
        print("No results found.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    keys = sorted(all_rows[0].keys())

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Merged {len(all_rows)} rows from {len(args.dirs)} directories → {args.output}")


if __name__ == "__main__":
    main()
