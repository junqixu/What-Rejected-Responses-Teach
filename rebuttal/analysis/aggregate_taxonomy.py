from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, write_json


def collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("diagnostic_metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root)
        parts = relative.parts
        condition = parts[-3] if len(parts) >= 3 else "unknown"
        seed_text = parts[-2] if len(parts) >= 2 else "unknown"
        for axis, values in sorted((metrics.get("by_axis") or {}).items()):
            rows.append(
                {
                    "condition": condition,
                    "seed": seed_text.removeprefix("seed_"),
                    "eval_axis": axis,
                    "n": values.get("n"),
                    "diagnostic_accuracy": values.get("length_normalized_diagnostic_accuracy"),
                    "mean_raw_margin": values.get("mean_raw_margin"),
                    "mean_length_normalized_margin": values.get("mean_length_normalized_margin"),
                    "source": str(path),
                }
            )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["eval_axis"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (condition, axis), values in sorted(grouped.items()):
        accuracies = [float(row["diagnostic_accuracy"]) for row in values if row["diagnostic_accuracy"] is not None]
        margins = [float(row["mean_length_normalized_margin"]) for row in values if row["mean_length_normalized_margin"] is not None]
        summary.append(
            {
                "condition": condition,
                "eval_axis": axis,
                "n_seeds": len(accuracies),
                "mean_diagnostic_accuracy": mean(accuracies) if accuracies else None,
                "min_diagnostic_accuracy": min(accuracies) if accuracies else None,
                "max_diagnostic_accuracy": max(accuracies) if accuracies else None,
                "mean_length_normalized_margin": mean(margins) if margins else None,
            }
        )
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate expanded taxonomy diagnostic metrics across seeds.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    args = parser.parse_args()
    configure_logging(args.log_level)
    rows, summary = collect(Path(args.root))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    report = {"n_prediction_rows": len(rows), "n_summary_rows": len(summary)}
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    output = ensure_output_dir(args.out, args.overwrite)
    _write_csv(output / "expanded_diagnostic_matrix.csv", rows)
    _write_csv(output / "expanded_diagnostic_summary.csv", summary)
    write_json(output / "expanded_diagnostic_manifest.json", report)


if __name__ == "__main__":
    main()
