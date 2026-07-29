from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, load_records


COLUMNS = (
    "error_pair",
    "regime",
    "n_pairs",
    "exposure_A",
    "exposure_B",
    "design_rank",
    "effective_rank",
    "condition_number",
    "train_pair_accuracy",
    "diag_A",
    "diag_B",
    "diag_AB",
    "mean_axis",
    "worst_axis",
    "axis_imbalance",
    "answer_128",
    "reasoning_128",
    "op_11_14_reasoning",
    "op_15_20_reasoning",
    "seed",
)


def _write(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(root.rglob("matrix_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        relative = report_path.relative_to(root)
        parts = relative.parts
        if parts[0] == "design" and len(parts) >= 4:
            error_pair, regime = parts[1], parts[2]
        elif len(parts) >= 3:
            error_pair, regime = parts[-3], parts[-2]
        else:
            continue
        exposures: Counter[str] = Counter()
        seed = None
        for source in report.get("source_files", []):
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path
            if not source_path.exists():
                continue
            for record in load_records(source_path):
                for label in record.get("error_labels", []):
                    exposures[str(label)] += 1
                seed = record.get("seed", seed)
        pair_names = error_pair.split("_", 1)
        exposure_values = sorted(exposures.values(), reverse=True)
        rows.append(
            {
                "error_pair": error_pair,
                "regime": regime,
                "n_pairs": report["n_samples"],
                "exposure_A": exposure_values[0] if exposure_values else None,
                "exposure_B": exposure_values[1] if len(exposure_values) > 1 else None,
                "design_rank": report["matrix_rank"],
                "effective_rank": report["effective_rank"],
                "condition_number": report["condition_number"],
                "train_pair_accuracy": None,
                "diag_A": None,
                "diag_B": None,
                "diag_AB": None,
                "mean_axis": None,
                "worst_axis": None,
                "axis_imbalance": None,
                "answer_128": None,
                "reasoning_128": None,
                "op_11_14_reasoning": None,
                "op_15_20_reasoning": None,
                "seed": seed,
            }
        )
    return rows


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(args.root)
    rows = aggregate(root)
    if args.dry_run:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return rows
    _write(root / "table_identifiability.csv", rows, COLUMNS)
    summary_columns = ("error_pair", "regime", "n_pairs", "design_rank", "effective_rank", "seed")
    _write(root / "table_identifiability_summary.csv", rows, summary_columns)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate design and evaluation results without filling absent metrics.")
    parser.add_argument("--root", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
