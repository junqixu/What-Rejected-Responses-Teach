from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_train_matrix import conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every taxonomy checkpoint on a shared held-out split.")
    parser.add_argument("--checkpoint_root", default="outputs/rebuttal_taxonomy/checkpoints")
    parser.add_argument("--output_root", default="outputs/rebuttal_taxonomy/diagnostics")
    parser.add_argument("--data_root", default="data/rebuttal_release")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--suite", choices=("p0", "full"), default="p0")
    parser.add_argument("--seeds", default="414,6201,2026")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for name, _error_type, _reduction in conditions(args.suite):
        for seed in seeds:
            checkpoint = Path(args.checkpoint_root) / name / f"seed_{seed}"
            output = Path(args.output_root) / args.split / name / f"seed_{seed}"
            command = [
                sys.executable,
                "-m",
                "rebuttal.evaluation.eval_identifiability",
                "--eval_data",
                str(Path(args.data_root) / f"{args.split}.jsonl"),
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                args.tokenizer,
                "--out",
                str(output),
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            if not args.execute:
                command.append("--dry_run")
            elif not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            subprocess.run(command, check=True)
    if args.execute:
        aggregate_out = Path(args.output_root) / args.split / "aggregate"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rebuttal.analysis.aggregate_taxonomy",
                "--root",
                str(Path(args.output_root) / args.split),
                "--out",
                str(aggregate_out),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
