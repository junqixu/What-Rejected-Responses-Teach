from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.identifiability.feature_schema import ERROR_FEATURES


P0 = [
    ("mix_sum", None, "sum"),
    ("mix_mean", None, "mean"),
    ("operation_substitution_sum", "operation_substitution", "sum"),
    ("wrong_target_sum", "wrong_target", "sum"),
]


def conditions(suite: str) -> list[tuple[str, str | None, str]]:
    if suite == "p0":
        return P0
    result = list(P0)
    existing = {axis for _name, axis, _reduction in result if axis}
    result.extend((f"{axis}_sum", axis, "sum") for axis in ERROR_FEATURES if axis not in existing)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible taxonomy DPO training matrix.")
    parser.add_argument("--checkpoint", default=os.getenv("SFT_CHECKPOINT"))
    parser.add_argument("--base_model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--data", default="data/rebuttal_release/train.jsonl")
    parser.add_argument("--output_root", default="outputs/rebuttal_taxonomy/checkpoints")
    parser.add_argument("--suite", choices=("p0", "full"), default="p0")
    parser.add_argument("--seeds", default="414,6201,2026")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Run GPU training; otherwise only run trainer dry-runs.")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required, or set SFT_CHECKPOINT")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    matrix: list[dict[str, object]] = []
    for name, error_type, reduction in conditions(args.suite):
        for seed in seeds:
            output = Path(args.output_root) / name / f"seed_{seed}"
            command = [
                sys.executable,
                "-m",
                "rebuttal.train_dpo",
                "--train_file",
                args.data,
                "--output_dir",
                str(output),
                "--checkpoint",
                args.checkpoint,
                "--base_model",
                args.base_model,
                "--sequence_logp_reduction",
                reduction,
                "--seed",
                str(seed),
            ]
            if error_type:
                command.extend(["--error_types", error_type])
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            if not args.execute:
                command.append("--dry_run")
            matrix.append(
                {
                    "name": name,
                    "error_type": error_type,
                    "reduction": reduction,
                    "seed": seed,
                    "output_dir": str(output),
                    "execute": args.execute,
                }
            )
            subprocess.run(command, check=True)
    manifest_path = Path(args.output_root) / "training_matrix.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
