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


def _completed(output: Path, expected: dict[str, object]) -> bool:
    manifest = output / "run_manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "complete" and all(payload.get(key) == value for key, value in expected.items())


def _write_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    parser.add_argument("--resume", action="store_true", help="Skip complete runs and restart incomplete outputs.")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required, or set SFT_CHECKPOINT")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    matrix: list[dict[str, object]] = []
    manifest_path = Path(args.output_root) / "training_matrix.json"
    for name, error_type, reduction in conditions(args.suite):
        for seed in seeds:
            output = Path(args.output_root) / name / f"seed_{seed}"
            entry: dict[str, object] = {
                "name": name,
                "error_type": error_type,
                "reduction": reduction,
                "seed": seed,
                "output_dir": str(output),
                "execute": args.execute,
                "status": "pending",
            }
            matrix.append(entry)
            expected = {
                "checkpoint": args.checkpoint,
                "base_model": args.base_model,
                "seed": seed,
                "sequence_logp_reduction": reduction,
                "train_file": args.data,
                "error_types": error_type,
            }
            if args.resume and args.execute and _completed(output, expected):
                entry["status"] = "skipped_complete"
                _write_matrix(manifest_path, matrix)
                print(f"Skipping complete run: {output}")
                continue
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
            elif args.resume and output.exists():
                command.extend(["--overwrite", "true"])
            entry["status"] = "running" if args.execute else "dry_run"
            _write_matrix(manifest_path, matrix)
            try:
                subprocess.run(command, check=True)
            except BaseException:
                entry["status"] = "failed"
                _write_matrix(manifest_path, matrix)
                raise
            entry["status"] = "complete" if args.execute else "dry_run_complete"
            _write_matrix(manifest_path, matrix)


if __name__ == "__main__":
    main()
