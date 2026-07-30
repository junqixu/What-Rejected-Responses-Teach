from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_train_matrix import conditions


def _write_matrix(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation_complete(output: Path, expected: dict[str, object]) -> bool:
    manifest = output / "evaluation_manifest.json"
    if not (output / "diagnostic_metrics.json").exists() or not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


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
    parser.add_argument("--resume", action="store_true", help="Skip complete evaluations and replace partial outputs.")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    matrix: list[dict[str, object]] = []
    matrix_path = Path(args.output_root) / args.split / "evaluation_matrix.json"
    for name, _error_type, _reduction in conditions(args.suite):
        for seed in seeds:
            checkpoint = Path(args.checkpoint_root) / name / f"seed_{seed}"
            output = Path(args.output_root) / args.split / name / f"seed_{seed}"
            evaluation_manifest = {
                "checkpoint": str(checkpoint),
                "checkpoint_run_manifest_sha256": _sha256(checkpoint / "run_manifest.json"),
                "data": str(Path(args.data_root) / f"{args.split}.jsonl"),
                "split": args.split,
                "tokenizer": args.tokenizer,
            }
            entry: dict[str, object] = {
                "condition": name,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "output_dir": str(output),
                "split": args.split,
                "status": "pending",
            }
            matrix.append(entry)
            if args.resume and args.execute and _evaluation_complete(output, evaluation_manifest):
                entry["status"] = "skipped_complete"
                _write_matrix(matrix_path, matrix)
                print(f"Skipping complete evaluation: {output}")
                continue
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
            elif args.resume and output.exists():
                command.extend(["--overwrite", "true"])
            entry["status"] = "running" if args.execute else "dry_run"
            _write_matrix(matrix_path, matrix)
            try:
                subprocess.run(command, check=True)
            except BaseException:
                entry["status"] = "failed"
                _write_matrix(matrix_path, matrix)
                raise
            entry["status"] = "complete" if args.execute else "dry_run_complete"
            if args.execute:
                (output / "evaluation_manifest.json").write_text(
                    json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            _write_matrix(matrix_path, matrix)
    if args.execute:
        aggregate_out = Path(args.output_root) / args.split / "aggregate"
        aggregate_command = [
            sys.executable,
            "-m",
            "rebuttal.analysis.aggregate_taxonomy",
            "--root",
            str(Path(args.output_root) / args.split),
            "--out",
            str(aggregate_out),
        ]
        if args.resume and aggregate_out.exists():
            aggregate_command.extend(["--overwrite", "true"])
        subprocess.run(aggregate_command, check=True)


if __name__ == "__main__":
    main()
