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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all confounded/orthogonal identifiability DPO conditions.")
    parser.add_argument("--checkpoint", default=os.getenv("SFT_CHECKPOINT"))
    parser.add_argument("--config_root", default="configs/rebuttal")
    parser.add_argument("--output_root", default="outputs/rebuttal_identifiability/checkpoints")
    parser.add_argument("--seeds", default="414,6201,2026")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip complete runs and restart incomplete outputs.")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required, or set SFT_CHECKPOINT")
    configs = sorted(Path(args.config_root).glob("ident_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"no identifiability configs under {args.config_root}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    matrix: list[dict[str, object]] = []
    matrix_path = Path(args.output_root) / "identifiability_matrix.json"
    for config in configs:
        name = config.stem.removeprefix("ident_")
        config_payload = json.loads(config.read_text(encoding="utf-8"))
        for seed in seeds:
            output = Path(args.output_root) / name / f"seed_{seed}"
            entry: dict[str, object] = {
                "condition": name,
                "config": str(config),
                "seed": seed,
                "output_dir": str(output),
                "status": "pending",
            }
            matrix.append(entry)
            expected = {
                "checkpoint": args.checkpoint,
                "seed": seed,
                "train_file": config_payload.get("train_file"),
                "sequence_logp_reduction": config_payload.get("sequence_logp_reduction", "sum"),
                "save_strategy": config_payload.get("save_strategy", "no"),
            }
            if args.resume and args.execute and _completed(output, expected):
                entry["status"] = "skipped_complete"
                _write_matrix(matrix_path, matrix)
                print(f"Skipping complete run: {output}")
                continue
            command = [
                sys.executable,
                "-m",
                "rebuttal.train_dpo",
                "--config",
                str(config),
                "--checkpoint",
                args.checkpoint,
                "--output_dir",
                str(output),
                "--seed",
                str(seed),
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            if not args.execute:
                command.append("--dry_run")
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
            _write_matrix(matrix_path, matrix)


if __name__ == "__main__":
    main()
