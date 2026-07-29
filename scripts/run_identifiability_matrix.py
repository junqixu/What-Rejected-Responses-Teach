from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all confounded/orthogonal identifiability DPO conditions.")
    parser.add_argument("--checkpoint", default=os.getenv("SFT_CHECKPOINT"))
    parser.add_argument("--config_root", default="configs/rebuttal")
    parser.add_argument("--output_root", default="outputs/rebuttal_identifiability/checkpoints")
    parser.add_argument("--seeds", default="414,6201,2026")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required, or set SFT_CHECKPOINT")
    configs = sorted(Path(args.config_root).glob("ident_*.yaml"))
    if not configs:
        raise FileNotFoundError(f"no identifiability configs under {args.config_root}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    for config in configs:
        name = config.stem.removeprefix("ident_")
        for seed in seeds:
            output = Path(args.output_root) / name / f"seed_{seed}"
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
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
