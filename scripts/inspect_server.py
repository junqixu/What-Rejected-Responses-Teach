from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> dict[str, Any]:
    executable = shutil.which(args[0])
    if not executable:
        return {"available": False, "command": args}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "command": args, "error": str(error)}
    return {
        "available": True,
        "command": args,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def gpu_report() -> dict[str, Any]:
    query = command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, Any]] = []
    if query.get("exit_code") == 0:
        for line in str(query.get("stdout", "")).splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 5:
                devices.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                        "memory_free_mib": int(parts[3]),
                        "driver_version": parts[4],
                    }
                )
    return {"query": query, "devices": devices}


def torch_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a secret-free GPU server environment report.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    disk = shutil.disk_usage(REPO_ROOT)
    checkpoint = os.getenv("SFT_CHECKPOINT")
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "cpu_count": os.cpu_count(),
        "memory": command(["free", "-h"]),
        "disk": {
            "path": str(REPO_ROOT),
            "total_gib": round(disk.total / 2**30, 2),
            "free_gib": round(disk.free / 2**30, 2),
        },
        "gpu": gpu_report(),
        "nvcc": command(["nvcc", "--version"]),
        "torch": torch_report(),
        "git": command(["git", "rev-parse", "HEAD"]),
        "checkpoint": {"set": bool(checkpoint), "exists": bool(checkpoint and Path(checkpoint).exists())},
    }
    blockers: list[str] = []
    warnings: list[str] = []
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        blockers.append("Python 3.10 or 3.11 is required")
    devices = report["gpu"]["devices"]
    if not devices:
        blockers.append("nvidia-smi did not report a GPU")
    elif min(device["memory_total_mib"] for device in devices) < 20_000:
        warnings.append("less than 20 GiB GPU memory; lower max length or use LoRA")
    if report["disk"]["free_gib"] < 100:
        warnings.append("less than 100 GiB free disk; the full checkpoint matrix may fill the disk")
    if not checkpoint:
        warnings.append("SFT_CHECKPOINT is not set")
    elif not Path(checkpoint).exists():
        blockers.append("SFT_CHECKPOINT does not exist")
    report["blockers"] = blockers
    report["warnings"] = warnings
    report["environment_ready"] = not blockers
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
