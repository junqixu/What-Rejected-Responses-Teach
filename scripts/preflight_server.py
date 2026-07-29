from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.common import iter_records
from rebuttal.generation.package_release import SPLITS, validate_complete_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_data(data_root: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    report: dict[str, Any] = {}
    manifest_path = data_root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    expected_hashes = manifest.get("output_hashes", {}) if isinstance(manifest, dict) else {}
    for split in SPLITS:
        path = data_root / f"{split}.jsonl"
        if not path.exists():
            blockers.append(f"missing dataset: {path}")
            continue
        rows, problems = validate_complete_split(iter_records(path), split)
        actual_hash = _sha256(path)
        expected_hash = expected_hashes.get(path.name)
        if expected_hash and expected_hash != actual_hash:
            problems.append("SHA256 does not match dataset_manifest.json")
        prompts = {str(row["prompt_id"]) for row in rows}
        report[split] = {
            "records": len(rows),
            "prompts": len(prompts),
            "sha256": actual_hash,
            "problems": problems,
        }
        blockers.extend(f"{split}: {problem}" for problem in problems)
    return report, blockers


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate release data, dependencies, GPU, and SFT checkpoint.")
    parser.add_argument("--data_root", default="data/rebuttal_release")
    parser.add_argument("--checkpoint", default=os.getenv("SFT_CHECKPOINT"))
    parser.add_argument("--data_only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_report, blockers = check_data(Path(args.data_root))
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data": data_report,
        "packages": {
            name: _version(name)
            for name in ("torch", "transformers", "datasets", "trl", "accelerate", "bitsandbytes")
        },
    }
    if not args.data_only:
        if not args.checkpoint:
            blockers.append("SFT checkpoint is not set; pass --checkpoint or set SFT_CHECKPOINT")
        elif not Path(args.checkpoint).exists():
            blockers.append(f"SFT checkpoint does not exist: {args.checkpoint}")
        try:
            import torch
        except ImportError:
            blockers.append("torch is not installed")
        else:
            report["cuda"] = {
                "available": torch.cuda.is_available(),
                "runtime": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
                "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            }
            if not torch.cuda.is_available():
                blockers.append("CUDA is unavailable")
            elif not torch.cuda.is_bf16_supported():
                blockers.append("the selected GPU does not report bfloat16 support")
        for package in ("transformers", "datasets", "trl", "accelerate", "bitsandbytes"):
            if report["packages"][package] is None:
                blockers.append(f"missing package: {package}")
    report["blockers"] = blockers
    report["ready"] = not blockers
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
