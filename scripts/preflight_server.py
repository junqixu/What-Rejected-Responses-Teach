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
    """Hash text datasets with canonical LF line endings across OS checkouts."""
    digest = hashlib.sha256()
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    digest.update(content)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_checkpoint(path: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    weights = sorted(path.glob("*.safetensors")) + sorted(path.glob("pytorch_model*.bin"))
    required = (path / "config.json", path / "tokenizer_config.json")
    if not path.is_dir():
        return {"path": str(path), "exists": False}, [f"SFT checkpoint does not exist: {path}"]
    for required_path in required:
        if not required_path.exists():
            blockers.append(f"SFT checkpoint is missing {required_path.name}: {path}")
    if not weights:
        blockers.append(f"SFT checkpoint has no model weights: {path}")
    manifest_path = path / "run_manifest.json"
    manifest_status = None
    if manifest_path.exists():
        try:
            manifest_status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except (json.JSONDecodeError, OSError) as error:
            blockers.append(f"cannot read SFT run manifest: {error}")
        if manifest_status != "complete":
            blockers.append(f"SFT run manifest status is not complete: {manifest_status!r}")
    return {
        "path": str(path),
        "exists": True,
        "weight_files": [item.name for item in weights],
        "manifest_status": manifest_status,
    }, blockers


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
        else:
            checkpoint_report, checkpoint_blockers = check_checkpoint(Path(args.checkpoint))
            report["checkpoint"] = checkpoint_report
            blockers.extend(checkpoint_blockers)
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
