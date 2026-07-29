from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, write_json


SEARCHES = {
    "dpo_trainers": "DPOTrainer",
    "checkpoint_loaders": "from_pretrained",
    "answer_passk": "pass@",
    "reasoning_mentions": "Reasoning",
    "computation_mentions": "computation_error",
    "dependency_mentions": "dependency_mismatch",
    "missing_mentions": "missing_nodes",
    "disorder_mentions": "disorder",
}


def _source_files(root: Path) -> list[Path]:
    files = list(root.glob("*.py"))
    for directory in ("dpo", "ppo", "rebuttal"):
        source_root = root / directory
        if source_root.exists():
            files.extend(source_root.rglob("*.py"))
    return sorted({path for path in files if "__pycache__" not in path.parts})


def _files_containing(root: Path, needle: str) -> list[str]:
    matches: list[str] = []
    for path in _source_files(root):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if needle in content:
            matches.append(str(path.relative_to(root)))
    return matches


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def audit(root: Path) -> dict[str, Any]:
    return {
        "repo_root": str(root.resolve()),
        "git_commit": _git_commit(root),
        "git_metadata_present": (root / ".git").exists(),
        "python_files": len(_source_files(root)),
        "entry_points": sorted(str(path.relative_to(root)) for path in root.glob("*.py")),
        "searches": {name: _files_containing(root, needle) for name, needle in SEARCHES.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit repository paths relevant to rebuttal experiments.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--out", default="rebuttal/repo_audit_runtime.json")
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    report = audit(Path(args.repo_root))
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
