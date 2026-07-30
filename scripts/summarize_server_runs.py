from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize training and evaluation progress.")
    parser.add_argument("--root", default="outputs")
    args = parser.parse_args()
    root = Path(args.root)
    training = Counter()
    for path in root.rglob("run_manifest.json") if root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            training[str(payload.get("status", "unknown"))] += 1
        except (OSError, json.JSONDecodeError):
            training["invalid_manifest"] += 1
    diagnostic_files = list(root.rglob("diagnostic_metrics.json")) if root.exists() else []
    answer_files = list(root.rglob("answer_at_k_metrics.json")) if root.exists() else []
    print(
        json.dumps(
            {
                "training_runs": dict(sorted(training.items())),
                "diagnostic_results": len(diagnostic_files),
                "answer_at_k_results": len(answer_files),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
