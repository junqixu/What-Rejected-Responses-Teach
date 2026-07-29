from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, write_json, write_jsonl


def _axis(row: dict[str, Any]) -> str:
    if row.get("eval_axis"):
        return str(row["eval_axis"])
    labels = row.get("error_labels")
    if isinstance(labels, list) and labels:
        return "+".join(sorted(str(label) for label in labels))
    return str(row.get("error_type", "unknown"))


def _score_response(model: Any, tokenizer: Any, prompt: str, response: str, max_length: int) -> tuple[float, int]:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + response_ids)[:max_length]
    response_start = min(len(prompt_ids), len(input_ids))
    if len(input_ids) < 2 or response_start >= len(input_ids):
        raise ValueError("prompt truncation left no response tokens to score")
    tensor = torch.tensor([input_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids=tensor).logits[:, :-1, :]
        targets = tensor[:, 1:]
        token_logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask_start = max(response_start - 1, 0)
    selected = token_logps[0, mask_start:]
    return float(selected.sum().item()), int(selected.numel())


def score_rows(rows: list[dict[str, Any]], checkpoint: str, tokenizer_path: str, max_length: int) -> list[dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("model scoring requires torch and transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    scored: list[dict[str, Any]] = []
    for row in rows:
        chosen_logp, chosen_count = _score_response(model, tokenizer, row["prompt"], row["chosen"], max_length)
        rejected_logp, rejected_count = _score_response(model, tokenizer, row["prompt"], row["rejected"], max_length)
        packed = dict(row)
        packed.update(
            {
                "eval_axis": _axis(row),
                "chosen_logp": chosen_logp,
                "rejected_logp": rejected_logp,
                "chosen_response_tokens": chosen_count,
                "rejected_response_tokens": rejected_count,
            }
        )
        scored.append(packed)
    return scored


def aggregate(scored: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        chosen_count = int(row.get("chosen_response_tokens", row.get("chosen_tokens", 0)))
        rejected_count = int(row.get("rejected_response_tokens", row.get("rejected_tokens", 0)))
        if chosen_count <= 0 or rejected_count <= 0:
            raise ValueError("scored rows require positive chosen/rejected token counts")
        row["raw_margin"] = float(row["chosen_logp"]) - float(row["rejected_logp"])
        row["length_normalized_margin"] = float(row["chosen_logp"]) / chosen_count - float(row["rejected_logp"]) / rejected_count
        row["diagnostic_correct"] = row["length_normalized_margin"] > 0
        groups[_axis(row)].append(row)

    by_axis: dict[str, Any] = {}
    for axis, rows in sorted(groups.items()):
        by_axis[axis] = {
            "n": len(rows),
            "length_normalized_diagnostic_accuracy": mean(float(row["diagnostic_correct"]) for row in rows),
            "mean_raw_margin": mean(float(row["raw_margin"]) for row in rows),
            "mean_length_normalized_margin": mean(float(row["length_normalized_margin"]) for row in rows),
        }
    singleton = [value["length_normalized_diagnostic_accuracy"] for key, value in by_axis.items() if "+" not in key]
    return {
        "n_samples": len(scored),
        "by_axis": by_axis,
        "mean_axis": mean(singleton) if singleton else None,
        "worst_axis": min(singleton) if singleton else None,
        "axis_imbalance": abs(singleton[0] - singleton[1]) if len(singleton) == 2 else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or aggregate isolated structural diagnostic preferences.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eval_data", nargs="+")
    source.add_argument("--predictions", nargs="+")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--out", required=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    paths = args.predictions or args.eval_data
    rows = load_records(paths, args.max_samples)
    required = ("prompt", "chosen", "rejected")
    for index, row in enumerate(rows):
        if any(not isinstance(row.get(field), str) or not row[field] for field in required):
            raise ValueError(f"row {index} is missing prompt/chosen/rejected text")
    if args.dry_run:
        print(json.dumps({"n_samples": len(rows), "axes": sorted({_axis(row) for row in rows})}, indent=2))
        return
    if args.predictions:
        scored = rows
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required with --eval_data")
        scored = score_rows(rows, args.checkpoint, args.tokenizer, args.max_length)
    report = aggregate(scored)
    output = ensure_output_dir(args.out, args.overwrite)
    write_jsonl(output / "diagnostic_predictions.jsonl", scored)
    _write_csv(output / "diagnostic_predictions.csv", scored)
    write_json(output / "diagnostic_metrics.json", report)


if __name__ == "__main__":
    main()
