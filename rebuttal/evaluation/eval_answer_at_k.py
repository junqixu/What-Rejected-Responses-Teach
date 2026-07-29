from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    load_records,
    sha256_file,
    write_json,
    write_jsonl,
)


NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: Any) -> float | None:
    values = NUMBER.findall(str(text or "").replace(",", ""))
    if not values:
        return None
    try:
        return float(values[-1])
    except ValueError:
        return None


def prompt_for(row: dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return row["prompt"]
    problem = str(row.get("problem") or "").strip()
    question = str(row.get("question") or "").strip()
    return (
        "<|im_start|>user\n"
        f"Problem:\n{problem}\n\nQuestion:\n{question}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def summarize(results: list[dict[str, Any]], k_values: list[int]) -> dict[str, Any]:
    overall = Counter()
    bucket_hits: dict[str, Counter[int]] = defaultdict(Counter)
    bucket_totals = Counter(str(row.get("op", "unknown")) for row in results)
    for row in results:
        candidates = row["candidates"]
        bucket = str(row.get("op", "unknown"))
        for k in k_values:
            hit = any(bool(candidate["correct"]) for candidate in candidates[:k])
            row.setdefault("metrics", {})[f"answer@{k}"] = hit
            overall[k] += int(hit)
            bucket_hits[bucket][k] += int(hit)
    total = len(results)
    return {
        "n_prompts": total,
        "overall": {
            f"answer@{k}": (overall[k] / total if total else None)
            for k in k_values
        },
        "by_op": {
            bucket: {
                "n": count,
                **{
                    f"answer@{k}": bucket_hits[bucket][k] / count
                    for k in k_values
                },
            }
            for bucket, count in sorted(bucket_totals.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Answer@k on a fixed held-out source pool.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--data", default="data/rebuttal_v2/source/test_clean.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", default="1,8,16,32,128")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    add_common_cli_args(parser)
    args = parser.parse_args()
    configure_logging(args.log_level)
    rows = load_records(args.data, args.max_samples)
    k_values = sorted({int(value) for value in args.k.split(",") if value.strip()})
    if not rows or not k_values or k_values[0] <= 0:
        raise ValueError("evaluation requires data and positive k values")
    dry_report = {
        "n_prompts": len(rows),
        "k": k_values,
        "checkpoint": args.checkpoint,
        "data_hash": sha256_file(args.data),
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return
    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError("Answer@k requires vllm; install requirements-eval.txt") from error
    model = LLM(
        model=args.checkpoint,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        n=max(k_values),
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    generated = model.generate([prompt_for(row) for row in rows], sampling)
    results: list[dict[str, Any]] = []
    for row, output in zip(rows, generated, strict=True):
        gold = extract_answer(row.get("solution") or row.get("answer"))
        candidates = []
        for index, candidate in enumerate(output.outputs):
            predicted = extract_answer(candidate.text)
            candidates.append(
                {
                    "index": index,
                    "text": candidate.text,
                    "predicted_answer": predicted,
                    "correct": predicted is not None and gold is not None and abs(predicted - gold) < 1e-6,
                }
            )
        results.append(
            {
                "prompt_id": row.get("prompt_id"),
                "dag_family_id": row.get("dag_family_id"),
                "op": row.get("op"),
                "gold_answer": gold,
                "candidates": candidates,
            }
        )
    summary = summarize(results, k_values)
    summary.update(dry_report)
    summary["seed"] = args.seed
    summary["temperature"] = args.temperature
    summary["top_p"] = args.top_p
    output = ensure_output_dir(args.out, args.overwrite)
    write_jsonl(output / "answer_at_k_predictions.jsonl", results)
    write_json(output / "answer_at_k_metrics.json", summary)


if __name__ == "__main__":
    main()
