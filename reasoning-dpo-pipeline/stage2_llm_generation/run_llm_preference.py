#!/usr/bin/env python3
"""
Stage 2: LLM-Assisted Preference Data Generator
================================================
Uses a large language model (DeepSeek-Chat by default) to generate high-quality
DPO rejected responses with semantic, human-like errors.

This complements Stage 1 (rule-based DAG corruption) by providing a second
preference dataset path: instead of algorithmic corruption, the LLM is asked to
produce rejection candidates that match specific error profiles:

  * ``computation_error``   — arithmetic/numerical mistake
  * ``dependency_mismatch`` — wrong variable/branch reference
  * ``missing_nodes``       — skip 2–3 critical reasoning steps

Features
--------
- **Resumable**: failed calls are cached; restarting picks up where it left off
- **Schema validation**: rejects malformed JSON before QC
- **Heuristic QC**: similarity, length-ratio, bad-marker, and signal checks
- **Per-error-type prompts**: each error profile has a dedicated system prompt
- **Token accounting**: reports total API token usage per run

Usage
-----
    python -m stage2_llm_generation.run_llm_preference \
        --input_path data/raw_solutions.json \
        --output_dir ./outputs/llm_preference \
        --error_types computation_error,dependency_mismatch,missing_nodes \
        --sample_count 500 \
        --model deepseek-chat \
        --api_key_env DEEPSEEK_API_KEY

Environment variables
---------------------
    DEEPSEEK_API_KEY   Your DeepSeek API key
    OPENAI_API_KEY     Alternative (OpenAI-compatible endpoint)

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
import argparse
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from difflib import SequenceMatcher

from openai import OpenAI

# ---------------------------------------------------------------------------
# Error-type system prompts
# ---------------------------------------------------------------------------

_PROMPTS = {
    "computation_error": """
You are creating DPO preference data for math word-problem reasoning.

You will receive: a problem, optionally a question, and a correct solution.

Your task: Create ONE rejected solution of type **computation_error**.

Requirements:
- Keep the reasoning structure almost identical to the correct solution
- Introduce ONLY one arithmetic/numerical mistake (e.g. change a number, wrong sign, wrong operation)
- Do NOT change any variable names, definitions, or logical steps
- The final answer MUST be wrong
- Keep wording, language, tone, and structure as close as possible

Return exactly:
{{"computation_error": {{"rejected": "..."}}}}
""".strip(),

    "dependency_mismatch": """
You are creating DPO preference data for math word-problem reasoning.

You will receive: a problem, optionally a question, and a correct solution.

Your task: Create ONE rejected solution of type **dependency_mismatch**.

Requirements:
- Keep text natural and plausible
- Introduce ONLY one wrong variable, wrong branch, or wrong referenced quantity
- Do NOT make it garbled, incomplete, or obviously broken
- The final answer should usually be wrong
- Keep wording, language, tone, and structure as close as possible

Return exactly:
{{"dependency_mismatch": {{"rejected": "..."}}}}
""".strip(),

    "missing_nodes": """
You are creating high-quality DPO preference data for math word-problem reasoning.

You will receive: a problem, optionally a question, and a correct solution.

Your task: Create ONE rejected solution of type **missing_nodes**.

CORE OBJECTIVE:
The rejected solution must contain a non-recoverable logical gap. A reader should NOT be able to fully reconstruct the reasoning without guessing.

Requirements:
- Remove or skip 2–3 CRITICAL reasoning steps that are necessary for solving
- The removed steps must break the dependency chain, not just shorten it
- Use an intermediate variable or result that was NEVER derived
- Skip the construction of a key equation but still use it
- Do NOT only merge steps or simplify expressions; the gap must make reasoning invalid
- Keep wording, language, tone, and structure almost identical

Return exactly:
{{"missing_nodes": {{"rejected": "..."}}}}
""".strip(),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LLMGenConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.3
    max_tokens: int = 1800
    max_retries: int = 3
    sleep_between_calls: float = 0.5
    sample_count: int = 0   # 0 = all
    seed: int = 42
    # QC
    min_rejected_len: int = 40
    max_similarity: float = 0.995
    min_similarity: float = 0.35


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def safe_strip(x) -> str:
    return "" if x is None else str(x).strip()


def stable_item_id(item: dict) -> str:
    if item.get("id") and str(item["id"]).strip():
        return str(item["id"]).strip()
    base = "\n".join([
        str(item.get("problem", "")).strip(),
        str(item.get("question", "")).strip(),
        str(item.get("solution", "")).strip(),
    ])
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def preview(text: str, max_len: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def robust_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first: last + 1])
        except Exception:
            pass
    raise ValueError("Cannot parse JSON from model output.")


def append_jsonl(path: str, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except json.JSONDecodeError:
        pass
    return rows


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def heuristic_check_rejected(
    chosen: str,
    rejected: str,
    error_type: str,
    cfg: LLMGenConfig,
) -> Tuple[bool, str]:
    """Return (pass, reason) for heuristic quality checks."""
    chosen = safe_strip(chosen)
    rejected = safe_strip(rejected)

    if not rejected:
        return False, "empty_rejected"
    if rejected == chosen:
        return False, "identical"
    if len(rejected) < cfg.min_rejected_len:
        return False, "too_short"
    if len(rejected) > len(chosen) * 1.3 and error_type == "missing_nodes":
        return False, "missing_nodes_too_long"

    sim = similarity(chosen, rejected)
    if sim > cfg.max_similarity:
        return False, "too_similar"
    if sim < cfg.min_similarity:
        return False, "too_different"

    lower = rejected.lower()
    bad_markers = ["here is the json", "error type", "rejected solution",
                   "chosen solution", "computation_error", "dependency_mismatch",
                   "missing_nodes"]
    for m in bad_markers:
        if m in lower:
            return False, "bad_marker"

    return True, "ok"


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def build_user_prompt(item: dict) -> str:
    parts = ["Problem:", safe_strip(item["problem"])]
    if safe_strip(item.get("question")):
        parts.extend(["", "Question:", safe_strip(item["question"])])
    parts.extend(["", "Correct solution:", safe_strip(item["solution"])])
    return "\n".join(parts)


def build_prompt_for_dpo(item: dict) -> str:
    problem = safe_strip(item.get("problem", ""))
    question = safe_strip(item.get("question", ""))
    user = f"Problem:\n{problem}\n\nQuestion:\n{question}" if question else f"Problem:\n{problem}"
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def call_llm_once(
    client: OpenAI,
    item: dict,
    error_type: str,
    cfg: LLMGenConfig,
) -> Tuple[Optional[dict], dict]:
    """Call the LLM once; returns (parsed_data, meta)."""
    user_prompt = build_user_prompt(item)
    system_prompt = _PROMPTS.get(error_type, "")

    meta = {
        "attempts": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    for attempt in range(1, cfg.max_retries + 1):
        info = {"attempt": attempt, "status": "unknown"}
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )

            content = resp.choices[0].message.content or ""
            info["raw_preview"] = preview(content)

            if hasattr(resp, "usage") and resp.usage:
                u = resp.usage
                meta["usage"]["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                meta["usage"]["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                meta["usage"]["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)

            data = robust_json_loads(content)
            # Basic schema check
            if error_type not in data:
                raise ValueError(f"Missing key: {error_type}")
            if "rejected" not in data[error_type]:
                raise ValueError(f"Missing rejected in {error_type}")
            if not isinstance(data[error_type]["rejected"], str):
                raise ValueError(f"rejected must be string")

            info["status"] = "ok"
            meta["attempts"].append(info)
            return data, meta

        except Exception as e:
            info["status"] = "failed"
            info["error"] = str(e)
            meta["attempts"].append(info)
            if attempt < cfg.max_retries:
                time.sleep(1.5 * attempt)

    return None, meta


# ---------------------------------------------------------------------------
# Per-error-type generation loop
# ---------------------------------------------------------------------------

def generate_for_type(
    client: OpenAI,
    data: List[dict],
    error_type: str,
    output_dir: str,
    cfg: LLMGenConfig,
    done_ids: set,
) -> Tuple[List[dict], Counter]:
    """
    Generate LLM-based rejected responses for one error type.
    Returns (samples, failure_counter).
    """
    random.seed(cfg.seed)
    samples: List[dict] = []
    failures: Counter = Counter()

    raw_path = os.path.join(output_dir, f"raw_{error_type}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    to_process = [x for x in data if stable_item_id(x) not in done_ids]

    for idx, item in enumerate(to_process, 1):
        item_id = stable_item_id(item)
        print(f"  [{error_type}] [{idx}/{len(to_process)}] {item_id[:12]}...")

        result, meta = call_llm_once(client, item, error_type, cfg)

        cache_entry = {"id": item_id, "item": item, "meta": meta}

        if result is None:
            failures[f"{error_type}:api_failed"] += 1
            cache_entry["status"] = "failed"
            append_jsonl(raw_path, cache_entry)
            print(f"    FAILED: {meta['attempts'][-1].get('error', 'unknown')}")
            time.sleep(cfg.sleep_between_calls)
            continue

        cache_entry["status"] = "ok"
        cache_entry["generation"] = result
        append_jsonl(raw_path, cache_entry)

        chosen = safe_strip(item.get("solution"))
        rejected = safe_strip(result[error_type]["rejected"])

        ok, reason = heuristic_check_rejected(chosen, rejected, error_type, cfg)
        if not ok:
            failures[f"{error_type}:{reason}"] += 1
            print(f"    QC FAILED: {reason}")
            continue

        sample = {
            "id": item_id,
            "template": item.get("template"),
            "op": item.get("op"),
            "mode": item.get("mode"),
            "length": item.get("length"),
            "d": item.get("d"),
            "error_type": error_type,
            "prompt": build_prompt_for_dpo(item),
            "chosen": chosen,
            "rejected": rejected,
            "source_problem": safe_strip(item.get("problem")),
            "source_question": safe_strip(item.get("question")),
        }
        samples.append(sample)
        print(f"    OK (n={len(samples)})")
        time.sleep(cfg.sleep_between_calls)

    return samples, failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 — LLM-assisted preference generation")
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--error_types",
                   default="computation_error,dependency_mismatch,missing_nodes")
    p.add_argument("--sample_count", type=int, default=0)
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--base_url", default="https://api.deepseek.com")
    p.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max_tokens", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = LLMGenConfig(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key or api_key == "YOUR_DEEPSEEK_API_KEY":
        raise RuntimeError(
            f"API key not found: set the {cfg.api_key_env} environment variable. "
            "Alternatively pass --api_key_env OPENAI_API_KEY with an OpenAI-compatible key."
        )

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    # Load source data
    print(f"Loading data from {args.input_path}...")
    if args.input_path.endswith(".jsonl"):
        data = load_jsonl(args.input_path)
    else:
        with open(args.input_path, encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a list of records.")
    data = [x for x in data
            if isinstance(x, dict)
            and safe_strip(x.get("problem"))
            and safe_strip(x.get("solution"))]
    if cfg.sample_count > 0 and cfg.sample_count < len(data):
        random.seed(cfg.seed)
        data = random.sample(data, cfg.sample_count)

    print(f"Using {len(data)} source records.")

    error_types = [e.strip() for e in args.error_types.split(",") if e.strip()]

    # Load done IDs from existing raw files (resumable)
    done_ids: set = set()
    for et in error_types:
        raw_path = os.path.join(args.output_dir, f"raw_{et}.jsonl")
        for row in load_jsonl(raw_path):
            if row.get("status") == "ok" and row.get("id"):
                done_ids.add(str(row["id"]))

    print(f"Already processed: {len(done_ids)}")

    all_samples: List[dict] = []
    total_failures: Counter = Counter()
    total_tokens = 0

    for et in error_types:
        print(f"\n--- Processing: {et} ---")
        samples, failures = generate_for_type(client, data, et, args.output_dir, cfg, done_ids)
        all_samples.extend(samples)
        total_failures.update(failures)
        for row in load_jsonl(os.path.join(args.output_dir, f"raw_{et}.jsonl")):
            if row.get("meta", {}).get("usage"):
                total_tokens += row["meta"]["usage"].get("total_tokens", 0)

    # Save
    random.seed(cfg.seed)
    random.shuffle(all_samples)

    mixed_path = os.path.join(args.output_dir, "llm_mixed.jsonl")
    save_jsonl(all_samples, mixed_path)
    print(f"\nMixed dataset: {len(all_samples)} samples → {mixed_path}")

    # QC report
    qc_report = {
        "config": asdict(cfg),
        "error_types": error_types,
        "total_samples": len(all_samples),
        "total_tokens": total_tokens,
        "failures": dict(total_failures),
        "by_type": {},
    }
    by_type: Dict[str, List[dict]] = {}
    for s in all_samples:
        by_type.setdefault(s["error_type"], []).append(s)
    for et, group in by_type.items():
        qc_report["by_type"][et] = {
            "n": len(group),
            "avg_chosen_len": round(sum(len(x["chosen"]) for x in group) / len(group), 1),
            "avg_rejected_len": round(sum(len(x["rejected"]) for x in group) / len(group), 1),
        }
    with open(os.path.join(args.output_dir, "qc_report.json"), "w", encoding="utf-8") as f:
        json.dump(qc_report, f, ensure_ascii=False, indent=2)

    print(f"\nTotal tokens used: {total_tokens:,}")
    print(f"Failures: {dict(total_failures)}")
    print("Done.")


if __name__ == "__main__":
    main()
