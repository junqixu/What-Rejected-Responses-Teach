#!/usr/bin/env python3
"""
Stage 4: vLLM Evaluation Engine
=================================
Fast batch inference and evaluation pipeline using vLLM,
with multi-level reasoning chain metrics.

Metrics computed
----------------
1. **answer-pass@k**       — does any of the k samples have the correct final number?
2. **reasoning-chain-pass@k** — does any of the k samples have a structurally correct
   reasoning chain (measured via DAG F1 on extracted variable dependencies)?
3. **answer-reasoning gap** — answer-pass@128 − reasoning-chain-pass@128
   (captures "correct-by-luck" vs "genuinely correct reasoning")
4. **generalization-decay-rate** — relative performance drop on harder OP buckets
   vs the easiest bucket (op_2_10)

Per-sample scoring
------------------
The model generates k candidates per problem. For each candidate:
- **Answer correctness**: extract last number, compare to ground truth (±1e-6)
- **Reasoning chain correctness**: 
    (a) Extract variable-dependency DAG from the generated text
    (b) Compute edge-F1, node-F1, entity-F1, raw-assignment-F1
    (c) Extract strict mathematical equation steps as fallback
    (d) Score = max(graph_score, equation_score); threshold = 0.35

Aggregation levels: overall → OP bucket (op_2_10, op_11_14, op_15_20) → individual OP

Usage
-----
    python -m stage4_evaluation.vllm_eval.run_eval \
        --model_path ./checkpoints/dpo_full_mix \
        --data_path ./data/half/validation.json \
        --output_dir ./results/eval_dpo_full_mix \
        --k_list 1 2 4 8 16 32 64 128

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import json
import argparse
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

from datasets import load_dataset
from vllm import LLM, SamplingParams

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOKENIZER_PATH = "Qwen/Qwen2-0.5B"
K_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
K_MAX = max(K_LIST)

MAX_TOKENS = 1024
TEMPERATURE = 0.65
TOP_P = 0.95

TEST_SIZE = 0.10   # fraction of data held out for eval
SEED = 30
EPS = 1e-6
GRAPH_MATCH_THRESHOLD = 0.35

CHATML_STOP_TOKEN_ID = 151645  # <|im_end|> for Qwen2 ChatML


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    model_path: str
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
    data_path: str = ""
    output_dir: str = "./eval_output"
    k_list: List[int] = field(default_factory=lambda: K_LIST)
    max_tokens: int = MAX_TOKENS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    test_size: float = TEST_SIZE
    seed: int = SEED
    eps: float = EPS
    graph_threshold: float = GRAPH_MATCH_THRESHOLD
    gpu_memory_utilization: float = 0.90


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def extract_last_number(text: str) -> Optional[float]:
    """Extract the last numeric value from text."""
    if text is None:
        return None
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_reasoning_text(text: str) -> str:
    """Drop everything after 'final answer' / 'answer:' to isolate reasoning."""
    text = str(text or "").strip()
    text = re.split(r"(?:final\s*answer|answer)\s*[:：]", text, flags=re.IGNORECASE)[0]
    return normalize_text(text)


def normalize_entity_name(name: str) -> str:
    name = normalize_text(name)
    name = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", name)
    return name


def split_reasoning_clauses(text: str) -> List[str]:
    text = str(text or "")
    text = re.sub(r"(?<!\s)(Define|Compute|State|Solve|Simplifying|Move|Isolate|Divide)\b", r" \1", text)
    parts = re.split(r"[.;\n]+", text)
    return [normalize_text(p) for p in parts if normalize_text(p)]


# ---------------------------------------------------------------------------
# Variable-dependency graph
# ---------------------------------------------------------------------------

def extract_variable_dependency_graph(text: str) -> Dict[str, Any]:
    """
    Parse a reasoning text into a variable-dependency graph.

    Returns
    -------
    dict with keys:
        edges           : set of (dependency, variable) tuples
        nodes           : set of node identifiers
        entity_to_var   : {entity_name: var_name}
        entity_nodes    : set of entity nodes
        raw_assignment_edges : fallback edges from direct variable assignments
    """
    text = str(text or "")
    entity_to_var: Dict[str, str] = {}
    var_to_entity: Dict[str, str] = {}
    reasoning_text = extract_reasoning_text(text)
    clauses = split_reasoning_clauses(reasoning_text)

    define_patterns = [
        re.compile(r"define\s+(.+?)\s+as\s+([A-Za-z][A-Za-z0-9_]*)\b", flags=re.IGNORECASE),
        re.compile(r"define\s+([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", flags=re.IGNORECASE),
    ]

    for clause in clauses:
        for pat in define_patterns:
            m = pat.search(clause)
            if m:
                if len(m.groups()) == 2:
                    entity, var = m.groups()
                    en = normalize_entity_name(entity)
                    if en:
                        entity_to_var[en] = var
                        var_to_entity[var] = en

    assignments: Dict[str, set] = {}
    raw_assignment_edges: set = set()
    RESERVED = {
        "define", "so", "answer", "final", "state", "compute", "solve",
        "simplifying", "move", "isolate", "divide", "therefore", "thus",
        "const",
    }

    for lhs, rhs in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^.;\n]+)", reasoning_text):
        dep_vars = set(re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)\b", rhs, flags=re.IGNORECASE))
        dep_vars.discard(lhs)
        dep_vars = {
            v for v in dep_vars
            if v.lower() not in RESERVED
        }
        if dep_vars:
            assignments[lhs] = dep_vars
            for dep in dep_vars:
                raw_assignment_edges.add((lhs, dep))
        elif re.search(r"-?\d+(?:\.\d+)?", rhs):
            assignments[lhs] = {"CONST"}
            raw_assignment_edges.add((lhs, "CONST"))

    def resolve_var(var: str, visited: Optional[set] = None) -> set:
        visited = visited or set()
        if var in visited:
            return set()
        if var in var_to_entity:
            return {f"ENTITY::{var_to_entity[var]}"}
        if var == "CONST":
            return {"CONST"}
        if var not in assignments:
            return set()
        resolved = set()
        for dep in assignments[var]:
            if dep == "CONST":
                resolved.add("CONST")
            elif dep in var_to_entity:
                resolved.add(f"ENTITY::{var_to_entity[dep]}")
            else:
                resolved |= resolve_var(dep, visited | {var})
        return resolved

    edges: set = set()
    nodes: set = set()
    entity_nodes = {f"ENTITY::{e}" for e in entity_to_var.keys()}

    for entity_name, var in entity_to_var.items():
        lhs = f"ENTITY::{entity_name}"
        nodes.add(lhs)
        for dep in resolve_var(var):
            edges.add((lhs, dep))
            nodes.add(dep)

    # Fallback to raw assignment graph if no entity graph found
    if not edges and raw_assignment_edges:
        edges = {
            (f"VAR::{lhs}", f"VAR::{rhs}" if rhs != "CONST" else "CONST")
            for lhs, rhs in raw_assignment_edges
        }
        nodes = set()
        for lhs, rhs in edges:
            nodes.add(lhs)
            nodes.add(rhs)

    return {
        "edges": edges,
        "nodes": nodes,
        "entity_to_var": entity_to_var,
        "entity_nodes": entity_nodes,
        "raw_assignment_edges": raw_assignment_edges,
    }


# ---------------------------------------------------------------------------
# Equation extraction
# ---------------------------------------------------------------------------

def extract_equations(text: str) -> set:
    """
    Extract strict mathematical equation steps: e.g. "7 + 3 = 10".
    Respects commutativity for + and *.
    """
    pattern = r"(\d+(?:\.\d+)?)\s*([\+\-\*\/x×÷])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, str(text).lower())
    eq_set = set()
    for m in matches:
        try:
            a, op, b, c = float(m[0]), m[1], float(m[2]), float(m[3])
            if op in ['x', '×']:
                op = '*'
            elif op == '÷':
                op = '/'
            if op in ['+', '*']:
                eq_set.add((min(a, b), op, max(a, b), c))
            else:
                eq_set.add((a, op, b, c))
        except Exception:
            pass
    return eq_set


# ---------------------------------------------------------------------------
# F1 scoring
# ---------------------------------------------------------------------------

def f1_from_sets(pred_items: set, gold_items: set) -> float:
    if not gold_items and not pred_items:
        return 1.0
    if not gold_items or not pred_items:
        return 0.0
    overlap = len(pred_items & gold_items)
    precision = overlap / max(1, len(pred_items))
    recall = overlap / max(1, len(gold_items))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Reasoning chain scoring
# ---------------------------------------------------------------------------

def compute_reasoning_chain_score(pred_text: str, gold_solution: str) -> Tuple[float, float, float]:
    """
    Score the quality of the predicted reasoning chain vs gold.

    Returns
    -------
    (final_score, graph_score, equation_score)
        final_score = max(graph_score, equation_score)
        graph_score  = 0.45*edge_F1 + 0.20*node_F1 + 0.20*entity_F1 + 0.15*raw_edge_F1
        equation_score = F1 of extracted equation steps
    """
    pred_graph = extract_variable_dependency_graph(pred_text)
    gold_graph = extract_variable_dependency_graph(gold_solution)

    edge_f1 = f1_from_sets(pred_graph["edges"], gold_graph["edges"])
    node_f1 = f1_from_sets(pred_graph["nodes"], gold_graph["nodes"])
    entity_f1 = f1_from_sets(pred_graph["entity_nodes"], gold_graph["entity_nodes"])
    raw_edge_f1 = f1_from_sets(
        pred_graph["raw_assignment_edges"],
        gold_graph["raw_assignment_edges"],
    )
    graph_score = 0.45 * edge_f1 + 0.20 * node_f1 + 0.20 * entity_f1 + 0.15 * raw_edge_f1

    pred_eqs = extract_equations(pred_text)
    gold_eqs = extract_equations(gold_solution)
    equation_score = f1_from_sets(pred_eqs, gold_eqs)

    final_score = max(graph_score, equation_score)
    return final_score, graph_score, equation_score


def is_reasoning_chain_correct(
    pred_text: str,
    gold_solution: str,
    true_answer: float,
    threshold: float = GRAPH_MATCH_THRESHOLD,
) -> Tuple[bool, float]:
    """
    A reasoning chain is correct if:
    1. The final answer is numerically correct
    2. The reasoning score >= threshold
    """
    pred_num = extract_last_number(pred_text)
    final_correct = abs(pred_num - true_answer) < EPS if pred_num is not None else False
    chain_score, _, _ = compute_reasoning_chain_score(pred_text, gold_solution)
    return final_correct and chain_score >= threshold, chain_score


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(example: dict) -> str:
    problem = str(example.get("problem", "")).strip()
    question = str(example.get("question", "")).strip()
    prompt = (
        "<|im_start|>system\n"
        "You are a helpful and logical assistant. "
        "Please show your complete reasoning process step by step.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Problem:\n{problem}\n\n"
        f"Question:\n{question}\n\n"
        "Please reason step by step, and show your complete reasoning process.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def get_ground_truth(example: dict) -> Optional[float]:
    for key in ["answer", "solution"]:
        if key in example and example[key] is not None:
            num = extract_last_number(str(example[key]))
            if num is not None:
                return num
    return None


def get_op(example: dict) -> Optional[int]:
    try:
        return int(example["op"])
    except Exception:
        return None


def get_op_bucket(op: Optional[int]) -> str:
    if op is None:
        return "unknown"
    if 2 <= op <= 10:
        return "op_2_10"
    elif 11 <= op <= 14:
        return "op_11_14"
    elif 15 <= op <= 20:
        return "op_15_20"
    else:
        return f"op_other_{op}"


# ---------------------------------------------------------------------------
# vLLM inference + scoring
# ---------------------------------------------------------------------------

def run_inference(
    llm: LLM,
    prompts: List[str],
    k: int,
    sampling_params: SamplingParams,
) -> List[Any]:
    """Run vLLM inference and return outputs."""
    params = SamplingParams(
        n=k,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop_token_ids=[CHATML_STOP_TOKEN_ID],
    )
    return llm.generate(prompts, params)


# ---------------------------------------------------------------------------
# Evaluation aggregator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Per-sample evaluation result."""
    index: int
    problem: str
    question: str
    solution: str
    gold_answer: Optional[float]
    op: Optional[int]
    bucket: str
    candidates: List[Dict] = field(default_factory=list)
    passk: Dict[str, bool] = field(default_factory=dict)
    chain_passk: Dict[str, bool] = field(default_factory=dict)
    top1_chain_correct: bool = False
    top1_chain_score: float = 0.0


def evaluate_model(
    cfg: EvalConfig,
    verbose: bool = True,
) -> Tuple[List[EvalResult], dict]:
    """
    Run the full evaluation pipeline: load model → load data → infer → score → aggregate.
    """
    # ---- Load model ----
    if verbose:
        print(f"[INFO] Loading model: {cfg.model_path}")
    llm = LLM(
        model=cfg.model_path,
        tokenizer=cfg.tokenizer_path,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        disable_log_stats=True,
        enforce_eager=True,
    )
    if verbose:
        print("[INFO] Model loaded.")

    # ---- Load data ----
    if verbose:
        print(f"[INFO] Loading data: {cfg.data_path}")
    ds = load_dataset("json", data_files=cfg.data_path)
    ds = ds["train"].train_test_split(test_size=cfg.test_size, seed=cfg.seed)
    test_ds = ds["test"]
    if verbose:
        print(f"[INFO] Test set: {len(test_ds)} samples")

    # ---- Build prompts and metadata ----
    prompts: List[str] = []
    truths: List[Optional[float]] = []
    metas: List[dict] = []
    bucket_counts: Dict[str, int] = defaultdict(int)

    for idx, example in enumerate(test_ds):
        prompts.append(build_prompt(example))
        truths.append(get_ground_truth(example))
        op = get_op(example)
        bucket = get_op_bucket(op)
        bucket_counts[bucket] += 1
        metas.append({
            "index": idx,
            "problem": example.get("problem", ""),
            "question": example.get("question", ""),
            "solution": example.get("solution", ""),
            "gold_answer": truths[-1],
            "op": op,
            "bucket": bucket,
        })

    if verbose:
        print(f"[INFO] Bucket distribution:")
        for b, c in sorted(bucket_counts.items()):
            print(f"  {b}: {c}")

    # ---- Batch inference ----
    sampling_params = SamplingParams(
        n=cfg.k_max,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        stop_token_ids=[CHATML_STOP_TOKEN_ID],
    )
    if verbose:
        print(f"[INFO] Generating {cfg.k_max} candidates per sample...")
    outputs = llm.generate(prompts, sampling_params)

    # ---- Score each sample ----
    all_results: List[EvalResult] = []
    total = len(test_ds)

    for i, out in enumerate(outputs):
        meta = metas[i]
        bucket = meta["bucket"]
        op_val = meta["op"]
        gold_answer = truths[i]
        candidate_results: List[dict] = []

        for cand_idx, cand in enumerate(out.outputs):
            pred_text = cand.text.strip()
            pred_num = extract_last_number(pred_text)
            correct = abs(pred_num - gold_answer) < cfg.eps if pred_num is not None and gold_answer is not None else False
            chain_correct, chain_score = is_reasoning_chain_correct(
                pred_text, meta["solution"], gold_answer, cfg.graph_threshold,
            ) if gold_answer is not None else (False, 0.0)

            pred_graph = extract_variable_dependency_graph(pred_text)
            gold_graph = extract_variable_dependency_graph(meta["solution"])

            candidate_results.append({
                "candidate_id": cand_idx,
                "text": pred_text,
                "pred_number": pred_num,
                "is_correct": correct,
                "reasoning_chain_score": chain_score,
                "reasoning_chain_correct": chain_correct,
                "pred_graph_edges": sorted([list(e) for e in pred_graph["edges"]]),
                "gold_graph_edges": sorted([list(e) for e in gold_graph["edges"]]),
            })

        per_sample_passk = {
            f"pass@{k}": any(c["is_correct"] for c in candidate_results[:k])
            for k in cfg.k_list
        }
        per_sample_chain_passk = {
            f"chain-pass@{k}": any(c["reasoning_chain_correct"] for c in candidate_results[:k])
            for k in cfg.k_list
        }

        result = EvalResult(
            index=meta["index"],
            problem=meta["problem"],
            question=meta["question"],
            solution=meta["solution"],
            gold_answer=gold_answer,
            op=op_val,
            bucket=bucket,
            candidates=candidate_results,
            passk=per_sample_passk,
            chain_passk=per_sample_chain_passk,
            top1_chain_correct=candidate_results[0]["reasoning_chain_correct"] if candidate_results else False,
            top1_chain_score=candidate_results[0]["reasoning_chain_score"] if candidate_results else 0.0,
        )
        all_results.append(result)

    # ---- Aggregate ----
    summary = aggregate_results(all_results, bucket_counts, cfg)

    if verbose:
        print_summary(summary, cfg)

    return all_results, summary


def aggregate_results(
    results: List[EvalResult],
    bucket_counts: Dict[str, int],
    cfg: EvalConfig,
) -> dict:
    """Aggregate per-sample results into overall / by-bucket / by-op summaries."""
    k_list = cfg.k_list
    total = len(results)

    overall_correct = {k: 0 for k in k_list}
    overall_chain_correct = {k: 0 for k in k_list}

    bucket_correct = defaultdict(lambda: {k: 0 for k in k_list})
    bucket_chain_correct = defaultdict(lambda: {k: 0 for k in k_list})
    bucket_total: Dict[str, int] = defaultdict(int)
    bucket_chain_top1_correct = defaultdict(int)

    op_correct = defaultdict(lambda: {k: 0 for k in k_list})
    op_chain_correct = defaultdict(lambda: {k: 0 for k in k_list})
    op_total: Dict[str, int] = defaultdict(int)

    top1_chain_total = 0

    for r in results:
        b = r.bucket
        op_key = str(r.op) if r.op is not None else "unknown"
        bucket_total[b] += 1
        if r.op is not None:
            op_total[op_key] += 1
        if r.gold_answer is not None:
            top1_chain_total += 1 if r.top1_chain_correct else 0

        for k in k_list:
            if r.passk.get(f"pass@{k}"):
                overall_correct[k] += 1
                bucket_correct[b][k] += 1
                if r.op is not None:
                    op_correct[op_key][k] += 1

            if r.chain_passk.get(f"chain-pass@{k}"):
                overall_chain_correct[k] += 1
                bucket_chain_correct[b][k] += 1
                if r.op is not None:
                    op_chain_correct[op_key][k] += 1

        if r.gold_answer is not None:
            bucket_chain_top1_correct[b] += 1 if r.top1_chain_correct else 0

    def build_overall():
        out = {}
        for k in k_list:
            out[f"answer-pass@{k}"] = overall_correct[k] / total if total > 0 else 0.0
            out[f"reasoning-chain-pass@{k}"] = overall_chain_correct[k] / total if total > 0 else 0.0
        out["reasoning-chain-accuracy"] = top1_chain_total / max(top1_chain_total, 1) if top1_chain_total > 0 else 0.0
        return out

    def build_by_bucket():
        out = {}
        for b in sorted(bucket_total.keys()):
            n = bucket_total[b]
            d = {"num_samples": n}
            for k in k_list:
                d[f"answer-pass@{k}"] = bucket_correct[b][k] / n if n > 0 else 0.0
                d[f"reasoning-chain-pass@{k}"] = bucket_chain_correct[b][k] / n if n > 0 else 0.0
            d["reasoning-chain-accuracy"] = bucket_chain_top1_correct[b] / n if n > 0 else 0.0
            out[b] = d
        return out

    def build_by_op():
        out = {}
        for op_key in sorted(op_total.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            n = op_total[op_key]
            d = {"num_samples": n}
            for k in k_list:
                d[f"answer-pass@{k}"] = op_correct[op_key][k] / n if n > 0 else 0.0
                d[f"reasoning-chain-pass@{k}"] = op_chain_correct[op_key][k] / n if n > 0 else 0.0
            out[op_key] = d
        return out

    # Generalization decay rate
    easy = build_by_bucket().get("op_2_10", {}).get("answer-pass@1", None)
    generalization_decay = None
    if easy and easy > 0:
        weighted_hard_score = 0.0
        hard_count = 0
        for b in ["op_11_14", "op_15_20"]:
            if b in bucket_total:
                n = bucket_total[b]
                s = build_by_bucket()[b]["answer-pass@1"]
                weighted_hard_score += s * n
                hard_count += n
        if hard_count > 0:
            avg_hard = weighted_hard_score / hard_count
            generalization_decay = max(0.0, (easy - avg_hard) / easy)

    summary = {
        "overall": build_overall(),
        "by_bucket": build_by_bucket(),
        "by_op": build_by_op(),
        "generalization_decay_rate": generalization_decay,
        "bucket_counts": dict(bucket_counts),
        "num_test_samples": total,
    }
    return summary


def print_summary(summary: dict, cfg: EvalConfig) -> None:
    print("\n" + "=" * 70)
    print("OVERALL PASS@K")
    for k in cfg.k_list:
        ans = summary["overall"][f"answer-pass@{k}"]
        chn = summary["overall"][f"reasoning-chain-pass@{k}"]
        print(f"  pass@{k}: {ans:.2%}  |  chain-pass@{k}: {chn:.2%}")
    print(f"  reasoning-chain-accuracy@1: {summary['overall']['reasoning-chain-accuracy']:.2%}")

    if summary.get("generalization_decay_rate") is not None:
        print(f"  generalization-decay-rate: {summary['generalization_decay_rate']:.2%}")

    print("\nBY BUCKET")
    for bucket, d in summary["by_bucket"].items():
        print(f"\n  [{bucket}] n={d['num_samples']}")
        for k in [1, 8, 32, 128]:
            if k in cfg.k_list:
                print(f"    pass@{k}: {d[f'answer-pass@{k}']:.2%}  |  chain: {d[f'reasoning-chain-pass@{k}']:.2%}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_markdown_report(results: List[EvalResult], summary: dict, cfg: EvalConfig) -> str:
    """Write a detailed per-sample markdown report."""
    lines = [
        "# Evaluation Report",
        f"Model: `{cfg.model_path}`",
        f"Tokenizer: `{cfg.tokenizer_path}`",
        f"Test samples: {cfg.num_test_samples}",
        "",
    ]
    for r in results:
        lines += [
            f"## Sample {r.index} (bucket={r.bucket}, op={r.op})",
            f"- Problem: {r.problem[:100]}...",
            f"- Gold answer: `{r.gold_answer}`",
            f"- Top-1 chain correct: `{r.top1_chain_correct}` (score={r.top1_chain_score:.3f})",
        ]
        for cand in r.candidates[:2]:
            lines += [
                f"  - Candidate `{cand['candidate_id']}`: pred={cand['pred_number']} "
                f"correct={cand['is_correct']} chain_score={cand['reasoning_chain_score']:.3f}",
                f"    `{cand['text'][:150]}...`",
            ]
        lines.append("")

    report_path = os.path.join(cfg.output_dir, "full_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def save_results(
    results: List[EvalResult],
    summary: dict,
    cfg: EvalConfig,
) -> Tuple[str, str]:
    """Save JSON summary and detailed results."""
    summary_payload = {
        **asdict(cfg),
        "metrics": summary,
    }
    # Remove non-serialisable fields
    for key in ["k_list"]:
        if key in summary_payload:
            summary_payload[key] = list(summary_payload[key])

    summary_path = os.path.join(cfg.output_dir, "summary_by_op.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    detail_data = [
        {
            "index": r.index,
            "problem": r.problem,
            "question": r.question,
            "solution": r.solution,
            "gold_answer": r.gold_answer,
            "op": r.op,
            "bucket": r.bucket,
            "candidates": r.candidates,
            "passk": r.passk,
            "chain_passk": r.chain_passk,
            "top1_chain_correct": r.top1_chain_correct,
            "top1_chain_score": r.top1_chain_score,
        }
        for r in results
    ]
    detail_path = os.path.join(cfg.output_dir, "detailed_results_by_op.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail_data, f, ensure_ascii=False, indent=2)

    report_path = write_markdown_report(results, summary, cfg)

    return summary_path, detail_path, report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 4 — vLLM Evaluation")
    p.add_argument("--model_path", required=True, help="Path to the model to evaluate")
    p.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    p.add_argument("--data_path", required=True, help="Path to validation .json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k_list", type=str, default="1,2,4,8,16,32,64,128")
    p.add_argument("--max_tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top_p", type=float, default=TOP_P)
    p.add_argument("--test_size", type=float, default=TEST_SIZE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    k_max = max(k_list)

    cfg = EvalConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        k_list=k_list,
        k_max=k_max,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        test_size=args.test_size,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print("=" * 70)
    print("Stage 4 — vLLM Evaluation")
    print(f"  Model   : {cfg.model_path}")
    print(f"  Data    : {cfg.data_path}")
    print(f"  Output  : {cfg.output_dir}")
    print(f"  k_max   : {cfg.k_max}")
    print("=" * 70)

    results, summary = evaluate_model(cfg, verbose=True)

    summary_path, detail_path, report_path = save_results(results, summary, cfg)

    print(f"\nSummary    : {summary_path}")
    print(f"Details    : {detail_path}")
    print(f"Report     : {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
