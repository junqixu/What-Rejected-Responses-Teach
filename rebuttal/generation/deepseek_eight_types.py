from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    iter_records,
    parse_bool,
    prompt_split_leakage,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from rebuttal.identifiability.feature_schema import ERROR_FEATURES, canonical_error_name
from rebuttal.taxonomy.base import ANSWER, apply_perturbation, parse_trace


LOGGER = logging.getLogger(__name__)
GENERATOR_VERSION = "deepseek-rebuttal-v3"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
OPERATOR = re.compile(r"(?<!\w)([+*/-])(?!\w)")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class DirectDeepSeekClient:
    """Small OpenAI-compatible fallback using only the Python standard library."""

    def __init__(self, api_key: str, base_url: str, timeout: int = 180) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def create_completion(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


ERROR_INSTRUCTIONS: dict[str, str] = {
    "computation": (
        "Change exactly one arithmetic/numeric computation. Keep variables, dependencies, "
        "step order, and wording as unchanged as possible. Propagate the numerical error to the final answer."
    ),
    "operation_substitution": (
        "Replace exactly one mathematical operator (+, -, *, /) at a reasoning node. Keep the same "
        "operands, dependencies, node order, and target question; propagate the changed result."
    ),
    "dependency_redirect": (
        "Redirect exactly one dependency to a different already-defined variable. Keep the same number "
        "of reasoning nodes and otherwise preserve wording and order."
    ),
    "extra_edge": (
        "Add exactly one inappropriate dependency from an already-defined variable into one calculation. "
        "Do not add or delete reasoning nodes."
    ),
    "missing_node": (
        "Delete exactly one required intermediate reasoning node so a later step uses an unsupported value. "
        "Do not replace the missing node with an equivalent shortcut."
    ),
    "spurious_node": (
        "Insert exactly one plausible but unnecessary intermediate variable and make at least one later "
        "reasoning node explicitly use it. Preserve all original required nodes and keep the final answer unchanged."
    ),
    "disorder": (
        "Reorder exactly two existing reasoning steps so a step appears before a dependency it requires. "
        "Do not rewrite, add, delete, or numerically change the steps."
    ),
    "wrong_target": (
        "Keep the complete reasoning prefix unchanged but answer a different previously computed quantity "
        "at the final Answer field. Change only the target/final answer."
    ),
}


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return (key, value) if key else None


def load_env_file(path: str | Path | None) -> bool:
    if not path:
        return False
    env_path = Path(path)
    if not env_path.exists():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        assignment = _parse_env_assignment(line)
        if assignment is not None:
            key, value = assignment
            os.environ.setdefault(key, value)
    return True


def _chosen(row: Mapping[str, Any]) -> str:
    for key in ("solution", "chosen", "gold_trace"):
        value = _clean(row.get(key))
        if value:
            return value
    raise ValueError("record has no solution/chosen/gold_trace")


def _prompt(row: Mapping[str, Any]) -> str:
    existing = _clean(row.get("prompt"))
    if existing:
        return existing
    problem, question = _clean(row.get("problem")), _clean(row.get("question"))
    body = f"Problem:\n{problem}"
    if question:
        body += f"\n\nQuestion:\n{question}"
    return f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"


def _source_prompt_id(row: Mapping[str, Any]) -> str:
    """Use prompt content because legacy `id` values repeat across distinct prompts."""
    problem, question = _clean(row.get("problem")), _clean(row.get("question"))
    if problem or question:
        return sha256_text(problem + "\n" + question)[:24]
    return sha256_text(_prompt(row))[:24]


def _source_prompt(row: Mapping[str, Any]) -> str:
    return "\n".join(
        part
        for part in (
            f"Problem:\n{_clean(row.get('problem'))}" if row.get("problem") else "",
            f"Question:\n{_clean(row.get('question'))}" if row.get("question") else "",
            f"Correct solution:\n{_chosen(row)}",
        )
        if part
    )


def _last_number(text: str) -> str | None:
    values = NUMBER.findall(text.replace(",", ""))
    return values[-1] if values else None


def _operators(text: str) -> list[str]:
    return OPERATOR.findall(text)


def _normalized_step(step: str) -> str:
    return " ".join(step.split())


def validate_candidate(error_type: str, chosen: str, rejected: str) -> tuple[bool, list[str]]:
    error_type = canonical_error_name(error_type)
    reasons: list[str] = []
    chosen, rejected = _clean(chosen), _clean(rejected)
    if not rejected:
        return False, ["empty_rejected"]
    if rejected == chosen:
        return False, ["identical_to_chosen"]
    similarity = SequenceMatcher(None, chosen, rejected).ratio()
    if similarity < 0.35:
        reasons.append("rewrite_too_large")
    if similarity > 0.999:
        reasons.append("edit_too_small")
    lower = rejected.lower()
    if any(marker in lower for marker in ("error type", "rejected solution", "```json", "here is the json")):
        reasons.append("contains_meta_explanation")

    before, after = parse_trace(chosen), parse_trace(rejected)
    if not before.steps or not after.steps:
        reasons.append("unparseable_trace")
        return not reasons, reasons

    if error_type == "computation":
        if before.node_count != after.node_count or before.edges != after.edges:
            reasons.append("computation_changed_structure")
        if _operators(chosen) != _operators(rejected):
            reasons.append("computation_changed_operator")
        if _last_number(chosen) == _last_number(rejected):
            reasons.append("computation_final_answer_unchanged")
    elif error_type == "operation_substitution":
        if before.node_count != after.node_count or before.edges != after.edges:
            reasons.append("operation_changed_structure")
        if _operators(chosen) == _operators(rejected):
            reasons.append("operator_not_changed")
        if len(before.steps) != len(after.steps):
            reasons.append("operation_changed_step_count")
    elif error_type == "dependency_redirect":
        if before.node_count != after.node_count:
            reasons.append("dependency_changed_node_count")
        if before.edges == after.edges:
            reasons.append("dependency_edges_unchanged")
        if len(before.steps) != len(after.steps):
            reasons.append("dependency_changed_step_count")
    elif error_type == "extra_edge":
        if before.node_count != after.node_count:
            reasons.append("extra_edge_changed_node_count")
        if after.edge_count <= before.edge_count:
            reasons.append("edge_count_not_increased")
    elif error_type == "missing_node":
        if not (
            len(after.steps) < len(before.steps)
            or after.node_count < before.node_count
            or len(after.forward_references) > len(before.forward_references)
        ):
            reasons.append("required_node_not_removed")
        if len(rejected) >= len(chosen):
            reasons.append("missing_node_not_shorter")
    elif error_type == "spurious_node":
        new_nodes = {name for name, _ in after.definitions} - {name for name, _ in before.definitions}
        if after.node_count != before.node_count + 1 or len(after.steps) <= len(before.steps) or len(new_nodes) != 1:
            reasons.append("spurious_node_not_inserted")
        used_as_dependency = {source for source, _target in after.edges}
        if not new_nodes or not any(name in used_as_dependency for name in new_nodes):
            reasons.append("spurious_node_not_used")
        if _last_number(chosen) != _last_number(rejected):
            reasons.append("spurious_node_changed_final_answer")
    elif error_type == "disorder":
        before_steps = sorted(_normalized_step(step) for step in before.steps)
        after_steps = sorted(_normalized_step(step) for step in after.steps)
        if before_steps != after_steps:
            reasons.append("disorder_rewrote_added_or_removed_steps")
        if len(after.forward_references) <= len(before.forward_references):
            reasons.append("disorder_no_new_forward_reference")
    elif error_type == "wrong_target":
        before_answer, after_answer = ANSWER.search(chosen), ANSWER.search(rejected)
        if not before_answer or not after_answer:
            reasons.append("answer_field_missing")
        else:
            if chosen[: before_answer.start(2)] != rejected[: after_answer.start(2)]:
                reasons.append("wrong_target_changed_reasoning_prefix")
            if before_answer.group(2) == after_answer.group(2):
                reasons.append("wrong_target_answer_unchanged")
    return not reasons, reasons


def _select_records(
    paths: list[str], max_samples: int | None, seed: int, op: int | None
) -> tuple[list[dict[str, Any]], int, int]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    eligible = 0
    duplicates = 0
    seen_prompts: set[str] = set()
    for row in iter_records(paths):
        if op is not None:
            try:
                if int(row.get("op")) != op:
                    continue
            except (TypeError, ValueError):
                continue
        try:
            _chosen(row)
        except ValueError:
            continue
        prompt_id = _source_prompt_id(row)
        if prompt_id in seen_prompts:
            duplicates += 1
            continue
        seen_prompts.add(prompt_id)
        eligible += 1
        if max_samples is None:
            reservoir.append(row)
        elif len(reservoir) < max_samples:
            reservoir.append(row)
        else:
            replacement = rng.randrange(eligible)
            if replacement < max_samples:
                reservoir[replacement] = row
    rng.shuffle(reservoir)
    return reservoir, eligible, duplicates


def _deterministic_hints(chosen: str, error_types: Iterable[str]) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    for error_type in error_types:
        result = apply_perturbation(chosen, error_type)
        if result is not None:
            hints[error_type] = {
                "changed_step_indices": list(result.changed_step_indices),
                "edit_metadata": result.metadata,
            }
    return hints


def _system_prompt(error_types: list[str]) -> str:
    lines = [
        "You create controlled DPO rejected solutions for mathematical reasoning.",
        "Preserve the original language, wording, variable names, style, and formatting as closely as possible.",
        "Each output must contain only its requested structural error and no other error type.",
        "Never add explanations, labels, markdown, or commentary inside rejected text.",
        "Return one strict JSON object with exactly the requested keys. Each value must be an object with",
        'a string field "rejected" and a short string field "edit_summary".',
        "",
    ]
    for error_type in error_types:
        lines.append(f"{error_type}: {ERROR_INSTRUCTIONS[error_type]}")
    lines.append("")
    lines.append("Required JSON keys: " + ", ".join(error_types))
    return "\n".join(lines)


def _parse_response(content: str, error_types: list[str]) -> dict[str, dict[str, str]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        first, last = content.find("{"), content.rfind("}")
        if first < 0 or last <= first:
            raise
        parsed = json.loads(content[first : last + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    output: dict[str, dict[str, str]] = {}
    for error_type in error_types:
        value = parsed.get(error_type)
        if not isinstance(value, dict) or not isinstance(value.get("rejected"), str):
            raise ValueError(f"missing {error_type}.rejected")
        output[error_type] = {
            "rejected": value["rejected"].strip(),
            "edit_summary": _clean(value.get("edit_summary")),
        }
    return output


def _call_api(
    client: Any,
    *,
    model: str,
    error_types: list[str],
    row: Mapping[str, Any],
    hints: Mapping[str, Any],
    validation_feedback: Mapping[str, list[str]] | None,
    max_output_tokens: int,
    temperature: float,
) -> tuple[dict[str, dict[str, str]], dict[str, int], str]:
    user_prompt = _source_prompt(row)
    if hints:
        user_prompt += "\n\nDeterministic structural locator hints (do not quote these):\n"
        user_prompt += json.dumps(hints, ensure_ascii=False, sort_keys=True)
    if validation_feedback:
        user_prompt += "\n\nPrevious candidates failed deterministic validation. Produce new candidates and fix every listed issue:\n"
        user_prompt += json.dumps(validation_feedback, ensure_ascii=False, sort_keys=True)
    messages = [
            {"role": "system", "content": _system_prompt(error_types)},
            {"role": "user", "content": user_prompt},
        ]
    if isinstance(client, DirectDeepSeekClient):
        response = client.create_completion({
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        })
        content = str(response["choices"][0]["message"].get("content") or "")
        usage = response.get("usage") or {}
        counts = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    else:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        counts = {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    return _parse_response(content, error_types), counts, content


def _record(
    source: Mapping[str, Any], error_type: str, rejected: str, edit_summary: str, args: argparse.Namespace
) -> dict[str, Any]:
    chosen = _chosen(source)
    prompt_id = _source_prompt_id(source)
    before, after = parse_trace(chosen), parse_trace(rejected)
    vector = {name: int(name == error_type) for name in ERROR_FEATURES}
    return {
        "id": f"{prompt_id}:{error_type}",
        "prompt_id": prompt_id,
        "prompt": _prompt(source),
        "chosen": chosen,
        "rejected": rejected,
        "error_type": error_type,
        "error_labels": [error_type],
        "error_vector": vector,
        "op": source.get("op"),
        "template": source.get("template"),
        "mode": source.get("mode"),
        "split": args.split,
        "seed": args.seed,
        "generator_version": GENERATOR_VERSION,
        "generator_model": args.model,
        "source_hash": sha256_text(chosen),
        "chosen_hash": sha256_text(chosen),
        "rejected_hash": sha256_text(rejected),
        "chosen_tokens": len(chosen.split()),
        "rejected_tokens": len(rejected.split()),
        "final_answer_correct": _last_number(chosen) == _last_number(rejected),
        "dag_metadata": {
            "before_nodes": before.node_count,
            "before_edges": before.edge_count,
            "after_nodes": after.node_count,
            "after_edges": after.edge_count,
            "before_forward_references": len(before.forward_references),
            "after_forward_references": len(after.forward_references),
            "model_edit_summary": edit_summary,
        },
    }


def _load_done(output: Path, error_types: list[str]) -> dict[str, set[str]]:
    done = {error_type: set() for error_type in error_types}
    for error_type in error_types:
        path = output / f"{error_type}.jsonl"
        if not path.exists():
            continue
        for row in iter_records(path):
            done[error_type].add(str(row.get("prompt_id", _source_prompt_id(row))))
    return done


def _append(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def _finish(output: Path, error_types: list[str], source_files: list[str], args: argparse.Namespace, usage: Counter[str]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    rows_by_type: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for error_type in error_types:
        path = output / f"{error_type}.jsonl"
        rows = list(iter_records(path)) if path.exists() else []
        rows_by_type[error_type] = rows
        counts[error_type] = len(rows)
        all_rows.extend(rows)
        if path.exists():
            hashes[path.name] = sha256_file(path)
    all_rows.sort(key=lambda row: (str(row.get("prompt_id")), str(row.get("error_type"))))
    write_jsonl(output / "all_types.jsonl", all_rows)
    complete_prompt_ids = (
        set.intersection(
            *[
                {str(row.get("prompt_id")) for row in rows_by_type[error_type]}
                for error_type in error_types
            ]
        )
        if error_types
        else set()
    )
    balanced_root = output / "balanced"
    balanced_root.mkdir(parents=True, exist_ok=True)
    balanced_rows: list[dict[str, Any]] = []
    balanced_hashes: dict[str, str] = {}
    for error_type in error_types:
        rows = [row for row in rows_by_type[error_type] if str(row.get("prompt_id")) in complete_prompt_ids]
        rows.sort(key=lambda row: str(row.get("prompt_id")))
        path = balanced_root / f"{error_type}.jsonl"
        write_jsonl(path, rows)
        balanced_hashes[str(path.relative_to(output))] = sha256_file(path)
        balanced_rows.extend(rows)
    balanced_rows.sort(key=lambda row: (str(row.get("prompt_id")), str(row.get("error_type"))))
    balanced_all = balanced_root / "all_types.jsonl"
    write_jsonl(balanced_all, balanced_rows)
    balanced_hashes[str(balanced_all.relative_to(output))] = sha256_file(balanced_all)
    leakage = prompt_split_leakage(all_rows)
    cumulative_usage: Counter[str] = Counter()
    raw_generations_path = output / "raw_generations.jsonl"
    if raw_generations_path.exists():
        for raw_record in iter_records(raw_generations_path):
            raw_usage = raw_record.get("usage")
            if isinstance(raw_usage, Mapping):
                cumulative_usage.update(
                    {
                        key: int(raw_usage.get(key, 0) or 0)
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    }
                )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "model": args.model,
        "base_url": args.base_url,
        "source_files": source_files,
        "split": args.split,
        "op_filter": args.op,
        "seed": args.seed,
        "error_types": error_types,
        "counts": counts,
        "output_hashes": hashes,
        "balanced_output_hashes": balanced_hashes,
        "token_usage_total": dict(cumulative_usage),
        "token_usage_this_run": dict(usage),
        "prompt_split_leakage": leakage,
        "complete_prompt_count": len(complete_prompt_ids),
        "balanced_record_count": len(balanced_rows),
        "record_generator_versions": dict(
            Counter(str(row.get("generator_version", "unknown")) for row in all_rows)
        ),
    }
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_file_loaded = load_env_file(args.env_file)
    error_types = [canonical_error_name(value) for value in args.error_types.split(",") if value.strip()]
    if len(set(error_types)) != len(error_types):
        raise ValueError("error_types contains duplicates")
    rows, eligible, duplicates = _select_records(args.input, args.max_samples, args.seed, args.op)
    dry_report = {
        "eligible_source_rows": eligible,
        "duplicate_source_prompts_skipped": duplicates,
        "selected_rows": len(rows),
        "requested_error_types": error_types,
        "potential_records": len(rows) * len(error_types),
        "api_key_configured": bool(os.getenv(args.api_key_env)),
        "env_file_loaded": env_file_loaded,
        "model": args.model,
        "split": args.split,
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2, sort_keys=True))
        return dry_report
    if not rows:
        raise ValueError("no eligible source rows")
    if not os.getenv(args.api_key_env):
        raise RuntimeError(f"environment variable {args.api_key_env} is not configured")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError("output is not empty; use --resume true or a new output directory")
    output.mkdir(parents=True, exist_ok=True)

    try:
        from openai import OpenAI
    except ImportError:
        LOGGER.info("openai package is unavailable; using the standard-library HTTP client")
        client = DirectDeepSeekClient(
            os.environ[args.api_key_env],
            args.base_url,
            timeout=args.request_timeout,
        )
    else:
        client = OpenAI(
            api_key=os.environ[args.api_key_env],
            base_url=args.base_url,
            timeout=args.request_timeout,
            max_retries=0,
        )
    done = _load_done(output, error_types) if args.resume else {name: set() for name in error_types}
    usage: Counter[str] = Counter()
    api_calls = 0
    validation_failures: Counter[str] = Counter()

    budget_exhausted = False
    for source in rows:
        prompt_id = _source_prompt_id(source)
        pending = [error_type for error_type in error_types if prompt_id not in done[error_type]]
        if not pending:
            continue
        chosen = _chosen(source)
        validation_feedback: dict[str, list[str]] = {}
        for attempt in range(1, args.max_retries + 1):
            if not pending:
                break
            if args.max_api_calls is not None and api_calls >= args.max_api_calls:
                budget_exhausted = True
                break
            hints = _deterministic_hints(chosen, pending) if args.hybrid_hints else {}
            try:
                generated, call_usage, raw = _call_api(
                    client,
                    model=args.model,
                    error_types=pending,
                    row=source,
                    hints=hints,
                    validation_feedback=validation_feedback,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                )
                api_calls += 1
                usage.update(call_usage)
                _append(output / "raw_generations.jsonl", {
                    "prompt_id": prompt_id,
                    "error_types": pending,
                    "attempt": attempt,
                    "status": "ok",
                    "raw_response": raw,
                    "usage": call_usage,
                })
            except Exception as error:
                api_calls += 1
                _append(output / "generation_failures.jsonl", {
                    "prompt_id": prompt_id,
                    "error_types": pending,
                    "attempt": attempt,
                    "stage": "api_or_parse",
                    "error": str(error),
                })
                LOGGER.warning("generation attempt %s failed for %s: %s", attempt, prompt_id, error)
                time.sleep(min(30.0, args.sleep_seconds * (2 ** (attempt - 1))))
                continue

            retry_types: list[str] = []
            for error_type in pending:
                candidate = generated[error_type]
                valid, reasons = validate_candidate(error_type, chosen, candidate["rejected"])
                if not valid:
                    validation_failures.update(f"{error_type}:{reason}" for reason in reasons)
                    _append(output / "generation_failures.jsonl", {
                        "prompt_id": prompt_id,
                        "error_type": error_type,
                        "attempt": attempt,
                        "stage": "validation",
                        "reasons": reasons,
                        "rejected": candidate["rejected"],
                    })
                    retry_types.append(error_type)
                    validation_feedback[error_type] = reasons
                    continue
                packed = _record(source, error_type, candidate["rejected"], candidate["edit_summary"], args)
                _append(output / f"{error_type}.jsonl", packed)
                done[error_type].add(prompt_id)
            pending = retry_types
            time.sleep(args.sleep_seconds)
        if budget_exhausted:
            break

    present_types = [
        error_type
        for error_type in ERROR_FEATURES
        if (output / f"{error_type}.jsonl").exists()
    ]
    manifest = _finish(output, present_types or error_types, args.input, args, usage)
    manifest["api_calls_this_run"] = api_calls
    manifest["validation_failures_this_run"] = dict(validation_failures)
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate all eight validated DPO error types with DeepSeek.")
    parser.add_argument("--input", nargs="+", required=True, help="Clean JSON/JSONL source files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--error_types", default=",".join(ERROR_FEATURES))
    parser.add_argument("--op", type=int, default=10)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key_env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--env_file", default="rebuttal/.env")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--request_timeout", type=int, default=90)
    parser.add_argument("--max_api_calls", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=0.5)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--hybrid_hints", type=parse_bool, default=True)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
