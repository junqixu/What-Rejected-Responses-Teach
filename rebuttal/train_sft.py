from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from rebuttal.common import (
    add_common_cli_args,
    configure_logging,
    ensure_output_dir,
    load_records,
    sha256_file,
    write_json,
)


DEFAULT_MODEL = "Qwen/Qwen2-0.5B"
DEFAULT_TRAIN_FILE = "data/rebuttal_v2/source/train_clean.jsonl"
DEFAULT_HELDOUT_FILES = (
    "data/rebuttal_v2/source/validation_clean.jsonl",
    "data/rebuttal_v2/source/test_clean.jsonl",
)
DEFAULT_OUTPUT_DIR = "outputs/rebuttal_sft/qwen2_0.5b_op10"


def format_prompt(row: dict[str, Any]) -> str:
    return f"Problem: {row['problem'].strip()}\nQuestion: {row['question'].strip()}\nAnswer: "


def _validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("SFT training data is empty")
    prompt_ids: set[str] = set()
    family_ids: set[str] = set()
    modes: set[str] = set()
    templates: set[str] = set()
    for index, row in enumerate(rows):
        for field in ("problem", "question", "solution", "prompt_id", "dag_family_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"row {index} has no non-empty {field!r}")
        if row.get("split") != "train":
            raise ValueError(f"row {index} has split={row.get('split')!r}; SFT accepts train only")
        if row.get("op") != 10:
            raise ValueError(f"row {index} has op={row.get('op')!r}; SFT requires clean OP=10 rows")
        prompt_id = row["prompt_id"]
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate SFT prompt_id: {prompt_id}")
        prompt_ids.add(prompt_id)
        family_ids.add(row["dag_family_id"])
        modes.add(str(row.get("mode") or "unspecified"))
        templates.add(str(row.get("template") or "unspecified"))
    return {
        "n_records": len(rows),
        "n_prompts": len(prompt_ids),
        "n_dag_families": len(family_ids),
        "modes": sorted(modes),
        "templates": sorted(templates),
    }


def _validate_heldout(rows: list[dict[str, Any]], heldout_files: Iterable[str]) -> dict[str, Any]:
    train_prompts = {str(row["prompt_id"]) for row in rows}
    train_families = {str(row["dag_family_id"]) for row in rows}
    report: dict[str, Any] = {}
    for raw_path in heldout_files:
        path = Path(raw_path)
        heldout = load_records(path)
        splits = sorted({str(row.get("split")) for row in heldout})
        if "train" in splits:
            raise ValueError(f"held-out file contains train rows: {path}")
        prompt_overlap = train_prompts & {str(row.get("prompt_id")) for row in heldout}
        family_overlap = train_families & {str(row.get("dag_family_id")) for row in heldout}
        if prompt_overlap or family_overlap:
            raise ValueError(
                f"split leakage against {path}: {len(prompt_overlap)} prompts and "
                f"{len(family_overlap)} DAG families overlap"
            )
        report[str(path)] = {
            "n_records": len(heldout),
            "splits": splits,
            "sha256": sha256_file(path),
        }
    return report


def tokenize_example(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer(format_prompt(row), add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(row["solution"].strip(), add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id")
    answer_ids = answer_ids + [eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("max_length leaves no answer token for an SFT sample")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "was_truncated": len(prompt_ids) + len(answer_ids) > max_length,
    }


@dataclass
class CompletionOnlyCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(feature["input_ids"]) for feature in features)
        target = ((longest + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = target - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None


def _config(args: argparse.Namespace, dataset_summary: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "train_file": str(Path(args.train_file)),
        "train_file_sha256": sha256_file(args.train_file),
        "heldout_files": list(args.heldout_files),
        "output_dir": str(Path(args.output_dir)),
        "dataset_summary": dataset_summary,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_per_process": (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "optimizer": args.optimizer,
        "seed": args.seed,
    }
    stable = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_hash"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()
    config["git_commit"] = _git_commit()
    return config


def _completed_checkpoint(output: Path, config_hash: str) -> dict[str, Any] | None:
    manifest_path = output / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_hash = manifest.get("config_hash")
    if existing_hash and existing_hash != config_hash:
        raise ValueError(
            f"existing SFT run at {output} has a different configuration; "
            "choose a new --output_dir"
        )
    if manifest.get("status") != "complete":
        return None
    has_weights = any(output.glob("*.safetensors")) or any(output.glob("pytorch_model*.bin"))
    required = (output / "config.json", output / "tokenizer_config.json")
    if not has_weights or not all(path.exists() for path in required):
        raise ValueError(f"completed SFT manifest exists but checkpoint files are incomplete: {output}")
    manifest["reused_completed_checkpoint"] = True
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_records(args.train_file, max_samples=args.max_samples)
    dataset_summary = _validate_rows(rows)
    dataset_summary["heldout_validation"] = _validate_heldout(rows, args.heldout_files)
    config = _config(args, dataset_summary)
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return config

    output_path = Path(args.output_dir)
    completed = _completed_checkpoint(output_path, config["config_hash"])
    if completed is not None:
        return completed
    retry_incomplete = (output_path / "run_manifest.json").exists()
    output = ensure_output_dir(output_path, args.overwrite or retry_incomplete)
    config.update(
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "initializing",
        }
    )
    write_json(output / "run_manifest.json", config)
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as error:
        config.update(status="failed", failure_type=type(error).__name__, failure_message=str(error))
        write_json(output / "run_manifest.json", config)
        raise RuntimeError("SFT dependencies are unavailable; run the server bootstrap first") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenized = [tokenize_example(tokenizer, row, args.max_length) for row in rows]
    truncated = sum(bool(row.pop("was_truncated")) for row in tokenized)
    dataset_summary["truncated_records"] = truncated
    dataset_summary["max_length"] = args.max_length

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    resolved_revision = getattr(model.config, "_commit_hash", None)
    config["resolved_model_revision"] = resolved_revision
    config["environment"] = {
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
    }
    config["status"] = "running"
    write_json(output / "run_manifest.json", config)

    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim=args.optimizer,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        eval_strategy="no",
        logging_steps=args.logging_steps,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(tokenized),
        data_collator=CompletionOnlyCollator(tokenizer.pad_token_id),
    )
    try:
        result = trainer.train()
        model.config.use_cache = True
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
    except BaseException as error:
        config.update(
            status="failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        write_json(output / "run_manifest.json", config)
        raise
    config["train_metrics"] = result.metrics
    config["status"] = "complete"
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run_manifest.json", config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a clean OP=10 SFT checkpoint for rebuttal DPO.")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default="main")
    parser.add_argument("--train_file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--heldout_files", nargs="*", default=list(DEFAULT_HELDOUT_FILES))
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--optimizer", default="adamw_torch")
    parser.add_argument("--logging_steps", type=int, default=5)
    add_common_cli_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
