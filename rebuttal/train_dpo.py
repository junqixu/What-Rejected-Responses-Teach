from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rebuttal.common import add_common_cli_args, configure_logging, ensure_output_dir, load_records, sha256_file, write_json
from rebuttal.controls.length_normalized_dpo import REDUCTIONS, make_reduction_trainer
from rebuttal.identifiability.feature_schema import canonical_error_name


DEFAULT_MODEL = "Qwen/Qwen2-0.5B"
DEFAULT_CHECKPOINT = "/root/autodl-tmp/checkpoint/qwen2_0.5b_sft_op10/checkpoint-10339"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _config(args: argparse.Namespace, n_samples: int) -> dict[str, Any]:
    return {
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "checkpoint": args.checkpoint,
        "train_file": str(Path(args.train_file)),
        "error_types": args.error_types,
        "n_samples": n_samples,
        "output_dir": str(Path(args.output_dir)),
        "epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_per_process": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "sequence_logp_reduction": args.sequence_logp_reduction,
        "seed": args.seed,
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
        "optimizer": args.optimizer,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
    }


def _validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("training data is empty")
    seen_ids: set[str] = set()
    prompt_chosen: dict[str, set[str]] = defaultdict(set)
    error_counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        for field in ("prompt", "chosen", "rejected"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"row {index} has no non-empty {field!r}")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"row {index} has identical chosen and rejected responses")
        record_id = str(row.get("id") or "")
        if record_id:
            if record_id in seen_ids:
                raise ValueError(f"duplicate training record id: {record_id}")
            seen_ids.add(record_id)
        prompt_id = str(row.get("prompt_id") or row.get("prompt") or "")
        prompt_chosen[prompt_id].add(row["chosen"])
        if row.get("chosen_hash") and row["chosen_hash"] != hashlib.sha256(row["chosen"].encode("utf-8")).hexdigest():
            raise ValueError(f"row {index} has an invalid chosen_hash")
        if row.get("rejected_hash") and row["rejected_hash"] != hashlib.sha256(row["rejected"].encode("utf-8")).hexdigest():
            raise ValueError(f"row {index} has an invalid rejected_hash")
        error_counts[str(row.get("error_type") or "unspecified")] += 1
    inconsistent = sorted(prompt_id for prompt_id, values in prompt_chosen.items() if len(values) > 1)
    if inconsistent:
        raise ValueError(f"{len(inconsistent)} prompts have inconsistent chosen responses")
    return {
        "n_records": len(rows),
        "n_prompts": len(prompt_chosen),
        "error_type_counts": dict(sorted(error_counts.items())),
        "record_ids_present": len(seen_ids),
    }


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _dataset_manifest_path(train_file: Path) -> Path | None:
    for parent in (train_file.parent, train_file.parent.parent):
        candidate = parent / "dataset_manifest.json"
        if candidate.exists():
            return candidate
    return None


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_records(args.train_file)
    if args.error_types:
        requested = {
            canonical_error_name(value)
            for value in args.error_types.split(",")
            if value.strip()
        }
        rows = [row for row in rows if canonical_error_name(str(row.get("error_type"))) in requested]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    dataset_summary = _validate_rows(rows)
    config = _config(args, len(rows))
    config["dataset_summary"] = dataset_summary
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["config_hash"] = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    config["dataset_hash"] = sha256_file(args.train_file)
    dataset_manifest = _dataset_manifest_path(Path(args.train_file))
    config["dataset_manifest"] = str(dataset_manifest) if dataset_manifest else None
    config["dataset_manifest_hash"] = sha256_file(dataset_manifest) if dataset_manifest else None
    config["git_commit"] = _git_commit(Path.cwd())

    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return config

    output = ensure_output_dir(args.output_dir, args.overwrite)
    config["started_at"] = datetime.now(timezone.utc).isoformat()
    config["status"] = "initializing"
    write_json(output / "run_manifest.json", config)
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as error:
        config["status"] = "failed"
        config["failure_type"] = type(error).__name__
        config["failure_message"] = str(error)
        write_json(output / "run_manifest.json", config)
        raise RuntimeError("GPU training dependencies are unavailable; use --dry_run for validation") from error

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        optim=args.optimizer,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=args.seed,
    )
    trainer_class = make_reduction_trainer(DPOTrainer, args.sequence_logp_reduction)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "beta": args.beta,
        "train_dataset": Dataset.from_list(rows),
        "max_prompt_length": args.max_prompt_length,
        "max_length": args.max_length,
    }
    signature = inspect.signature(DPOTrainer.__init__)
    trainer_kwargs["processing_class" if "processing_class" in signature.parameters else "tokenizer"] = tokenizer
    trainer = trainer_class(**trainer_kwargs)

    config["environment"] = {
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "trl": _package_version("trl"),
    }
    config["status"] = "running"
    write_json(output / "run_manifest.json", config)
    try:
        trainer.train()
        trainer.save_model(str(output))
        tokenizer.save_pretrained(str(output))
    except BaseException as error:
        config["status"] = "failed"
        config["failure_type"] = type(error).__name__
        config["failure_message"] = str(error)
        config["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_manifest.json", config)
        raise
    config["status"] = "complete"
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run_manifest.json", config)
    return config


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuttal DPO trainer with explicit sequence log-probability reduction.")
    parser.add_argument("--config", default=None, help="JSON-compatible YAML experiment config.")
    parser.add_argument("--train_file", required=not bool(defaults and defaults.get("train_file")))
    parser.add_argument("--output_dir", required=not bool(defaults and defaults.get("output_dir")))
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--error_types",
        default=None,
        help="Optional comma-separated error-type filter for a shared all-types dataset.",
    )
    parser.add_argument("--model_revision", default=None)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--sequence_logp_reduction", choices=REDUCTIONS, default="sum")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--optimizer", default="paged_adamw_8bit")
    add_common_cli_args(parser)
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    known, _unknown = config_parser.parse_known_args()
    defaults: dict[str, Any] | None = None
    if known.config:
        with Path(known.config).open("r", encoding="utf-8") as handle:
            defaults = json.load(handle)
        if not isinstance(defaults, dict):
            raise ValueError("config must contain a JSON object")
    args = build_parser(defaults).parse_args()
    configure_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
