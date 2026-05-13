#!/usr/bin/env python3
"""
Stage 3: Base DPO Trainer
=========================
Shared training utilities consumed by both full-parameter and LoRA DPO scripts.

Provides:
- Environment setup (HF mirrors, cache paths)
- Data loading with cleaning and filtering
- Tokenizer setup (with Qwen2 pad_token fix)
- Model loading (bf16, device_map)
- SwanLab integration

Author: Reasoning-DPO Pipeline
"""

from __future__ import annotations

import os
import re
import torch
import warnings
import argparse
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from trl import DPOTrainer


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def setup_environment(cache_dir: str = "/root/autodl-tmp/.cache") -> None:
    """
    Configure HuggingFace cache and mirror for Chinese mainland environments.
    Call this before importing any HF libraries.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", os.path.join(cache_dir, "huggingface"))
    os.environ.setdefault(
        "TRANSFORMERS_CACHE",
        os.path.join(cache_dir, "transformers"),
    )
    os.environ.setdefault(
        "HUGGINGFACE_HUB_CACHE",
        os.path.join(cache_dir, "huggingface", "hub"),
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str, trust_remote_code: bool = True) -> AutoTokenizer:
    """
    Load tokenizer with Qwen2-specific fixes:
    - pad_token → eos_token (Qwen2 has no pad_token)
    - bos_token → eos_token  (Qwen2 has no bos_token)
    - padding_side = "right"
    - truncation_side = "right"
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Load a causal LM with bfloat16 and automatic device placement.
    Disables use_cache to be compatible with training.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
    )
    model.config.use_cache = False
    return model


def apply_dropout(model: AutoModelForCausalLM,
                   attention_dropout: float = 0.1,
                   hidden_dropout: float = 0.1) -> None:
    """Enable dropout for regularization during DPO training."""
    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = attention_dropout
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = hidden_dropout
    model.train()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dpo_jsonl(file_path: str) -> Dataset:
    """
    Load a DPO dataset from a .jsonl file.
    Each record must have: prompt, chosen, rejected.
    """
    dataset = load_dataset("json", data_files=file_path, split="train")
    return dataset


def load_dpo_json(file_path: str) -> Dataset:
    """Load DPO dataset from a .json array file."""
    dataset = load_dataset("json", data_files=file_path, split="train")
    return dataset


def clean_text(text: Optional[str]) -> str:
    """Normalise whitespace in DPO text fields."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def build_prompt_from_record(record: dict) -> str:
    """
    Reconstruct the ChatML prompt from source fields.
    Used when the .jsonl does not store the prompt field.
    """
    problem = clean_text(record.get("problem", ""))
    question = clean_text(record.get("question", ""))
    user = f"Problem:\n{problem}\n\nQuestion:\n{question}" if question else f"Problem:\n{problem}"
    return (
        "<|im_start|>user\n"
        f"{user}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_dataset(dataset: Dataset, min_prompt_len: int = 5,
                  min_chosen_len: int = 10,
                  min_rejected_len: int = 10) -> Dataset:
    """
    Clean and filter a DPO dataset:
    - Remove entries where chosen == rejected
    - Remove entries where any field is empty
    - Remove entries that are too short
    """

    def _clean(example):
        prompt = clean_text(example.get("prompt", ""))
        chosen = clean_text(example.get("chosen", ""))
        rejected = clean_text(example.get("rejected", ""))

        # If prompt is missing, try to rebuild from source fields
        if not prompt and (example.get("problem") or example.get("question")):
            prompt = build_prompt_from_record(example)

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def _filter(example):
        p = example["prompt"]
        c = example["chosen"]
        r = example["rejected"]
        if not p or not c or not r:
            return False
        if len(p) < min_prompt_len or len(c) < min_chosen_len or len(r) < min_rejected_len:
            return False
        if c.strip() == r.strip():
            return False
        return True

    dataset = dataset.map(_clean)
    dataset = dataset.filter(_filter)
    return dataset


def inspect_dataset(dataset: Dataset, n: int = 3) -> None:
    """Print a few examples for debugging."""
    for i, ex in enumerate(dataset.select(range(min(n, len(dataset))))):
        print(f"\n--- Example {i} ---")
        print(f"prompt : {ex['prompt'][:120]}...")
        print(f"chosen : {ex['chosen'][:120]}...")
        print(f"rejected: {ex['rejected'][:120]}...")


# ---------------------------------------------------------------------------
# Training arguments factory
# ---------------------------------------------------------------------------

def dpo_training_args(
    output_dir: str,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 1,
    logging_steps: int = 10,
    save_steps: int = 200,
    save_total_limit: int = 2,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    report_to: str = "none",
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_grad_norm: float = 0.5,
    lr_scheduler_type: str = "cosine",
    optim: str = "paged_adamw_32bit",
) -> TrainingArguments:
    """Standard DPO TrainingArguments with hard饱和 defaults."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy="no",
        report_to=report_to,
        remove_unused_columns=False,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=None,
    )


# ---------------------------------------------------------------------------
# DPO Trainer factory
# ---------------------------------------------------------------------------

def build_dpo_trainer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset,
    training_args: TrainingArguments,
    beta: float = 0.1,
    max_prompt_length: int = 512,
    max_length: int = 1024,
    callbacks: Optional[List] = None,
) -> DPOTrainer:
    """Construct a DPOTrainer with consistent defaults."""
    return DPOTrainer(
        model=model,
        args=training_args,
        beta=beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        callbacks=callbacks or [],
    )
