#!/usr/bin/env python3
"""
Stage 3: LoRA DPO Training
===========================
Fine-tunes a base model using Direct Preference Optimization with
Low-Rank Adaptation (LoRA), dramatically reducing VRAM requirements
compared to full-parameter fine-tuning.

Target modules for LoRA adaptation:
    q_proj, k_proj, v_proj, o_proj   (attention)
    gate_proj, up_proj, down_proj    (MLP)

This script is suitable for development and ablation experiments
where a full fine-tune is not necessary.

Usage
-----
    python -m stage3_training.lora_dpo.run_lora_dpo \
        --base_model Qwen/Qwen2-0.5B \
        --sft_checkpoint ./checkpoints/qwen2_0.5b_sft/checkpoint-10339 \
        --train_file ./data/dpo_mixed.jsonl \
        --output_dir ./checkpoints/dpo_lora \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --learning_rate 1e-5 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1

Hardware
--------
Tested on: NVIDIA A100 (40 GB) or RTX 3090 (24 GB)
Expected VRAM: ~10–14 GB for Qwen2-0.5B with LoRA rank 16

Author: Reasoning-DPO Pipeline
"""

import os
import torch
import warnings
import argparse

warnings.filterwarnings("ignore")

from transformers import TrainingArguments

from stage3_training.trainer_base import (
    setup_environment,
    load_tokenizer,
    load_model,
    load_dpo_jsonl,
    clean_dataset,
    inspect_dataset,
    dpo_training_args,
    build_dpo_trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    import swanlab
    from swanlab.integration.huggingface import SwanLabCallback
    _SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None
    SwanLabCallback = lambda: None
    _SWANLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2-0.5B"
DEFAULT_SFT_CHECKPOINT = "/root/autodl-tmp/qwen2_0.5b_sft_op10/checkpoint-10339"
DEFAULT_TRAIN_FILE = "./data/dpo_mixed.jsonl"
DEFAULT_OUTPUT_DIR = "./checkpoints/dpo_lora"


# ---------------------------------------------------------------------------
# LoRA config
# ---------------------------------------------------------------------------

def default_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """
    Standard LoRA configuration for Qwen2 causal LMs.

    Targets all linear layers in the transformer stack that are
    most impactful for instruction following and reasoning.
    """
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    sft_checkpoint: str,
    train_file: str,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    beta: float = 0.1,
    max_prompt_length: int = 512,
    max_length: int = 1024,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    use_swanlab: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run LoRA DPO training.

    The SFT checkpoint provides the base model weights; LoRA adapters are
    initialised on top and trained while the base weights remain frozen.
    """
    os.makedirs(output_dir, exist_ok=True)

    callbacks = []
    if use_swanlab and _SWANLAB_AVAILABLE:
        swanlab.init(
            project="reasoning-dpo-lora",
            experiment_name=os.path.basename(output_dir),
            description=f"LoRA DPO | r={lora_r} α={lora_alpha} | β={beta}",
            logdir="./swanlog_lora",
        )
        callbacks.append(SwanLabCallback())

    # ---- Tokenizer ----
    if verbose:
        print(f"[INFO] Loading tokenizer: {base_model}")
    tokenizer = load_tokenizer(base_model)

    # ---- Tokenization sanity check ----
    if verbose:
        print("[INFO] Running tokenization sanity check...")
    dataset_raw = load_dpo_jsonl(train_file)
    dataset_raw = clean_dataset(dataset_raw)
    test_prompt = dataset_raw[0]["prompt"]
    test_ids = tokenizer(test_prompt, add_special_tokens=False)["input_ids"]
    assert all(x is not None for x in test_ids), "Tokenizer returned None IDs!"
    if verbose:
        print("[INFO] Tokenization check passed.")

    # ---- Model ----
    if verbose:
        print(f"[INFO] Loading SFT model: {sft_checkpoint}")
    model = load_model(sft_checkpoint)

    # ---- LoRA ----
    lora_cfg = default_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    if verbose:
        print(f"[INFO] Applying LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---- Data ----
    if verbose:
        print(f"[INFO] Loading training data: {train_file}")
    dataset = load_dpo_jsonl(train_file)
    dataset = clean_dataset(dataset)
    if verbose:
        inspect_dataset(dataset, n=2)
    print(f"[INFO] Training samples: {len(dataset)}")

    # ---- Training args ----
    training_args = dpo_training_args(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="swanlab" if use_swanlab else "none",
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
    )

    # ---- Trainer ----
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        training_args=training_args,
        beta=beta,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        callbacks=callbacks,
    )

    print("[INFO] Starting LoRA DPO training ...")
    trainer.train()

    print(f"[INFO] Saving model: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    if use_swanlab and _SWANLAB_AVAILABLE:
        swanlab.finish()
    print("[DONE] LoRA DPO training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — LoRA DPO Training")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--sft_checkpoint", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--no_swanlab", dest="use_swanlab", action="store_false", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_environment()
    train(
        base_model=args.base_model,
        sft_checkpoint=args.sft_checkpoint,
        train_file=args.train_file,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        use_swanlab=args.use_swanlab,
    )


if __name__ == "__main__":
    main()
