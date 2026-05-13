#!/usr/bin/env python3
"""
Stage 3: Full-Parameter DPO Training
=====================================
Fine-tunes a base model (e.g. Qwen2-0.5B) using Direct Preference Optimization
with ALL model parameters trainable.

This is the primary training script used for the paper's DPO experiments.

Key design decisions
--------------------
- **Full fine-tuning** (no LoRA): maximum capacity, highest VRAM requirements
- **Gradient accumulation** to simulate larger batch sizes
- **bfloat16** for mixed-precision stability
- **Gradient checkpointing** to reduce memory footprint
- **Cosine LR schedule** with warmup
- **SwanLab** callback for experiment tracking
- **Dropout** (0.1) on attention and hidden layers to combat overfitting

Usage
-----
    python -m stage3_training.full_dpo.run_full_dpo \
        --base_model Qwen/Qwen2-0.5B \
        --sft_checkpoint ./checkpoints/qwen2_0.5b_sft/checkpoint-10339 \
        --train_file ./data/dpo_mixed.jsonl \
        --output_dir ./checkpoints/dpo_full_mix \
        --learning_rate 2e-5 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs 1 \
        --beta 0.1

Hardware
--------
Tested on: NVIDIA RTX 4090 (24 GB VRAM) + 32 GB RAM
Expected VRAM: ~18–22 GB depending on max_length

Author: Reasoning-DPO Pipeline
"""

import os
import sys
import torch
import warnings
import argparse

warnings.filterwarnings("ignore")

from transformers import TrainingArguments

# Set up environment before any other HF imports
from stage3_training.trainer_base import (
    setup_environment,
    load_tokenizer,
    load_model,
    apply_dropout,
    load_dpo_jsonl,
    clean_dataset,
    inspect_dataset,
    dpo_training_args,
    build_dpo_trainer,
)

# SwanLab integration (optional import)
try:
    import swanlab
    from swanlab.integration.transformers import SwanLabCallback
    _SWANLAB_AVAILABLE = True
except ImportError:
    swanlab = None
    SwanLabCallback = None
    _SWANLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL = "Qwen/Qwen2-0.5B"
DEFAULT_SFT_CHECKPOINT = "/root/autodl-tmp/checkpoint/qwen2_0.5b_sft_op10/checkpoint-10339"
DEFAULT_TRAIN_FILE = "./data/dpo_mixed.jsonl"
DEFAULT_OUTPUT_DIR = "./checkpoints/dpo_full"


# ---------------------------------------------------------------------------
# SwanLab helpers
# ---------------------------------------------------------------------------

def init_swanlab(project: str, experiment: str, description: str, logdir: str):
    if not _SWANLAB_AVAILABLE:
        print("[WARN] SwanLab not installed. Skipping experiment tracking.")
        return None
    swanlab.init(
        project=project,
        experiment_name=experiment,
        description=description,
        logdir=logdir,
    )
    return SwanLabCallback()


def finish_swanlab():
    if _SWANLAB_AVAILABLE and swanlab is not None:
        swanlab.finish()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    base_model: str,
    sft_checkpoint: str,
    train_file: str,
    output_dir: str,
    learning_rate: float,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    beta: float,
    max_prompt_length: int = 512,
    max_length: int = 1024,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    max_grad_norm: float = 0.5,
    learning_rate_full: float = 5e-6,
    use_swanlab: bool = True,
    logdir: str = "./swanlog_full",
    verbose: bool = True,
) -> None:
    """
    Run full-parameter DPO training.
    
    Parameters
    ----------
    base_model : HuggingFace model ID or local path
        Base model to initialise from (only its tokenizer is used).
    sft_checkpoint : local path
        Path to the SFT fine-tuned checkpoint (the actual model weights).
    train_file : path
        .jsonl DPO training file with prompt/chosen/rejected fields.
    output_dir : local path
        Where to save model checkpoints.
    learning_rate : float
        Learning rate for the DPO beta parameter head and newly initialised layers.
        Note: for full-parameter training all layers share this LR (via layerwise lr decay
        can be added via TrainingArguments if needed).
    learning_rate_full : float
        Override LR for all parameters (use this instead of learning_rate for
        full fine-tuning).
    beta : float
        DPO beta ( KL penalty strength). 0.1 is the paper default.
    max_prompt_length : int
        Max token length for the prompt section.
    max_length : int
        Max total token length (prompt + response).
    warmup_ratio : float
        Fraction of training steps used for LR warmup.
    weight_decay : float
        L2 regularisation coefficient.
    max_grad_norm : float
        Gradient clipping threshold.
    use_swanlab : bool
        Enable SwanLab experiment tracking.
    verbose : bool
        Print dataset statistics before training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- SwanLab ----
    callbacks = []
    if use_swanlab:
        cb = init_swanlab(
            project="reasoning-dpo-full",
            experiment=os.path.basename(output_dir),
            description=f"Full DPO | lr={learning_rate_full} | β={beta} | epochs={num_train_epochs}",
            logdir=logdir,
        )
        if cb:
            callbacks.append(cb)

    # ---- Tokenizer (from base model) ----
    if verbose:
        print(f"[INFO] Loading tokenizer from: {base_model}")
    tokenizer = load_tokenizer(base_model)
    if verbose:
        print(f"[INFO] pad_token={tokenizer.pad_token} | eos_token={tokenizer.eos_token}")

    # ---- Model (from SFT checkpoint) ----
    if verbose:
        print(f"[INFO] Loading SFT model from: {sft_checkpoint}")
    model = load_model(sft_checkpoint)
    apply_dropout(model, attention_dropout=0.1, hidden_dropout=0.1)
    if verbose:
        print(f"[INFO] Model loaded. Trainable params: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
        learning_rate=learning_rate_full,
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
        optim="paged_adamw_32bit",
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

    # ---- Train ----
    print("[INFO] Starting full DPO training ...")
    trainer.train()

    # ---- Save ----
    print(f"[INFO] Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    finish_swanlab()
    print("[DONE] Full DPO training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 — Full DPO Training")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--sft_checkpoint", required=True,
                   help="Path to SFT fine-tuned checkpoint (model weights)")
    p.add_argument("--train_file", required=True,
                   help="Path to .jsonl DPO dataset (prompt/chosen/rejected)")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the DPO checkpoint")
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--per_device_batch_size", "--per_device_train_batch_size",
                   dest="per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
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
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        learning_rate_full=args.learning_rate,
        use_swanlab=args.use_swanlab,
    )


if __name__ == "__main__":
    main()
