#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
import swanlab
from swanlab.integration.transformers import SwanLabCallback


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def main():
    MODEL_PATH = "/root/autodl-tmp/qwen2_0.5b_sft_op10/checkpoint-10339"
    TRAIN_FILE = "/root/dpo_outputs_missing_only/dpo_train.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/dpo_full_final_missing_only_1"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # SwanLab 初始化
    # =========================
    swanlab.init(
        project="qwen2-dpo-full-training",
        experiment_name="qwen2-0.5b-dpo-full-4090",
        description="Qwen2-0.5B 全量DPO训练",
        logdir="./swanlog_full",
    )

    # =========================
    # 数据加载
    # =========================
    dataset = load_dataset("json", data_files=TRAIN_FILE)  # 简化加载

    def clean_example(example):
        example["prompt"] = clean_text(example.get("prompt"))
        example["chosen"] = clean_text(example.get("chosen"))
        example["rejected"] = clean_text(example.get("rejected"))
        return example

    dataset = dataset.map(clean_example)

    def valid_example(example):
        return (
            isinstance(example["prompt"], str)
            and isinstance(example["chosen"], str)
            and isinstance(example["rejected"], str)
            and len(example["prompt"]) > 0
            and len(example["chosen"]) > 0
            and len(example["rejected"]) > 0
            and example["chosen"] != example["rejected"]
        )

    dataset = dataset.filter(valid_example)

    # ✅ 正确打乱（只针对 train 集）
    # dataset["train"] = dataset["train"].shuffle(seed=42)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))

    print(f"[INFO] 当前训练样本数: {len(dataset['train'])}")

    # =========================
    # Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # =========================
    # 模型
    # =========================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.use_cache = False

    # 开启 dropout 抗过拟合
    model.config.attention_probs_dropout_prob = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.train()

    # =========================
    # 训练参数（抗过拟合）
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.1,
        max_grad_norm=0.5,
    )

    # =========================
    # DPO Trainer
    # =========================
    trainer = DPOTrainer(
        model=model,
        args=training_args,
       dpo_beta=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_length=1024,
        max_target_length=512,
        callbacks=[SwanLabCallback()]
    )

    # =========================
    # 训练
    # =========================
    print("[INFO] 开始全量 DPO 训练 ✅")
    trainer.train()

    # =========================
    # 保存
    # =========================
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    swanlab.finish()
    print("[✅ 训练完成！模型已保存]")


if __name__ == "__main__":
    main()