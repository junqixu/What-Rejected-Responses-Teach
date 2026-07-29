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
# 新版正确导入
from swanlab.integration.transformers import SwanLabCallback


def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def main():
    MODEL_PATH = "/root/autodl-tmp/checkpoint/qwen2_0.5b_sft_op10/checkpoint-10339"
    TRAIN_FILE = "/root/autodl-tmp/data/yzh_data/dpo_missing_nodes_414.json"
    OUTPUT_DIR = "/root/autodl-tmp/checkpoint/dpo_full_missing_nodes_1"


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
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE})



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

    dataset["train"] = dataset["train"].filter(valid_example)
    # dataset["train"] = dataset["train"].shuffle(seed=42).select(range())


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
    # 模型（全量）
    # =========================
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.use_cache = False

    # =========================
    # 训练参数
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
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
        warmup_ratio=0.03,
    )
#     training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=8,  # 保持不变
#     learning_rate=8e-6,            # ✅ 降低学习率（DPO 最关键）
#     num_train_epochs=1,            # 保持不变
#     logging_steps=10,
#     save_steps=200,
#     save_total_limit=2,
#     bf16=True,
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={"use_reentrant": False},
#     report_to="none",
#     remove_unused_columns=False,
#     optim="paged_adamw_8bit",
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.01,              # ✅ 增加预热，稳定训练
#     # max_grad_norm=1.0,              # ✅ 梯度裁剪（解决你图中的震荡！）
# )

    # =========================
    # DPO Trainer + SwanLab 图表
    # =========================
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,   # ✅ trl 0.14.0 正确写法=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_length=1024,
        # 新版回调，能正常显示图表
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
