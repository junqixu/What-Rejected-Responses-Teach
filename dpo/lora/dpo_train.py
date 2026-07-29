#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import DPOTrainer
# 导入 swanlab
import swanlab
from swanlab.integration.huggingface import SwanLabCallback

def clean_text(text):
    """清理文本，避免空值、换行、奇怪空格"""
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def main():
    # =========================
    # 路径配置
    # =========================
    MODEL_PATH = "/root/autodl-tmp/qwen2_0.5b_sft_op10/checkpoint-10339"
    TRAIN_FILE = "/root/dpo_outputs/dpo_mixed.json"
    OUTPUT_DIR = "/root/autodl-tmp/dpo_outputs_new"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================
    # 初始化 SwanLab
    # =========================
    swanlab.init(
        project="qwen2-dpo-training",  # 你的项目名
        experiment_name="qwen2-0.5b-dpo",  # 实验名
        description="Qwen2-0.5B DPO对齐训练",  # 实验描述
        logdir="./swanlog",  # 日志本地保存路径
    )

    # =========================
    # 1. 加载并清洗数据
    # =========================
    print("[INFO] 加载数据...")
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

    print(f"[INFO] 训练样本数: {len(dataset['train'])}")
    print("[INFO] 样例:")
    print(dataset["train"][0])

    # =========================
    # 2. 加载 tokenizer
    # =========================
    print("[INFO] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True
    )

    # Qwen2 没有 bos_token，这里手动补
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    print("[INFO] tokenizer special tokens:")
    print("  pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("  eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("  bos_token:", tokenizer.bos_token, tokenizer.bos_token_id)

    # =========================
    # 3. 简单自检：确保 tokenizer 不会出 None
    # =========================
    print("[INFO] 做 tokenization 自检...")
    test_prompt = dataset["train"][0]["prompt"]
    test_chosen = dataset["train"][0]["chosen"]
    test_rejected = dataset["train"][0]["rejected"]

    test_prompt_ids = tokenizer(test_prompt, add_special_tokens=False)["input_ids"]
    test_chosen_ids = tokenizer(test_chosen, add_special_tokens=False)["input_ids"]
    test_rejected_ids = tokenizer(test_rejected, add_special_tokens=False)["input_ids"]

    assert all(x is not None for x in test_prompt_ids), "prompt token ids 中有 None"
    assert all(x is not None for x in test_chosen_ids), "chosen token ids 中有 None"
    assert all(x is not None for x in test_rejected_ids), "rejected token ids 中有 None"

    print("[INFO] tokenization 自检通过 ✅")

    # =========================
    # 4. 加载模型
    # =========================
    print("[INFO] 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.config.use_cache = False

    # =========================
    # 5. LoRA 配置
    # =========================
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
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

    # =========================
    # 6. 训练参数
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",  # 关闭其他日志，只留swanlab
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    # =========================
    # 7. 构建 DPOTrainer + 加入 SwanLab 回调
    # =========================
    print("[INFO] 构建 DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=0.1,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=512,
        max_length=1024,
        # 加入 SwanLab 回调
        callbacks=[SwanLabCallback()],
    )

    # =========================
    # 8. 开始训练
    # =========================
    print("[INFO] 开始 DPO 训练 ✅")
    trainer.train()

    # =========================
    # 9. 保存模型和 tokenizer
    # =========================
    print("[INFO] 保存模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 结束 SwanLab
    swanlab.finish()

    print("[✅ 训练完成！]")
    print(f"[INFO] 模型已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()