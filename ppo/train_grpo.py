# /root/autodl-tmp/ppo/train_grpo.py
# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse
from typing import List, Dict, Any, Optional

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# =========================
# 1. 基础工具
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_local_data(path: str) -> List[Dict[str, Any]]:
    """
    支持:
    1) .json
    2) .jsonl
    3) datasets.save_to_disk() 目录
    """
    if os.path.isdir(path):
        ds = load_from_disk(path)
        return [ds[i] for i in range(len(ds))]

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON 文件必须是 list[dict] 格式。")

    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    raise ValueError(f"不支持的数据路径: {path}")


def normalize_answer(text: str) -> str:
    text = str(text).strip()

    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()

    m = re.findall(r"Final\s*Answer\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".。")

    m = re.findall(r"Answer\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".。")

    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return re.sub(r"\s+", " ", text).strip()


def extract_gold_answer_from_solution(solution: str) -> str:
    solution = str(solution)
    m = re.findall(r"Answer\s*:\s*([^\n\r]+)", solution, flags=re.IGNORECASE)
    if m:
        return normalize_answer(m[-1])
    return normalize_answer(solution)


def extract_pred_answer(response_text: str) -> str:
    return normalize_answer(response_text)


def build_prompt(sample: Dict[str, Any]) -> str:
    problem = str(sample["problem"]).strip()
    question = str(sample["question"]).strip()

    prompt = (
        "You are a careful math reasoning assistant.\n"
        "Solve the problem step by step.\n"
        "At the end, output the final answer in the exact format:\n"
        "Answer: <number>\n\n"
        f"Problem: {problem}\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


def truncate_prompt(prompt: str, tokenizer, max_len: int = 1024) -> str:
    ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


# =========================
# 2. 数据处理
# =========================

def prepare_dataset(
    train_path: str,
    tokenizer,
    max_prompt_tokens: int = 1024,
    max_samples: Optional[int] = None,
    only_op10: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> Dataset:
    raw_data = load_local_data(train_path)

    processed = []
    for sample in raw_data:
        try:
            if only_op10 and int(sample["op"]) != 10:
                continue

            prompt = build_prompt(sample)
            prompt = truncate_prompt(prompt, tokenizer, max_len=max_prompt_tokens)
            gold_answer = extract_gold_answer_from_solution(sample["solution"])

            processed.append({
                "prompt": prompt,
                "gold_answer": gold_answer,
                "problem": sample["problem"],
                "question": sample["question"],
                "solution": sample["solution"],
                "op": sample["op"],
                "id": sample.get("id", ""),
            })

        except Exception as e:
            print(f"[WARN] 跳过样本: {e}")
            continue

    if shuffle:
        random.Random(seed).shuffle(processed)

    if max_samples is not None:
        processed = processed[:max_samples]

    if len(processed) == 0:
        raise ValueError("处理后数据为空，请检查 train_path 或 op 过滤。")

    print(f"✅ 最终训练样本数: {len(processed)}")
    return Dataset.from_list(processed)


# =========================
# 3. Reward 函数
# =========================

def _completion_to_text(completion) -> str:
    """
    兼容不同 completion 格式
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # 可能是 [{"role": "...", "content": "..."}]
        if len(completion) > 0 and isinstance(completion[0], dict):
            if "content" in completion[0]:
                return str(completion[0]["content"])
        return str(completion)

    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        return str(completion)

    return str(completion)


def accuracy_reward_func(completions, gold_answer, **kwargs):
    rewards = []
    for completion, gold in zip(completions, gold_answer):
        text = _completion_to_text(completion)
        pred = extract_pred_answer(text)
        gold_norm = normalize_answer(gold)
        reward = 1.0 if pred == gold_norm else 0.0
        rewards.append(reward)
    return rewards


def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        reward = 0.2 if re.search(r"Answer\s*:", text, flags=re.IGNORECASE) else 0.0
        rewards.append(reward)
    return rewards


def length_penalty_reward_func(completions, **kwargs):
    """
    可选：过长轻微惩罚，避免胡乱拉长
    """
    rewards = []
    for completion in completions:
        text = _completion_to_text(completion)
        reward = -0.1 if len(text) > 1200 else 0.0
        rewards.append(reward)
    return rewards


# =========================
# 4. 主训练逻辑
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_op10_output")

    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # 数据侧 prompt 截断
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)

    # GRPO
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)

    # generation / group sampling
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)

    # misc
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_op_filter", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("🚀 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 处理训练数据...")
    train_dataset = prepare_dataset(
        train_path=args.train_path,
        tokenizer=tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_samples=args.max_samples,
        only_op10=(not args.no_op_filter),
        shuffle=True,
        seed=args.seed,
    )

    print("train_dataset.column_names =", train_dataset.column_names)
    print("train_dataset[0] =", train_dataset[0])

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
    )

    print("🚀 初始化 GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[
            format_reward_func,
            accuracy_reward_func,
            length_penalty_reward_func,
        ],
        processing_class=tokenizer,
    )

    print("🚀 开始 GRPO 训练...")
    trainer.train()

    print("💾 保存最终模型...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("✅ GRPO 训练完成")
    print(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()