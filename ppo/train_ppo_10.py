# train_ppo_10_fixed.py
# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)


# =========================
# 1. 基础工具
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    # 先抓 boxed
    m = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if m:
        return m[-1].strip()

    # 再抓 Final Answer:
    m = re.findall(r"Final\s*Answer\s*[:：]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        ans = m[-1].strip().rstrip(".。")
        return ans

    # 再抓 Answer:
    m = re.findall(r"Answer\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if m:
        ans = m[-1].strip().rstrip(".。")
        return ans

    # 最后抓最后一个数字
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if nums:
        return nums[-1].strip()

    return re.sub(r"\s+", " ", text).strip()


def extract_gold_answer_from_solution(solution: str) -> str:
    """
    从 solution 里提取标准答案
    你的数据格式末尾一般是:
    Answer: 1.
    """
    solution = str(solution)

    m = re.findall(r"Answer\s*:\s*([^\n\r]+)", solution, flags=re.IGNORECASE)
    if m:
        return normalize_answer(m[-1])

    # 如果没有明确 Answer:，退化到 normalize_answer
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


def reward_fn(pred_text: str, gold_text: str) -> float:
    """
    先保留你原来的 0/1 reward 逻辑，方便和原实验对齐。
    后面你如果想做稠密奖励，我再给你升级。
    """
    pred = extract_pred_answer(pred_text)
    gold = normalize_answer(gold_text)
    return 1.0 if pred == gold else 0.0


# =========================
# 2. 数据处理
# =========================

def prepare_dataset(
    train_path: str,
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
            gold_answer = extract_gold_answer_from_solution(sample["solution"])

            processed.append({
                "query": prompt,
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


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# =========================
# 3. 解码工具
# =========================

def safe_batch_decode(tokenizer, responses):
    """
    兼容不同 TRL 版本 generate 返回格式：
    - list[Tensor]
    - Tensor
    - list[list[int]]
    """
    texts = []

    if isinstance(responses, torch.Tensor):
        return tokenizer.batch_decode(responses, skip_special_tokens=True)

    for resp in responses:
        if isinstance(resp, torch.Tensor):
            ids = resp.detach().cpu().tolist()
        else:
            ids = resp
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


# =========================
# 4. 主训练逻辑
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ppo_op10_output")

    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    # PPO
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--target_kl", type=float, default=0.1)

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # misc
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_op_filter", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("🚀 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 处理训练数据...")
    train_dataset = prepare_dataset(
        train_path=args.train_path,
        max_samples=args.max_samples,
        only_op10=(not args.no_op_filter),
        shuffle=True,
        seed=args.seed,
    )

    print("train_dataset.column_names =", train_dataset.column_names)
    print("train_dataset[0] =", train_dataset[0])
    print("train_dataset[0]['gold_answer'] =", train_dataset[0]["gold_answer"])

    print("🚀 加载 PPO 模型...")
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    ref_model = create_reference_model(model)

    config = PPOConfig(
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        target_kl=args.target_kl,
        optimize_cuda_cache=True,
        seed=args.seed,
        log_with=None,
    )

    # 关键修改：这里不再把 dataset 交给 PPOTrainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # 关键修改：自己构造 DataLoader，确保 gold_answer 不丢
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_prompt": False,
    }

    print("🚀 开始 PPO 训练...")
    global_step = 0
    reward_history = []

    for batch in tqdm(train_dataloader, desc="PPO Training"):
        if "query" not in batch:
            raise KeyError(f"batch 中缺少 query，当前 keys={list(batch.keys())}")
        if "gold_answer" not in batch:
            raise KeyError(f"batch 中缺少 gold_answer，当前 keys={list(batch.keys())}")

        queries = batch["query"]
        gold_answers = batch["gold_answer"]

        query_tensors = []
        for q in queries:
            q_ids = tokenizer(
                q,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).input_ids[0]
            query_tensors.append(q_ids.to(ppo_trainer.accelerator.device))

        response_tensors = ppo_trainer.generate(
            query_tensors,
            **generation_kwargs
        )

        response_texts = safe_batch_decode(tokenizer, response_tensors)

        rewards = []
        for pred_text, gold_text in zip(response_texts, gold_answers):
            r = reward_fn(pred_text, gold_text)
            rewards.append(
                torch.tensor(float(r), dtype=torch.float32).to(ppo_trainer.accelerator.device)
            )
            reward_history.append(float(r))

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # 这里日志字段只是为了看，不影响训练
        ppo_trainer.log_stats(
            stats,
            {
                "query": queries,
                "response": response_texts,
                "gold_answer": gold_answers,
            },
            rewards
        )

        global_step += 1

        if global_step % 10 == 0:
            recent = reward_history[-100:] if len(reward_history) >= 100 else reward_history
            avg_reward = sum(recent) / max(1, len(recent))

            print(f"\n[step={global_step}] recent_avg_reward={avg_reward:.4f}")
            print("----- sample -----")
            print("QUERY:", queries[0][:300].replace("\n", " "))
            print("RESP :", response_texts[0][:300].replace("\n", " "))
            print("GOLD :", gold_answers[0])
            print("PRED :", extract_pred_answer(response_texts[0]))
            print("REWARD:", float(rewards[0]))
            print("------------------")

        if global_step % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ppo_trainer.model.pretrained_model.save_pretrained(ckpt_dir)
            # ppo_trainer.model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"💾 已保存 checkpoint 到: {ckpt_dir}")

    final_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    ppo_trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    final_avg_reward = sum(reward_history) / max(1, len(reward_history))
    print(f"\n✅ PPO 训练完成")
    print(f"最终平均 reward: {final_avg_reward:.4f}")
    print(f"模型已保存到: {final_dir}")


if __name__ == "__main__":
    main()


"""
运行示例：

python /root/autodl-tmp/ppo/train_ppo_10.py \
  --model_name_or_path /root/autodl-tmp/checkpoint/qwen2_0.5b_sft_op10/checkpoint-10339 \
  --train_path /root/autodl-tmp/data/half/train.json \
  --output_dir /root/autodl-tmp/checkpoint/ppo_op10_output \
  --max_samples 200 \
  --batch_size 4 \
  --mini_batch_size 2 \
  --learning_rate 1e-6 \
  --max_new_tokens 128 \
  --save_every 20 \
  --bf16
"""
