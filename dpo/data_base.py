#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把原始推理数据批量转换成 DPO 数据：
- prompt   = problem + question
- chosen   = 原始 solution
- rejected = 仅修改 solution 最后答案数值后的错误版本

支持输入：
1. JSON  : 整体是一个 list[dict]
2. JSONL : 每行一个 dict

输出：
- 默认输出 JSONL，每行一个 DPO 样本
"""

import os
import re
import json
import random
from typing import List, Dict, Any, Optional


# =========================
# 配置区：直接改这里
# =========================
INPUT_PATH = r"/root/autodl-tmp/dpo_sampled_data.json"          # 你的原始数据路径：json 或 jsonl
OUTPUT_PATH = r"/root/autodl-tmp/dpo_output.jsonl"   # 输出 DPO 数据路径
RANDOM_SEED = 42
OUTPUT_FORMAT = "jsonl"               # "jsonl" 或 "json"

# 是否把 problem 和 question 拼成标准 prompt
USE_STANDARD_PROMPT = True

# 是否保留原字段，方便后续追踪
KEEP_META_FIELDS = True

# 错误答案采样范围：优先从正确答案附近采样
OFFSETS = [-2, -1, 1, 2]

# 是否允许负数答案
ALLOW_NEGATIVE_ANSWER = False


# =========================
# 工具函数
# =========================
def load_data(path: str) -> List[Dict[str, Any]]:
    """读取 json 或 jsonl 数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在: {path}")

    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        data.append(item)
                    else:
                        print(f"[WARN] 第 {line_num} 行不是 dict，已跳过")
                except Exception as e:
                    print(f"[WARN] 第 {line_num} 行 JSON 解析失败，已跳过: {e}")
        return data

    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
        else:
            raise ValueError("JSON 文件内容必须是 dict 或 list[dict]")
    else:
        raise ValueError("仅支持 .json 或 .jsonl 输入文件")


def save_data(data: List[Dict[str, Any]], path: str, fmt: str = "jsonl") -> None:
    """保存为 json 或 jsonl"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError("fmt 只能是 'jsonl' 或 'json'")


def build_prompt(problem: str, question: str) -> str:
    """构造 prompt"""
    if USE_STANDARD_PROMPT:
        return (
            "You are given a word problem. Solve it step by step and give the final answer.\n\n"
            f"Problem: {problem}\n"
            f"Question: {question}"
        )
    return f"{problem}\nQuestion: {question}"


def extract_final_answer_number(solution: str) -> Optional[int]:
    """
    从 solution 末尾提取最后答案数值
    优先匹配：
    - Answer: 1.
    - Answer: 1
    - answer: 1
    其次回退到最后一个独立整数
    """
    if not solution or not isinstance(solution, str):
        return None

    # 先找显式 Answer:
    patterns = [
        r"Answer\s*:\s*(-?\d+)\s*\.?\s*$",
        r"answer\s*:\s*(-?\d+)\s*\.?\s*$",
        r"Final answer\s*:\s*(-?\d+)\s*\.?\s*$",
        r"final answer\s*:\s*(-?\d+)\s*\.?\s*$",
    ]
    for p in patterns:
        m = re.search(p, solution)
        if m:
            return int(m.group(1))

    # 回退：找全文最后一个整数
    nums = re.findall(r"-?\d+", solution)
    if nums:
        return int(nums[-1])

    return None


def sample_wrong_answer(correct: int) -> int:
    """从正确答案附近采样一个错误答案"""
    candidates = []
    for offset in OFFSETS:
        val = correct + offset
        if not ALLOW_NEGATIVE_ANSWER and val < 0:
            continue
        if val != correct:
            candidates.append(val)

    # 如果附近没候选，就兜底
    if not candidates:
        candidates = [correct + 1] if correct >= 0 else [correct - 1]

    return random.choice(candidates)


def replace_final_answer(solution: str, wrong_answer: int) -> Optional[str]:
    """
    只替换最后答案，不改前文推理。
    优先替换显式的 Answer: n
    若找不到，则尝试替换文本最后一个整数。
    """
    if not solution or not isinstance(solution, str):
        return None

    explicit_patterns = [
        r"(Answer\s*:\s*)(-?\d+)(\s*\.?\s*$)",
        r"(answer\s*:\s*)(-?\d+)(\s*\.?\s*$)",
        r"(Final answer\s*:\s*)(-?\d+)(\s*\.?\s*$)",
        r"(final answer\s*:\s*)(-?\d+)(\s*\.?\s*$)",
    ]

    for p in explicit_patterns:
        if re.search(p, solution):
            return re.sub(p, rf"\g<1>{wrong_answer}\g<3>", solution)

    # 回退：替换最后一个独立整数
    matches = list(re.finditer(r"-?\d+", solution))
    if not matches:
        return None

    last = matches[-1]
    start, end = last.span()
    new_solution = solution[:start] + str(wrong_answer) + solution[end:]
    return new_solution


def convert_item_to_dpo(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """把单条原始样本转成 DPO 样本"""
    problem = item.get("problem", "")
    question = item.get("question", "")
    solution = item.get("solution", "")

    if not problem or not question or not solution:
        return None

    correct_answer = extract_final_answer_number(solution)
    if correct_answer is None:
        return None

    wrong_answer = sample_wrong_answer(correct_answer)
    rejected = replace_final_answer(solution, wrong_answer)
    if rejected is None:
        return None

    dpo_item = {
        "prompt": build_prompt(problem, question),
        "chosen": solution,
        "rejected": rejected,
    }

    if KEEP_META_FIELDS:
        for k in ["id", "op", "template", "mode", "length", "d"]:
            if k in item:
                dpo_item[k] = item[k]
        dpo_item["correct_answer"] = correct_answer
        dpo_item["wrong_answer"] = wrong_answer

    return dpo_item


# =========================
# 主程序
# =========================
def main():
    random.seed(RANDOM_SEED)

    raw_data = load_data(INPUT_PATH)
    print(f"[INFO] 读入原始样本数: {len(raw_data)}")

    dpo_data = []
    skipped = 0

    for idx, item in enumerate(raw_data, 1):
        try:
            out = convert_item_to_dpo(item)
            if out is None:
                skipped += 1
                print(f"[WARN] 第 {idx} 条处理失败，已跳过。id={item.get('id', 'N/A')}")
                continue
            dpo_data.append(out)
        except Exception as e:
            skipped += 1
            print(f"[WARN] 第 {idx} 条处理异常，已跳过。id={item.get('id', 'N/A')} error={e}")

    save_data(dpo_data, OUTPUT_PATH, OUTPUT_FORMAT)

    print("=" * 60)
    print(f"[INFO] 成功输出 DPO 样本数: {len(dpo_data)}")
    print(f"[INFO] 跳过样本数: {skipped}")
    print(f"[INFO] 输出文件: {OUTPUT_PATH}")
    print("=" * 60)

    # 打印前 2 条看看效果
    for i, sample in enumerate(dpo_data[:2], 1):
        print(f"\n[Sample {i}]")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()