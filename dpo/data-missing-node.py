#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher

from openai import OpenAI

# =========================================================
# 配置区
# =========================================================
API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

INPUT_JSON_PATH = "/root/autodl-tmp/train.json"
OUTPUT_DIR = "./dpo_outputs_missing_only_1"

# 设为 0 表示全量
SAMPLE_COUNT = 100

SEED = 42
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5
MAX_OUTPUT_TOKENS = 1200
TEMPERATURE = 0.2

SORT_BY_LENGTH = True

# 质量过滤阈值
MIN_REJECTED_ABS_LEN = 40
MAX_TEXT_PREVIEW = 300

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# =========================================================
# Prompt
# =========================================================
SYSTEM_PROMPT = """
You are creating high-quality DPO preference data for math word-problem reasoning.

You will receive:
- a problem
- optionally a question
- a correct solution

Your task:
Create ONE rejected solution of type **missing_nodes** based ONLY on the correct solution.

CORE OBJECTIVE:
The rejected solution must contain a **non-recoverable logical gap**.
A reader should NOT be able to fully reconstruct the reasoning without guessing.

missing_nodes requirements (STRICT):
- Remove or skip **2–3 CRITICAL reasoning steps that are necessary for solving**
- The removed steps must break the **dependency chain**, not just shorten it
- The reasoning must become **logically incomplete**, not just concise

MANDATORY gap types (at least ONE must happen):
- Use an intermediate variable or result that was NEVER derived
- Skip the construction of a key equation but still use it
- Jump from partial setup directly to a later result without justification
- Omit the step that connects two dependent quantities

DO NOT create weak shortcuts:
- Do NOT only merge steps or simplify expressions
- Do NOT remove redundant or obvious steps
- Do NOT produce a valid but shorter reasoning
- The gap must make the reasoning **invalid or unjustified**

Style constraints:
- Keep wording, language, tone, and structure almost identical
- Change as little text as possible (minimal edit principle)
- Preserve all original variable names
- Do NOT introduce new variables or symbols
- Keep the same sentence flow whenever possible

Additional strict rules:
- Keep the original language
- Do NOT fix or improve the solution
- Do NOT reorder the reasoning
- Do NOT add explanations, comments, or hints
- Do NOT use markdown, bullet points, or formatting
- Do NOT include any extra text outside the JSON

Quality check (VERY IMPORTANT):
Before output, ensure:
- At least one variable or equation is used without being properly derived
- There is a clear missing dependency that cannot be trivially inferred
- The reasoning chain is broken (not just shortened)

Return exactly and only:
{
  "missing_nodes": {
    "rejected": "..."
  }
}
""".strip()

# =========================================================
# 路径
# =========================================================
RAW_CACHE_PATH = os.path.join(OUTPUT_DIR, "raw_generations.jsonl")
FAILED_CACHE_PATH = os.path.join(OUTPUT_DIR, "failed_generations.jsonl")
QC_REPORT_PATH = os.path.join(OUTPUT_DIR, "qc_report.json")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "run_summary.json")

# =========================================================
# 工具函数
# =========================================================
def stable_item_id(item: Dict) -> str:
    if item.get("id") is not None and str(item.get("id")).strip():
        return str(item["id"]).strip()

    base = "\n".join([
        str(item.get("problem", "")).strip(),
        str(item.get("question", "")).strip(),
        str(item.get("solution", "")).strip(),
    ])
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def safe_strip(x: Any) -> str:
    return "" if x is None else str(x).strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def preview_text(text: str, max_len: int = MAX_TEXT_PREVIEW) -> str:
    text = text.replace("\n", "\\n")
    return text[:max_len] + ("..." if len(text) > max_len else "")


def append_jsonl(path: str, obj: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def robust_json_loads(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        return json.loads(candidate)

    raise ValueError("Cannot parse JSON from model output.")


def extract_final_number(text: str) -> Optional[str]:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


def split_sentences(text: str) -> List[str]:
    # 针对你这种英文 CoT，按句号切分基本够用
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def extract_defined_vars(text: str) -> set:
    vars_found = set()

    # Define xxx as X;
    for m in re.finditer(r"\bDefine\b.*?\bas\s+([A-Za-z])\b", text):
        vars_found.add(m.group(1))

    # X =
    for m in re.finditer(r"\b([A-Za-z])\s*=", text):
        vars_found.add(m.group(1))

    return vars_found


def contains_undefined_variable_signal(chosen: str, rejected: str) -> bool:
    chosen_vars = extract_defined_vars(chosen)
    rejected_vars = extract_defined_vars(rejected)

    # 如果 rejected 中出现了大写/小写单字母变量，但定义集明显缩水，通常说明断链了
    all_used = set(re.findall(r"\b[A-Za-z]\b", rejected))
    undefined_used = all_used - rejected_vars

    # 只要存在在 chosen 中本来出现、但 rejected 没定义却使用的变量，认为是好信号
    broken = [v for v in undefined_used if v in chosen_vars]
    return len(broken) > 0


def looks_like_equivalent_compression(chosen: str, rejected: str) -> bool:
    """
    过滤“只是更短但仍是完整等价表达”的 rejected。
    启发式，不要求完美。
    """
    # 常见情况：直接把多步加总压成一行
    if "Answer:" in rejected and "Answer:" in chosen:
        chosen_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", chosen)
        rejected_nums = re.findall(r"[-+]?\d+(?:\.\d+)?", rejected)

        # 若最终数字一样，且 rejected 没明显断链信号，可能只是压缩
        if chosen_nums and rejected_nums and chosen_nums[-1] == rejected_nums[-1]:
            if not contains_undefined_variable_signal(chosen, rejected):
                # 再看 rejected 是否大量出现“直接算式”
                if re.search(r"=\s*[-+*/()\d\sA-Za-z]+=\s*[-+]?\d+", rejected):
                    return True

    return False


def has_shortcut_solve_pattern(rejected: str) -> bool:
    patterns = [
        r"We know .*?, so .*?\. Solution:",
        r"We know .*?, so .*?\. Answer:",
        r"Solve:\s*",
        r"thus\s+[A-Za-z]\s*=",
        r"So\s+[A-Za-z]\s*=\s*[-+]?\d+",
    ]
    return any(re.search(p, rejected, flags=re.IGNORECASE) for p in patterns)


# =========================================================
# 数据读取
# =========================================================
def load_data(path: str, sample_n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是 list[dict] 格式。")

    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not safe_strip(item.get("problem")):
            continue
        if not safe_strip(item.get("solution")):
            continue
        cleaned.append(item)

    data = cleaned

    if sample_n > 0 and sample_n < len(data):
        data = random.sample(data, sample_n)

    if SORT_BY_LENGTH:
        data.sort(key=lambda x: len(safe_strip(x.get("problem"))))

    return data


# =========================================================
# Prompt 构造
# =========================================================
def build_user_prompt(item: Dict) -> str:
    parts = [
        "Problem:",
        safe_strip(item["problem"])
    ]

    if safe_strip(item.get("question")):
        parts.extend([
            "",
            "Question:",
            safe_strip(item["question"])
        ])

    parts.extend([
        "",
        "Correct solution:",
        safe_strip(item["solution"])
    ])
    return "\n".join(parts)


def build_prompt_for_dpo(item: Dict) -> str:
    problem = safe_strip(item.get("problem"))
    question = safe_strip(item.get("question"))

    if question:
        user_content = f"Problem:\n{problem}\n\nQuestion:\n{question}"
    else:
        user_content = f"Problem:\n{problem}"

    return f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"


# =========================================================
# 输出校验
# =========================================================
def validate_generation_schema(data: Dict) -> None:
    if "missing_nodes" not in data:
        raise ValueError("Missing top-level key: missing_nodes")
    if not isinstance(data["missing_nodes"], dict):
        raise ValueError("missing_nodes must be an object.")
    if "rejected" not in data["missing_nodes"]:
        raise ValueError("Missing 'rejected' in missing_nodes")
    if not isinstance(data["missing_nodes"]["rejected"], str):
        raise ValueError("missing_nodes.rejected must be a string.")


def heuristic_check_missing_nodes(chosen: str, rejected: str) -> Tuple[bool, str]:
    chosen = safe_strip(chosen)
    rejected = safe_strip(rejected)

    if not rejected:
        return False, "empty_rejected"

    if rejected == chosen:
        return False, "identical_to_chosen"

    if len(rejected) < MIN_REJECTED_ABS_LEN:
        return False, "too_short_absolute"

    sim = similarity(chosen, rejected)
    if sim > 0.995:
        return False, "too_similar"
    if sim < 0.45:
        return False, "too_different"

    chosen_len = len(chosen)
    rejected_len = len(rejected)
    ratio = rejected_len / max(chosen_len, 1)

    # missing_nodes 必须明显更短
    if ratio > 0.92:
        return False, "missing_nodes_not_short_enough"

    # 句子数最好更少
    chosen_sents = split_sentences(chosen)
    rejected_sents = split_sentences(rejected)
    if len(rejected_sents) >= len(chosen_sents):
        return False, "sentence_count_not_reduced"

    # 不能出现元说明
    bad_markers = [
        "here is the json",
        "missing_nodes",
        "error type",
        "rejected solution"
    ]
    lower = rejected.lower()
    for marker in bad_markers:
        if marker in lower:
            return False, "contains_meta_explanation"

    # 过滤“只是等价压缩”
    if looks_like_equivalent_compression(chosen, rejected):
        return False, "equivalent_compression"

    # 希望至少满足一个“真 missing”信号
    broken_signal = contains_undefined_variable_signal(chosen, rejected)
    shortcut_signal = has_shortcut_solve_pattern(rejected)

    if not (broken_signal or shortcut_signal):
        return False, "no_clear_missing_signal"

    return True, "ok"


# =========================================================
# 模型调用
# =========================================================
def call_model_once(item: Dict) -> Tuple[Optional[Dict], Dict]:
    user_prompt = build_user_prompt(item)
    debug_meta = {
        "attempts": [],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        attempt_info = {"attempt": attempt, "status": "unknown"}

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )

            content = resp.choices[0].message.content or ""
            attempt_info["raw_preview"] = preview_text(content)

            usage = getattr(resp, "usage", None)
            if usage is not None:
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
                debug_meta["usage"]["prompt_tokens"] += pt
                debug_meta["usage"]["completion_tokens"] += ct
                debug_meta["usage"]["total_tokens"] += tt

            data = robust_json_loads(content)
            validate_generation_schema(data)

            attempt_info["status"] = "ok"
            debug_meta["attempts"].append(attempt_info)
            return data, debug_meta

        except Exception as e:
            attempt_info["status"] = "failed"
            attempt_info["error"] = str(e)
            debug_meta["attempts"].append(attempt_info)

            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
            else:
                return None, debug_meta

    return None, debug_meta


# =========================================================
# DPO 打包
# =========================================================
def pack_sample(item: Dict, rejected: str) -> Tuple[Optional[Dict], str]:
    chosen = safe_strip(item.get("solution"))
    rejected = safe_strip(rejected)

    ok, reason = heuristic_check_missing_nodes(chosen, rejected)
    if not ok:
        return None, reason

    sample = {
        "id": stable_item_id(item),
        "template": item.get("template"),
        "op": item.get("op"),
        "mode": item.get("mode"),
        "length": item.get("length"),
        "d": item.get("d"),
        "error_type": "missing_nodes",
        "prompt": build_prompt_for_dpo(item),
        "chosen": chosen,
        "rejected": rejected,
        "source_problem": safe_strip(item.get("problem")),
        "source_question": safe_strip(item.get("question")),
    }
    return sample, "ok"


# =========================================================
# 断点续跑
# =========================================================
def load_done_ids() -> set:
    done = set()
    for row in load_jsonl(RAW_CACHE_PATH):
        if row.get("status") == "ok" and row.get("id"):
            done.add(str(row["id"]))
    return done


# =========================================================
# 主流程
# =========================================================
def main():
    print("=" * 60)
    print("DPO 数据生成工具（missing_nodes only）")
    print(f"模型: {MODEL}")
    print(f"输入文件: {INPUT_JSON_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    data = load_data(INPUT_JSON_PATH, SAMPLE_COUNT)
    done_ids = load_done_ids()

    print(f"总样本数: {len(data)}")
    print(f"已完成样本: {len(done_ids)}")
    print(f"待处理样本: {sum(1 for x in data if stable_item_id(x) not in done_ids)}")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    processed = 0
    succeeded = 0
    failed = 0

    for idx, item in enumerate(data, 1):
        item_id = stable_item_id(item)

        if item_id in done_ids:
            continue

        processed += 1
        print(f"[{idx}/{len(data)}] 处理 {item_id} ...")

        result, meta = call_model_once(item)

        total_prompt_tokens += meta["usage"]["prompt_tokens"]
        total_completion_tokens += meta["usage"]["completion_tokens"]
        total_tokens += meta["usage"]["total_tokens"]

        if result is None:
            failed += 1
            append_jsonl(FAILED_CACHE_PATH, {
                "id": item_id,
                "status": "failed",
                "item": item,
                "meta": meta,
            })
            append_jsonl(RAW_CACHE_PATH, {
                "id": item_id,
                "status": "failed",
                "item": item,
                "meta": meta,
            })
            last_err = meta["attempts"][-1]["error"] if meta["attempts"] else "unknown"
            print(f"  ❌ 失败: {last_err}")
        else:
            succeeded += 1
            append_jsonl(RAW_CACHE_PATH, {
                "id": item_id,
                "status": "ok",
                "item": item,
                "generation": result,
                "meta": meta,
            })
            print("  ✅ 成功")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n" + "=" * 60)
    print("整理输出文件中...")

    missing_ds: List[Dict] = []

    qc_report = {
        "kept": {
            "missing_nodes": 0
        },
        "dropped": {
            "missing_nodes": {}
        }
    }

    def add_drop_reason(reason: str):
        qc_report["dropped"]["missing_nodes"][reason] = \
            qc_report["dropped"]["missing_nodes"].get(reason, 0) + 1

    for row in load_jsonl(RAW_CACHE_PATH):
        if row.get("status") != "ok":
            continue

        item = row["item"]
        gen = row["generation"]

        sample, reason = pack_sample(item, gen["missing_nodes"]["rejected"])
        if sample is not None:
            missing_ds.append(sample)
            qc_report["kept"]["missing_nodes"] += 1
        else:
            add_drop_reason(reason)

    random.shuffle(missing_ds)

    with open(os.path.join(OUTPUT_DIR, "dpo_missing_nodes.json"), "w", encoding="utf-8") as f:
        json.dump(missing_ds, f, ensure_ascii=False, indent=2)

    with open(QC_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(qc_report, f, ensure_ascii=False, indent=2)

    summary = {
        "model": MODEL,
        "input_json_path": INPUT_JSON_PATH,
        "output_dir": OUTPUT_DIR,
        "sample_count_config": SAMPLE_COUNT,
        "temperature": TEMPERATURE,
        "max_retries": MAX_RETRIES,
        "processed_this_run": processed,
        "succeeded_this_run": succeeded,
        "failed_this_run": failed,
        "token_usage_this_run": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens
        },
        "final_dataset_size": {
            "missing_nodes": len(missing_ds)
        }
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("生成完成")
    print(f"本轮成功样本: {succeeded}")
    print(f"本轮失败样本: {failed}")
    print(f"本轮 token 总量: {total_tokens}")
    print("-" * 60)
    print(f"missing_nodes: {len(missing_ds)}")
    print("-" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()