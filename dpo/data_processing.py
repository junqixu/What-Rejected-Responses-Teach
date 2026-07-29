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
OUTPUT_DIR = "./dpo_outputs"

# 设为 0 表示全量
SAMPLE_COUNT = 1000

SEED = 42
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.5
MAX_OUTPUT_TOKENS = 1800
TEMPERATURE = 0.3

# 是否按 problem 长度排序，增加缓存命中概率
SORT_BY_LENGTH = True

# 质量过滤阈值
MIN_REJECTED_ABS_LEN = 50
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
You are creating DPO preference data for math word-problem reasoning.

You will receive:
- a problem
- optionally a question
- a correct solution

Your task:
Create THREE rejected solutions from the correct solution, preserving wording, language, tone, and structure as much as possible while changing as little as possible.

ERROR TYPE 1: computation_error
- Keep reasoning structure almost identical
- Introduce ONLY arithmetic / numeric mistakes
- Keep dependency graph and referenced variables unchanged
- Final answer must be wrong
- Intermediate numeric steps must reflect the numeric mistake

ERROR TYPE 2: dependency_mismatch
- Keep text natural and plausible
- Introduce ONLY one core dependency/reference mistake
- Use a wrong parent variable / wrong branch / wrong referenced quantity
- Do NOT make it garbled or obviously broken
- Final answer should usually be wrong

ERROR TYPE 3: missing_nodes
- Remove or shortcut 2-3 key intermediate steps
- Keep the solution shorter but still natural
- The reasoning should be meaningfully incomplete
- Prefer final answer to be wrong, but if the reasoning is clearly incomplete, a correct final answer is still acceptable

CRITICAL CONSTRAINTS:
- Keep the original language
- Keep the original style
- Do NOT rewrite from scratch
- Do NOT add labels or explanations inside the rejected text
- Do NOT use markdown
- Return strict JSON only

Return exactly:
{
  "computation_error": {"rejected": "..."},
  "dependency_mismatch": {"rejected": "..."},
  "missing_nodes": {"rejected": "..."}
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
    """
    为样本生成稳定 ID。
    优先使用原始 id，否则基于 problem/question/solution 生成 md5。
    """
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
    """
    先正常 json.loads，失败后尝试从文本中提取最外层 JSON 对象。
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    # 尝试截取第一个大括号对象
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        return json.loads(candidate)

    raise ValueError("Cannot parse JSON from model output.")


def extract_final_number(text: str) -> Optional[str]:
    """
    简单提取最后出现的数字，作为粗糙 final answer proxy。
    """
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    return nums[-1]


# =========================================================
# 数据读取
# =========================================================
def load_data(path: str, sample_n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 必须是 list[dict] 格式。")

    cleaned = []
    for i, item in enumerate(data):
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
    required_keys = ["computation_error", "dependency_mismatch", "missing_nodes"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing top-level key: {key}")
        if not isinstance(data[key], dict):
            raise ValueError(f"{key} must be an object.")
        if "rejected" not in data[key]:
            raise ValueError(f"Missing 'rejected' in {key}")
        if not isinstance(data[key]["rejected"], str):
            raise ValueError(f"{key}.rejected must be a string.")


def heuristic_check_rejected(
    chosen: str,
    rejected: str,
    error_type: str
) -> Tuple[bool, str]:
    """
    只做启发式过滤，不追求完美，但尽量去掉明显脏样本。
    """
    chosen = safe_strip(chosen)
    rejected = safe_strip(rejected)

    if not rejected:
        return False, "empty_rejected"

    if rejected == chosen:
        return False, "identical_to_chosen"

    if len(rejected) < MIN_REJECTED_ABS_LEN:
        return False, "too_short_absolute"

    sim = similarity(chosen, rejected)

    # 太像：几乎没改
    if sim > 0.995:
        return False, "too_similar"

    # 太不像：说明不是“局部改错”，可能整段重写
    if sim < 0.35:
        return False, "too_different"

    chosen_len = len(chosen)
    rejected_len = len(rejected)
    ratio = rejected_len / max(chosen_len, 1)

    # 各类型的粗糙长度控制
    if error_type == "missing_nodes":
        # missing_nodes 应该更短
        if ratio > 0.98:
            return False, "missing_nodes_not_shorter"
    else:
        # 另外两类不应该短得离谱
        if ratio < 0.45:
            return False, f"{error_type}_too_short_ratio"

    # 不能出现明显的元说明
    bad_markers = [
        "here is the json",
        "computation_error",
        "dependency_mismatch",
        "missing_nodes",
        "error type",
        "rejected solution"
    ]
    lower = rejected.lower()
    for marker in bad_markers:
        if marker in lower:
            return False, "contains_meta_explanation"

    # computation_error：最好最终数字和 chosen 不同
    if error_type == "computation_error":
        a = extract_final_number(chosen)
        b = extract_final_number(rejected)
        if a is not None and b is not None and a == b:
            return False, "computation_same_final_number"

    return True, "ok"


# =========================================================
# 模型调用
# =========================================================
def call_model_once(item: Dict) -> Tuple[Optional[Dict], Dict]:
    """
    返回:
    - result: 成功解析且 schema 合法时返回 dict，否则 None
    - meta:   记录调试信息 / token / 错误信息
    """
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
def pack_sample(item: Dict, rejected: str, error_type: str) -> Tuple[Optional[Dict], str]:
    chosen = safe_strip(item.get("solution"))
    rejected = safe_strip(rejected)

    ok, reason = heuristic_check_rejected(chosen, rejected, error_type)
    if not ok:
        return None, reason

    sample = {
        "id": stable_item_id(item),
        "template": item.get("template"),
        "op": item.get("op"),
        "mode": item.get("mode"),
        "length": item.get("length"),
        "d": item.get("d"),
        "error_type": error_type,
        "prompt": build_prompt_for_dpo(item),
        "chosen": chosen,
        "rejected": rejected,

        # 方便后续排查与人工抽检
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
    print("DPO 数据生成工具（增强版）")
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

    # -----------------------------------------
    # 逐条生成并缓存
    # -----------------------------------------
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

    # -----------------------------------------
    # 二次整理输出
    # -----------------------------------------
    print("\n" + "=" * 60)
    print("整理输出文件中...")

    compute_ds: List[Dict] = []
    dep_ds: List[Dict] = []
    missing_ds: List[Dict] = []

    qc_report = {
        "kept": {
            "computation_error": 0,
            "dependency_mismatch": 0,
            "missing_nodes": 0
        },
        "dropped": {
            "computation_error": {},
            "dependency_mismatch": {},
            "missing_nodes": {}
        }
    }

    def add_drop_reason(error_type: str, reason: str):
        qc_report["dropped"][error_type][reason] = qc_report["dropped"][error_type].get(reason, 0) + 1

    for row in load_jsonl(RAW_CACHE_PATH):
        if row.get("status") != "ok":
            continue

        item = row["item"]
        gen = row["generation"]

        # computation_error
        sample, reason = pack_sample(item, gen["computation_error"]["rejected"], "computation_error")
        if sample is not None:
            compute_ds.append(sample)
            qc_report["kept"]["computation_error"] += 1
        else:
            add_drop_reason("computation_error", reason)

        # dependency_mismatch
        sample, reason = pack_sample(item, gen["dependency_mismatch"]["rejected"], "dependency_mismatch")
        if sample is not None:
            dep_ds.append(sample)
            qc_report["kept"]["dependency_mismatch"] += 1
        else:
            add_drop_reason("dependency_mismatch", reason)

        # missing_nodes
        sample, reason = pack_sample(item, gen["missing_nodes"]["rejected"], "missing_nodes")
        if sample is not None:
            missing_ds.append(sample)
            qc_report["kept"]["missing_nodes"] += 1
        else:
            add_drop_reason("missing_nodes", reason)

    # 混合集
    mixed = compute_ds + dep_ds + missing_ds
    random.shuffle(mixed)

    # 保存
    with open(os.path.join(OUTPUT_DIR, "dpo_compute_error.json"), "w", encoding="utf-8") as f:
        json.dump(compute_ds, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "dpo_dependency_mismatch.json"), "w", encoding="utf-8") as f:
        json.dump(dep_ds, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "dpo_missing_nodes.json"), "w", encoding="utf-8") as f:
        json.dump(missing_ds, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "dpo_mixed.json"), "w", encoding="utf-8") as f:
        json.dump(mixed, f, ensure_ascii=False, indent=2)

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
            "computation_error": len(compute_ds),
            "dependency_mismatch": len(dep_ds),
            "missing_nodes": len(missing_ds),
            "mixed": len(mixed)
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
    print(f"computation_error:   {len(compute_ds)}")
    print(f"dependency_mismatch: {len(dep_ds)}")
    print(f"missing_nodes:       {len(missing_ds)}")
    print(f"mixed:               {len(mixed)}")
    print("-" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()