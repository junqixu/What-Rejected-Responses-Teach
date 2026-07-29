import os
import re
import json
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# =========================
# 配置区
# =========================
MODEL_PATH = "/root/autodl-tmp/checkpoint/grpo_op10_output"
TOKENIZER_PATH = "Qwen/Qwen2-0.5B"
DATA_PATH = "/root/autodl-tmp/data/half/validation.json"

OUTPUT_DIR = "/root/autodl-tmp/passk_eval_by_op"
os.makedirs(OUTPUT_DIR, exist_ok=True)

K_LIST = [1, 8, 16, 32]
K_MAX = max(K_LIST)

MAX_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.95

TEST_SIZE = 0.05
SEED = 42
EPS = 1e-6


# =========================
# 工具函数
# =========================
def extract_last_number(text):
    if text is None:
        return None
    text = str(text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None


def build_prompt(example):
    problem = str(example.get("problem", "")).strip()
    question = str(example.get("question", "")).strip()

    prompt = (
        "You are given a math word problem. "
        "Solve it and give the final numeric answer at the end.\n\n"
        f"Problem: {problem}\n"
        f"Question: {question}\n"
        "Answer: "
    )
    return prompt


def get_ground_truth(example):
    if "answer" in example and example["answer"] is not None:
        ans = extract_last_number(example["answer"])
        if ans is not None:
            return ans

    if "solution" in example and example["solution"] is not None:
        ans = extract_last_number(example["solution"])
        if ans is not None:
            return ans

    return None


def is_correct(pred, true, eps=EPS):
    if pred is None or true is None:
        return False
    return abs(pred - true) < eps


def get_op(example):
    # 你的数据就是这种格式："op": 10
    if "op" not in example or example["op"] is None:
        return None
    try:
        return int(example["op"])
    except Exception:
        return None


def get_op_bucket(op):
    if op is None:
        return "unknown"
    if 2 <= op <= 10:
        return "op_2_10"
    elif 11 <= op <= 14:
        return "op_11_14"
    elif 15 <= op <= 20:
        return "op_15_20"
    else:
        return f"op_other_{op}"


# =========================
# 1. 加载模型
# =========================
print("加载模型...")
llm = LLM(
    model=MODEL_PATH,
    tokenizer=TOKENIZER_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    disable_log_stats=True,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)
print("模型加载成功！")


# =========================
# 2. 加载数据
# =========================
print("加载数据...")
ds = load_dataset("json", data_files=DATA_PATH)
ds = ds["train"].train_test_split(test_size=TEST_SIZE, seed=SEED)
test_ds = ds["test"]
print(f"测试集大小: {len(test_ds)}")


# =========================
# 3. 构造 prompts / truths / op
# =========================
prompts = []
truths = []
meta_infos = []

bucket_counts = defaultdict(int)

for idx, example in enumerate(test_ds):
    prompt = build_prompt(example)
    true = get_ground_truth(example)
    op = get_op(example)
    bucket = get_op_bucket(op)

    prompts.append(prompt)
    truths.append(true)

    meta_infos.append({
        "index": idx,
        "problem": example.get("problem", ""),
        "question": example.get("question", ""),
        "solution": example.get("solution", ""),
        "gold_answer": true,
        "op": op,
        "bucket": bucket,
    })

    bucket_counts[bucket] += 1

print("各 bucket 样本数：")
for b, c in sorted(bucket_counts.items()):
    print(f"  {b}: {c}")

print(f"\n开始批量生成：共 {len(prompts)} 条，每题采样 {K_MAX} 个候选...")


# =========================
# 4. 批量生成
# =========================
sampling_params = SamplingParams(
    n=K_MAX,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    stop_token_ids=[151645],
)

outputs = llm.generate(prompts, sampling_params)


# =========================
# 5. 统计 overall 和 by-bucket
# =========================
overall_correct = {k: 0 for k in K_LIST}
bucket_correct = defaultdict(lambda: {k: 0 for k in K_LIST})
bucket_total = defaultdict(int)

all_results = []

for i, out in enumerate(outputs):
    true = truths[i]
    meta = meta_infos[i]
    bucket = meta["bucket"]

    bucket_total[bucket] += 1

    candidate_results = []
    for cand_idx, cand in enumerate(out.outputs):
        pred_text = cand.text.strip()
        pred_num = extract_last_number(pred_text)
        correct = is_correct(pred_num, true)

        candidate_results.append({
            "candidate_id": cand_idx,
            "text": pred_text,
            "pred_number": pred_num,
            "is_correct": correct,
        })

    per_sample_passk = {}
    for k in K_LIST:
        hit = any(c["is_correct"] for c in candidate_results[:k])
        per_sample_passk[f"pass@{k}"] = hit

        if hit:
            overall_correct[k] += 1
            bucket_correct[bucket][k] += 1

    all_results.append({
        "index": meta["index"],
        "problem": meta["problem"],
        "question": meta["question"],
        "solution": meta["solution"],
        "gold_answer": true,
        "op": meta["op"],
        "bucket": bucket,
        "candidates": candidate_results,
        "metrics": per_sample_passk,
    })


# =========================
# 6. 输出结果
# =========================
total = len(test_ds)
summary = {
    "overall": {},
    "by_bucket": {}
}

print("\n" + "=" * 70)
print("Overall pass@k:")
for k in K_LIST:
    score = overall_correct[k] / total if total > 0 else 0.0
    summary["overall"][f"answer-pass@{k}"] = score
    print(f"answer-pass@{k}: {overall_correct[k]}/{total} = {score:.2%}")

print("\n" + "=" * 70)
print("By OP bucket:")
for bucket in sorted(bucket_total.keys()):
    n = bucket_total[bucket]
    summary["by_bucket"][bucket] = {"num_samples": n}

    print(f"\n[{bucket}] (n={n})")
    for k in K_LIST:
        c = bucket_correct[bucket][k]
        score = c / n if n > 0 else 0.0
        summary["by_bucket"][bucket][f"answer-pass@{k}"] = score
        print(f"  answer-pass@{k}: {c}/{n} = {score:.2%}")

print("=" * 70)


# =========================
# 7. 保存结果
# =========================
summary_path = os.path.join(OUTPUT_DIR, "summary_by_op.json")
detail_path = os.path.join(OUTPUT_DIR, "detailed_results_by_op.json")

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "k_list": K_LIST,
        "k_max": K_MAX,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "eps": EPS,
        "num_test_samples": total,
        "bucket_counts": dict(bucket_counts),
        "metrics": summary,
    }, f, ensure_ascii=False, indent=2)

with open(detail_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n评测摘要已保存到: {summary_path}")
print(f"详细结果已保存到: {detail_path}")