import os
import re
from collections import defaultdict
from datasets import load_dataset
from vllm import LLM, SamplingParams

# ===================== 配置区 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = "/root/autodl-tmp/dpo_full_final_compute_error_1"
TOKENIZER_PATH = "Qwen/Qwen2-0.5B"
DATA_PATH = "/root/autodl-tmp/validation.json"   # 这里换成你的测试集路径

USE_EAGER = True    # 如果环境稳定，可以试试改成 False 更快
BATCH_SIZE = 1000
MAX_TOKENS = 256

# 你的数据里 op 字段名如果不是 "op_num"，改这里
OP_FIELD_CANDIDATES = ["op", "op_num", "operations", "num_ops"]

# 问题字段候选
PROBLEM_FIELD_CANDIDATES = ["problem"]
QUESTION_FIELD_CANDIDATES = ["question"]
ANSWER_FIELD_CANDIDATES = ["solution", "answer", "target"]

# =================================================


def find_first_existing_key(example, candidates):
    for k in candidates:
        if k in example:
            return k
    return None


def extract_last_number(text):
    nums = re.findall(r'-?\d+\.?\d*', str(text))
    return float(nums[-1]) if nums else None


def build_prompt(example):
    problem_key = find_first_existing_key(example, PROBLEM_FIELD_CANDIDATES)
    question_key = find_first_existing_key(example, QUESTION_FIELD_CANDIDATES)

    problem = example.get(problem_key, "") if problem_key else ""
    question = example.get(question_key, "") if question_key else ""

    if problem and question:
        return f"Problem: {problem}\nQuestion: {question}\nAnswer: "
    elif question:
        return f"{question}\nAnswer: "
    elif problem:
        return f"{problem}\nAnswer: "
    else:
        return "Answer: "


def get_op_value(example):
    for k in OP_FIELD_CANDIDATES:
        if k in example:
            return int(example[k])
    return None


print("加载模型...")
llm = LLM(
    model=MODEL_PATH,
    tokenizer=TOKENIZER_PATH,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    disable_log_stats=True,
    enforce_eager=USE_EAGER,
    disable_custom_all_reduce=True,
)
print("模型加载成功！")

sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.0,
    top_p=1.0,
    stop_token_ids=[151645]
)

print("加载测试集...")
ds = load_dataset("json", data_files=DATA_PATH)["train"]
print(f"测试样本数: {len(ds)}")

# 自动识别字段
sample = ds[0]
op_key = find_first_existing_key(sample, OP_FIELD_CANDIDATES)
ans_key = find_first_existing_key(sample, ANSWER_FIELD_CANDIDATES)

if ans_key is None:
    raise ValueError(f"没有找到答案字段，请检查 {ANSWER_FIELD_CANDIDATES}")

print(f"检测到答案字段: {ans_key}")
print(f"检测到 op 字段: {op_key if op_key else '未找到，将无法按 op 统计'}")

# 按 op 分桶
op_buckets = defaultdict(list)
for ex in ds:
    op_val = get_op_value(ex)
    if op_val is None:
        continue
    op_buckets[op_val].append(ex)

if not op_buckets:
    raise ValueError("没有成功读到任何 op 字段，无法按 op 测试。")

print("\n========== OP 数量统计 ==========")
for op in sorted(op_buckets.keys()):
    print(f"op{op:<2}: {len(op_buckets[op])} 条")

results = {}

for op in sorted(op_buckets.keys()):
    subset = op_buckets[op]
    prompts = [build_prompt(ex) for ex in subset]
    truths = [extract_last_number(ex[ans_key]) for ex in subset]

    correct = 0
    total = len(subset)

    print(f"\n开始测试 op{op}，共 {total} 条...")

    for start in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[start:start + BATCH_SIZE]
        batch_truths = truths[start:start + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        for out, true in zip(outputs, batch_truths):
            pred_text = out.outputs[0].text.strip()
            pred = extract_last_number(pred_text)

            ok = (pred is not None and true is not None and abs(pred - true) < 1e-6)
            if ok:
                correct += 1

    acc = correct / total if total > 0 else 0.0
    results[op] = {
        "correct": correct,
        "total": total,
        "acc": acc
    }

print("\n" + "=" * 60)
print("按 op 统计准确率")
print("=" * 60)
print(f"{'OP':<6}{'Correct':<12}{'Total':<10}{'Acc':<10}")
for op in sorted(results.keys()):
    r = results[op]
    print(f"{op:<6}{r['correct']:<12}{r['total']:<10}{r['acc']:.2%}")

# 区间汇总
def summarize_range(name, op_list):
    total = sum(results[op]["total"] for op in op_list if op in results)
    correct = sum(results[op]["correct"] for op in op_list if op in results)
    acc = correct / total if total > 0 else 0.0
    print(f"{name:<12} {correct:<8}/{total:<8} = {acc:.2%}")

print("\n" + "=" * 60)
print("区间汇总")
print("=" * 60)
summarize_range("op2-9", list(range(2, 10)))
summarize_range("op10", [10])
summarize_range("op11-20", list(range(11, 21)))