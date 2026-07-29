import os
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = "/root/autodl-tmp/checkpoint/dpo_full_missing_nodes_1"
DATA_PATH = "/root/autodl-tmp/data/half/validation.json"

sampling_params = SamplingParams(
    max_tokens=256,          # 先别开太大
    temperature=0.0,         # 做评测建议更稳一点
    top_p=1.0,
    stop_token_ids=[151645]
)

print("加载模型...")
llm = LLM(
    model=MODEL_PATH,
    tokenizer="Qwen/Qwen2-0.5B",
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    disable_log_stats=True,

    # 如果你机器之前报错，就先保留
    enforce_eager=True,
    disable_custom_all_reduce=True
)
print("模型加载成功！")

# 加载数据
ds = load_dataset("json", data_files=DATA_PATH)
ds = ds["train"].train_test_split(test_size=0.05, seed=42)
test_ds = ds["test"]

def extract_last_number(text):
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[-1]) if nums else -99999

# 1. 先构造所有 prompt
prompts = []
truths = []

for example in test_ds:
    prompt = f"Problem: {example['problem']}\nQuestion: {example['question']}\nAnswer: "
    prompts.append(prompt)
    truths.append(extract_last_number(example["solution"]))

print(f"开始批量生成，共 {len(prompts)} 条...")

# 2. 一次性批量推理 —— 这才更像 vLLM 的正确用法
outputs = llm.generate(prompts, sampling_params)

# 3. 统一统计
correct = 0
for i, out in enumerate(outputs):
    pred_text = out.outputs[0].text.strip()
    pred = extract_last_number(pred_text)
    true = truths[i]
    ok = abs(true - pred) < 1e-6
    if ok:
        correct += 1

print("\n" + "=" * 50)
print(f"最终正确率：{correct}/{len(prompts)} = {correct / len(prompts):.2%}")
print("=" * 50)