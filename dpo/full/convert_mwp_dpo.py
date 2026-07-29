import json
import re

# ===================== 配置 =====================
INPUT_PATH = "/root/dpo_outputs_missing_only_1/raw_generations.jsonl"       # 你的原始数据文件
OUTPUT_PATH = "/root/dpo_outputs_missing_only_1/dpo_train.jsonl"   # 输出 DPO 数据
TEMPLATE = "chatml"              # Qwen / 通义千问 用这个
# ==================================================

def build_prompt(problem, question, template="chatml"):
    prompt = f"Problem:\n{problem}\n\nQuestion:\n{question}"
    if template == "chatml":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif template == "llama3":
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif template == "llama2":
        return f"[INST] {prompt} [/INST] "
    else:
        return f"Problem:\n{problem}\nQuestion:\n{question}\nSolution:\n"

def clean_text(s):
    # 清理多余换行、空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def process_one(obj):
    item = obj["item"]
    problem = item["problem"]
    question = item["question"]
    chosen = item["solution"]
    rejected = obj["generation"]["missing_nodes"]["rejected"]

    prompt = build_prompt(problem, question, template=TEMPLATE)

    return {
        "id": obj["id"],
        "prompt": prompt,
        "chosen": clean_text(chosen),
        "rejected": clean_text(rejected)
    }

def main():
    data = []
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(process_one(obj))
            except Exception as e:
                print(f"跳过一行解析失败: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"处理完成！共 {len(data)} 条 DPO 数据，已保存到 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()