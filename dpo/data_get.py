import json
import random
from datasets import Dataset

# ===================== 配置 =====================
RAW_DATA_PATH = "/root/autodl-tmp/train.json"    # 你的2万条数据
SAMPLE_SIZE = 5000                 # 你想抽多少条 DPO 数据（5090 推荐 2000~10000）
OUTPUT_PATH = "/root/autodl-tmp/dpo_sampled_data.json"
SEED = 42                          # 固定抽样，可复现

# 加载 2w 条数据
with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 固定随机种子，抽样
random.seed(SEED)
sampled_data = random.sample(all_data, SAMPLE_SIZE)

# 保存抽样后的数据
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"✅ 抽样完成！原数据 {len(all_data)} 条 → 抽样后 {len(sampled_data)} 条")