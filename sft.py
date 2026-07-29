import torch
import os

# ====================== 缓存全放 autodl-tmp ======================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/.cache/transformers"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/.cache/huggingface/hub"

# ====================== 配置 ======================
MODEL_PATH = "Qwen/Qwen2-0.5B"
DATA_PATH = "/root/autodl-tmp/train.json"
OUTPUT_DIR = "/root/autodl-tmp/qwen2_0.5b_sft_op10"

MAX_SEQ_LEN = 1536
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 3e-5
EPOCHS = 2

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ---------------- 模型 ----------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------------- 数据格式化 ----------------
def format_example(example):
    prompt = (
        f"Problem: {example['problem']}\n"
        f"Question: {example['question']}\n"
        "Answer: "
    )
    response = example["solution"]
    text = prompt + response + tokenizer.eos_token

    tok = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

# ====================== ✅ 自动划分：训练集95% + 测试集5% ======================
ds = load_dataset("json", data_files=DATA_PATH)
ds = ds["train"].train_test_split(test_size=0.05, seed=42)

train_ds = ds["train"].map(format_example)
test_ds = ds["test"].map(format_example)

# ds = load_dataset("json", data_files=DATA_PATH)
# ds = ds["train"].select(range(100))  # 取前100条

# # 划分训练/测试
# ds = ds.train_test_split(test_size=0.1, seed=42)
# train_ds = ds["train"].map(format_example)
# test_ds = ds["test"].map(format_example)

# ====================== ✅ 训练参数（已修复报错） ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    bf16=True,
    gradient_checkpointing=True,

    save_strategy="epoch",
    save_total_limit=2,

    eval_strategy="epoch",  # ✅ 修复在这里！老版本用 eval_strategy ！

    logging_steps=10,
    report_to="swanlab",
    run_name="qwen2-0.5b-sft-op10",
)

# ====================== 训练器 ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 开始训练
trainer.train()

# 保存最终模型
model.save_pretrained(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

import swanlab
swanlab.finish()

print("🎉 训练完成！一切正常！")
