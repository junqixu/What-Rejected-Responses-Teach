from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 路径（你的不变）
BASE_MODEL = "/root/autodl-tmp/qwen2_0.5b_sft_op10/checkpoint-10339"
LORA_MODEL = "/root/autodl-tmp/dpo_outputs"
MERGED_SAVE_PATH = "/root/autodl-tmp/qwen2_0.5b_dpo_final"

# 加载基础模型
print("加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

# 加载 LoRA
print("加载 LoRA 权重...")
model = PeftModel.from_pretrained(model, LORA_MODEL)

# 合并
print("开始合并...")
model = model.merge_and_unload()

# 保存完整模型
print("保存完整模型...")
model.save_pretrained(MERGED_SAVE_PATH, safe_serialization=True)

# ==============================================
# ✅ 关键修复：不从本地加载坏 tokenizer，直接用官方的
# ==============================================
print("加载官方 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",  # 直接用官方，完美解决
    trust_remote_code=True
)
tokenizer.save_pretrained(MERGED_SAVE_PATH)

print("\n✅ 合并完成！模型路径：")
print(MERGED_SAVE_PATH)