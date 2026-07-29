import os
import shutil
from transformers import Qwen2ForCausalLM, AutoTokenizer
import torch

# ================= 配置区域 =================
# 你的原始模型路径 (带 v_head 的那个)
original_model_path = "/root/autodl-tmp/checkpoint/ppo_op10_output/final_model"

# 新的模型保存路径 (不带 v_head，给 vLLM 用的)
# 这个文件夹不需要提前创建，脚本会自动生成
new_model_path = "/root/autodl-tmp/checkpoint/ppo_op10_output/final_model_for_inference"
# ===========================================

def main():
    print("=" * 60)
    print("开始处理模型：移除 v_head 权重")
    print("=" * 60)

    # 1. 检查原始文件是否存在
    print(f"\n[1/5] 检查原始模型路径...")
    if not os.path.isdir(original_model_path):
        raise NotADirectoryError(f"❌ 错误：找不到原始模型文件夹！\n路径: {original_model_path}")

    config_path = os.path.join(original_model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 错误：原始文件夹下找不到 config.json！\n请确认这是一个完整的 HuggingFace 模型目录。")

    print(f"✅ 原始模型路径确认无误。")

    # 2. 检查新路径是否已存在
    print(f"\n[2/5] 检查目标路径...")
    if os.path.exists(new_model_path):
        print(f"⚠️  警告：目标路径已存在！")
        response = input(f"是否删除旧的 '{new_model_path}' 并继续？(输入 yes 确认): ")
        if response.lower() != 'yes':
            print("操作已取消。")
            return
        shutil.rmtree(new_model_path)
        print(f"✅ 已清理旧路径。")

    print(f"✅ 目标路径准备就绪。")

    # 3. 加载模型
    print(f"\n[3/5] 正在加载原始模型 (这可能需要一点时间)...")
    print(f"   (正在加载到 CPU 以节省显存...)")

    # 强制加载到 CPU，避免显存溢出，处理完再存走
    model = Qwen2ForCausalLM.from_pretrained(
        original_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # 关键：先放 CPU
        local_files_only=True # 关键：不联网，避免 Hub 报错
    )

    tokenizer = AutoTokenizer.from_pretrained(
        original_model_path,
        local_files_only=True
    )
    print(f"✅ 模型和 Tokenizer 加载成功。")

    # 4. 移除 v_head
    print(f"\n[4/5] 正在移除 v_head 权重...")
    if hasattr(model, "v_head"):
        del model.v_head
        print(f"✅ 成功删除 'v_head' 模块。")
    else:
        print(f"⚠️  警告：当前模型中未找到 'v_head'，可能已经被移除过了。")

    # 5. 保存新模型
    print(f"\n[5/5] 正在保存处理后的模型...")
    print(f"   保存路径: {new_model_path}")

    os.makedirs(new_model_path, exist_ok=True)

    # 保存模型权重
    model.save_pretrained(new_model_path)
    # 保存 Tokenizer
    tokenizer.save_pretrained(new_model_path)

    print("\n" + "=" * 60)
    print("🎉 处理完成！")
    print("=" * 60)
    print(f"原始模型保留在: {original_model_path}")
    print(f"新模型 (无 v_head) 已保存至: {new_model_path}")
    print("\n下一步操作：")
    print(f"请修改你的 'eval_answer_passk.py' 或 'ppo_use.py' 中的模型路径为：")
    print(f"👉 {new_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
