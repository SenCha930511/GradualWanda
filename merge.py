import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 設定裝置，優先使用 GPU，但允許 CPU 參與計算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 釋放 GPU 記憶體
torch.cuda.empty_cache()

# 8-bit 量化配置，降低 VRAM 需求
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True  # 啟用 8-bit 量化，減少 VRAM 消耗
)

# 加載基礎模型（使用 bfloat16 或 8-bit）
model_base = "/media/GradualWanda/out/meta-llama_Llama-2-7b-hf_pruned"
model = AutoModelForCausalLM.from_pretrained(
    model_base,
    torch_dtype=torch.bfloat16,  # bfloat16 通常比 fp16 更穩定
    quantization_config=bnb_config,  # 使用 8-bit 量化
    device_map="auto"  # 讓 Hugging Face 自動管理記憶體分配
)

# 強制釋放記憶體
torch.cuda.empty_cache()

# 加載 LoRA 適配器
lora_model_path = "/media/GradualWanda/lora_finetuned_model/checkpoint-89078"
lora_model = PeftModel.from_pretrained(
    model,
    lora_model_path,
    torch_dtype=torch.bfloat16,  # 同樣使用 bfloat16
    device_map="auto"
)

# 釋放記憶體
torch.cuda.empty_cache()

# **逐層合併 LoRA**（避免 OOM）
merged_model = lora_model.merge_and_unload()

# 儲存合併後的模型
save_path = "/media/GradualWanda/merged_model"
merged_model.save_pretrained(save_path)

print(f"✅ 模型已儲存至 {save_path}")
