import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def merge_lora(base_model_path: str, lora_model_path: str, save_path: str):
    # 設定裝置，優先使用 GPU，但允許 CPU 參與計算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 釋放 GPU 記憶體
    torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # 加載基礎模型（使用 bfloat16 或 8-bit）
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config, 
        device_map="auto"  # 讓 Hugging Face 自動管理記憶體分配
    )

    # 強制釋放記憶體
    torch.cuda.empty_cache()

    # 加載 LoRA 適配器
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
    merged_model.save_pretrained(save_path)

    print(f"✅ 模型已儲存至 {save_path}")
