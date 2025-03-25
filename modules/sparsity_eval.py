# modules/sparsity_eval.py
import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig

def evaluate_model_sparsity(model_path: str):
    """
    針對 8-bit quantized (bitsandbytes) LLaMA 模型，計算大約的稀疏度。
    回傳：
      1. total_params  : 模型參數總數
      2. nonzero_params: 非零參數數量
      3. sparsity      : 稀疏度 = 1 - (nonzero_params / total_params)
    """
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    total_params = 0
    nonzero_params = 0

    for name, param in model.named_parameters():
        param_data = param.data.float()
        param_count = param_data.numel()
        total_params += param_count

        nonzero_count = torch.count_nonzero(param_data).item()
        nonzero_params += nonzero_count

    sparsity = 1.0 - (nonzero_params / total_params)
    return total_params, nonzero_params, sparsity

if __name__ == "__main__":
    model_path = "/media/GradualWanda/merged_model"
    total_params, nonzero_params, sparsity = evaluate_model_sparsity(model_path)
    
    # 將稀疏度轉成百分比顯示
    sparsity_percent = sparsity * 100
    print(f"模型參數總數  : {total_params}")
    print(f"模型非零參數 : {nonzero_params}")
    print(f"模型稀疏度   : {sparsity:.6f} (比例)")
    print(f"模型稀疏度   : {sparsity_percent:.2f}% (百分比)")
