import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True  # 可選，根據你的需求設定
)


torch.cuda.empty_cache()
torch.cuda.memory_summary()

# 定義 GPU 記憶體監測函數
def check_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2  # 轉換成 MB
    reserved = torch.cuda.memory_reserved() / 1024**2  # 轉換成 MB
    print(f"已分配: {allocated:.2f} MB, 預留: {reserved:.2f} MB")
    return allocated

# 定義推理函數
def inference(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    torch.cuda.empty_cache()  # 清理記憶體
    check_gpu_memory()  # 記錄推理前記憶體使用量

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=50)
    end_time = time.time()
    
    torch.cuda.synchronize()  # 確保記憶體計算準確
    gpu_mem_used = check_gpu_memory()  # 記錄推理後記憶體使用量
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, gpu_mem_used, end_time - start_time

# 測試用的輸入文字
prompt = "The quick brown fox jumps over the lazy dog."

# **載入原始模型**

print("\n====== 測試原始模型 ======")
model_path_original = "/media/GradualWanda/out/meta-llama_Llama-2-7b-hf_pruned"
tokenizer = AutoTokenizer.from_pretrained(model_path_original)
model_original = AutoModelForCausalLM.from_pretrained(
    model_path_original, 
    torch_dtype=torch.float16, 
    quantization_config=quant_config,
    low_cpu_mem_usage=True, 
    ).eval()

generated_text_original, mem_used_original, time_original = inference(model_original, tokenizer, prompt)

print(f"原始模型: {mem_used_original:.2f} MB")
print(f"原始模型推理時間: {time_original:.4f} 秒")
print("原始模型輸出:", generated_text_original)

torch.cuda.empty_cache()
torch.cuda.memory_summary()
"""

# **載入剪枝後的模型**
print("\n====== 測試剪枝後的模型 ======")
model_path_pruned = "/media/GradualWanda/merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path_pruned)
model_pruned = AutoModelForCausalLM.from_pretrained(
    model_path_pruned, 
    torch_dtype=torch.float16, 
    quantization_config=quant_config,
    low_cpu_mem_usage=True, 
).eval()



generated_text_pruned, mem_used_pruned, time_pruned = inference(model_pruned, tokenizer, prompt)

print(f"剪枝模型: {mem_used_pruned:.2f} MB")
print(f"剪枝模型推理時間: {time_pruned:.4f} 秒")
print("剪枝模型輸出:", generated_text_pruned)
"""

"""
# **結果比較**
print("\n====== GPU 記憶體使用量對比 ======")
print(f"原始模型: {mem_used_original:.2f} MB")
print(f"剪枝模型: {mem_used_pruned:.2f} MB")
print(f"記憶體減少: {mem_used_original - mem_used_pruned:.2f} MB ({(1 - mem_used_pruned / mem_used_original) * 100:.2f}%)")

print("\n====== 推理時間對比 ======")
print(f"原始模型推理時間: {time_original:.4f} 秒")
print(f"剪枝模型推理時間: {time_pruned:.4f} 秒")
print(f"推理時間減少: {time_original - time_pruned:.4f} 秒 ({(1 - time_pruned / time_original) * 100:.2f}%)")

print("\n====== 生成結果範例 ======")
print("原始模型輸出:", generated_text_original)
print("剪枝模型輸出:", generated_text_pruned)
"""
