import wanda
import torch
from lib.pruning_config import PruningConfig
import time

def gradual_pruning(model_name, total_steps=5, final_sparsity=0.8, nsamples=2, cache_dir="llm_weights"):
    """
    使用逐步剪枝進行 Wanda 剪枝，逐步增加稀疏度，並在每次剪枝後進行微調。
    
    參數:
    - model_name: 模型名稱 (如 "meta-llama/Llama-2-7b-hf")
    - total_steps: 剪枝步驟數
    - final_sparsity: 最終的目標稀疏度
    - nsamples: 校準資料樣本數
    - cache_dir: 權重儲存目錄
    """
    
    # 計算每一步的稀疏度增加量
    sparsity_increment = final_sparsity / total_steps  
    current_sparsity = 0.0  # 初始沒有剪枝

    for step in range(total_steps):
        start_time = time.time()
        current_sparsity += sparsity_increment  # 逐步提高稀疏度
        
        print(f"Step {step+1}/{total_steps}: Pruning to {current_sparsity:.2%} sparsity...")

        # 設定 Wanda 剪枝參數
        prune_config = PruningConfig(
            model=model_name,
            seed=0,
            nsamples=nsamples,
            sparsity_ratio=current_sparsity,
            sparsity_type="unstructured",  # 使用非結構化剪枝
            cache_dir=cache_dir,
            use_variant=False,
            save=f"out/{model_name.replace('/', '_')}_ansonTesting/",
            save_model=f"out/{model_name.replace('/', '_')}_ansonTesting/"
        )

        # 執行剪枝
        wanda.prune_wanda(prune_config)

        print(f"Pruning completed for step {step+1}, starting fine-tuning...")

        # 進行微調（fine-tuning）
        
        
        #train_model(prune_config.save_model, epochs=2)
        
        

        print(f"Step {step+1} completed in {time.time() - start_time:.2f} seconds\n")

    print("Gradual pruning finished!")

# 模擬微調函數 (這裡只是簡單示範，實際需使用 PyTorch 等工具訓練)
def train_model(model_path, epochs=2):
    """
    進行微調，以恢復剪枝後的模型性能。
    
    參數:
    - model_path: 剪枝後模型的存放路徑
    - epochs: 訓練輪數
    """
    print(f"Fine-tuning model at {model_path} for {epochs} epochs...")
    time.sleep(2)  # 模擬訓練時間
    print("Fine-tuning completed.\n")

# 主程式入口
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    gradual_pruning(model_name, total_steps=5, final_sparsity=0.8)