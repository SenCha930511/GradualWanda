# modules/gradual_pruning.py

import os
import time
import torch
import gc
import sys
from . import wanda
from . import lora

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.pruning_config import PruningConfig
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from lib.model_utils import get_llm
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from config import GradualConfig

def compute_sparsity(t, t0, n, delta_t, s_init, s_final):
    """
    t: 當前訓練步

    t0: 剪枝開始步

    n: 剪枝總步數

    delta_t: 兩次剪枝之間的步數間隔

    s_init: 初始稀疏度（通常 0)

    s_final: 目標最終稀疏度（例如 0.8)
    
    """
    """
    計算訓練步 t 時的目標稀疏度 current_sparsity:_t
    current_sparsity_t = s_f + (s_i - s_f) * (1 - (t - t0)/(n*delta_t))^3
    且在 t < t0 時返回 s_init, t >= t0 + n*delta_t 時返回 s_final。
    """
    if t < t0:
        return s_init
    elif t > t0 + n * delta_t:
        return s_final
    else:
        progress = 1 - (t - t0) / (n * delta_t)
        return s_final + (s_init - s_final) * (progress ** 3)

def gradual_pruning(config: GradualConfig, model_path: str):
    
    s_init = 0.0
    t0 = config.t0
    n = config.total_steps
    s_final = config.s_final

    
    """
    #舊稀疏度計算
    sparsity_increment = config.final_sparsity / config.total_steps  
    current_sparsity = 0.0
    """

    """
    #單獨測試lora
    lora_config = config.lora_config
    prune_config = config.pruning_config
    lora.lora_finetune(lora_config, prune_config.save_model)
    print(f" completed in {time.time()}\n")
    
    """
    current_sparsity = s_init
    for step in range(n):
        start_time = time.time()
        
        current_sparsity_t = compute_sparsity(step , 0, n, 1 , current_sparsity, s_final)
        print(f"[Step {step+1}/{n}] , target sparsity={current_sparsity_t:.4%}")
        
        #current_sparsity = compute_sparsity(t, t0, config.total_steps, delta_t, s_init, config.final_sparsity)
        
        #current_sparsity += sparsity_increment  
        #print(f"Step {step+1}/{config.total_steps}: Pruning to {current_sparsity:.2%} sparsity...")

        lora_config = config.lora_config
        
        prune_config = PruningConfig(
            model = model_path,
            seed=0,
            nsamples=config.nsamples,
            sparsity_ratio=current_sparsity_t,
            sparsity_type="unstructured",
            #cache_dir=config.cache_dir,
            use_variant=False,
            save=f"out/{model_path.replace('/', '_')}_pruned/",
            save_model=f"out/{model_path.replace('/', '_')}_pruned/"
        )
        
        torch.cuda.empty_cache()
        
        #執行wanda
        #wanda.prune_wanda(prune_config, model_path)
        current_sparsity = wanda.prune_wanda(prune_config, model_path)
        print(f"目前模型剪枝後的實際 sparsity: {current_sparsity:.4f}")

        print(f"Pruning completed for step {step+1}, starting LoRA fine-tuning...") 
        #lora.lora_finetune(lora_config, prune_config.save_model)
        print(f"Step {step+1} completed in {time.time() - start_time:.2f} seconds\n")

    print("Gradual pruning finished!")
    
    print(config)

"""
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    gradual_pruning(model_name, total_steps=5, final_sparsity=0.8)
"""
