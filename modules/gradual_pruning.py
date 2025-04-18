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

def gradual_pruning(config: GradualConfig, model_path: str):
    
    sparsity_increment = config.final_sparsity / config.total_steps  
    current_sparsity = 0.0

    """
    #單獨測試lora
    lora_config = config.lora_config
    prune_config = config.pruning_config
    lora.lora_finetune(lora_config, prune_config.save_model)
    print(f" completed in {time.time()}\n")
    
    """
    for step in range(config.total_steps):
        start_time = time.time()
        current_sparsity += sparsity_increment  
        print(f"Step {step+1}/{config.total_steps}: Pruning to {current_sparsity:.2%} sparsity...")

        lora_config = config.lora_config
        
        prune_config = PruningConfig(
            model = get_llm(model_path, config.cache_dir),
            seed=0,
            nsamples=config.nsamples,
            sparsity_ratio=current_sparsity,
            sparsity_type="unstructured",
            cache_dir=config.cache_dir,
            use_variant=False,
            save=f"out/{model_name.replace('/', '_')}_pruned/",
            save_model=f"out/{model_name.replace('/', '_')}_pruned/"
        )
        



        torch.cuda.empty_cache()
        wanda.prune_wanda(prune_config)

        print(f"Pruning completed for step {step+1}, starting LoRA fine-tuning...") 
        lora.lora_finetune(lora_config, prune_config.save_model)
        print(f"Step {step+1} completed in {time.time() - start_time:.2f} seconds\n")

    print("Gradual pruning finished!")
    
    print(config)

"""
if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    gradual_pruning(model_name, total_steps=5, final_sparsity=0.8)
"""
