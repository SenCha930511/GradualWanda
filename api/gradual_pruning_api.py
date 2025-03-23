import os
import time
import wanda_api as wanda
import lora_api as lora
import torch
import gc
from config.pruning_config import PruningConfig
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

def gradual_pruning(model_name, total_steps=5, final_sparsity=0.8, nsamples=2, cache_dir="llm_weights"):
    sparsity_increment = final_sparsity / total_steps  
    current_sparsity = 0.0

    for step in range(total_steps):
        start_time = time.time()
        current_sparsity += sparsity_increment  
        print(f"Step {step+1}/{total_steps}: Pruning to {current_sparsity:.2%} sparsity...")

        prune_config = PruningConfig(
            model=model_name,
            seed=0,
            nsamples=nsamples,
            sparsity_ratio=current_sparsity,
            sparsity_type="unstructured",
            cache_dir=cache_dir,
            use_variant=False,
            save=f"out/{model_name.replace('/', '_')}_pruned/",
            save_model=f"out/{model_name.replace('/', '_')}_pruned/"
        )

        torch.cuda.empty_cache()
        wanda.prune_wanda(prune_config)

        print(f"Pruning completed for step {step+1}, starting LoRA fine-tuning...") 
        lora.lora_finetune(prune_config.save_model, epochs=2)
        print(f"Step {step+1} completed in {time.time() - start_time:.2f} seconds\n")

    print("Gradual pruning finished!")


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    gradual_pruning(model_name, total_steps=5, final_sparsity=0.8)
