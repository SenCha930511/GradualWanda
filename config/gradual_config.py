from dataclasses import dataclass
from typing import Optional
from .pruning_config import PruningConfig
from .lora_config import LoRaConfig

@dataclass
class GradualConfig:
    pruning_config: PruningConfig
    lora_config: LoRaConfig
    s_final: float = 0.8    # 目標最終稀疏度
    model_name: str = "meta-llama/Llama-2-7b-hf"
    total_steps: int = 5
    #final_sparsity: float = 0.8
    nsamples: int = 2
    cache_dir: str = "llm_weights"
    t0: int = 0       # 剪枝從第 t0 步開始
    #delta_t: int  = 1     # 每隔多少步進行一次剪枝(預設為每步都剪)
