from dataclasses import dataclass
from typing import Optional
from .pruning_config import PruningConfig
from .lora_config import LoRaConfig

@dataclass
class GradualConfig:
    pruning_config: PruningConfig
    lora_config: LoRaConfig
    model_name: str  = "meta-llama/Llama-2-7b-hf"
    total_steps: int = 5
    final_sparsity: float = 0.8
    nsamples: int = 2
    cache_dir: str = "llm_weights"
