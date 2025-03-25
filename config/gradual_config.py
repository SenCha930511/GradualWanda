from dataclasses import dataclass
from typing import Optional
from config import PruningConfig
from config import EvaluateConfig

@dataclass
class GradualConfig:
    model_name: str 
    total_steps: int = 5
    final_sparsity: float = 0.8
    nsamples: int = 2
    cache_dir: str = "llm_weights"
    pruning_config: PruningConfig
    lora_config: EvaluateConfig
