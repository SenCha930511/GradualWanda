from dataclasses import dataclass
from typing import Optional
from pruning_config import PruningConfig
from lora_config import LoRaConfig

@dataclass
class GradualConfig:
    model_name: str
    total_steps: int
    final_sparsity: float
    nsamples: int
    cache_dir: str
    pruning_config: PruningConfig
    lora_config: LoRaConfig

    @staticmethod
    def default():
        """提供預設的 `GradualConfig` 初始化方式"""
        return GradualConfig(
            total_steps=5
            final_sparsity=0.8
            nsamples=2
            cache_dir="llm_weights"
            pruning_config=PruningConfig(model=model_name),
            lora_config=LoRaConfig(model=model_name)
        )