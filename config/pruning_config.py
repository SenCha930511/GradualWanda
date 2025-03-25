from dataclasses import dataclass
from typing import Optional

# 建立參數物件
@dataclass
class PruningConfig:
    seed: int = 0
    nsamples: int = 2
    sparsity_ratio: float = 0.0
    sparsity_type: str = "unstructured"
    cache_dir: str = "llm_weights"
    use_variant: bool = False
    save: Optional[str] = None
    save_model: Optional[str] = None