from dataclasses import dataclass
from typing import Optional

@dataclass
class WandappConfig:
    seed: int = 42
    nsamples: int = 2
    sparsity_ratio: float = 0.5
    sparsity_type: str = "unstructured"
    model: str = "/media/GradualWanda/llm_weights/llama2-7b/models--meta-llama--Llama-2-7b-hf"
    use_variant: bool = True
    save: Optional[str] = "out/wandapp/llama2-7b"
    save_model: Optional[str] = "out/wandapp/llama2-7b"
