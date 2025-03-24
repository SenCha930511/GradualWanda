from dataclasses import dataclass
from typing import List
@dataclass
class EvaluateConfig:
    ntrain : int = 5
    data_dir : str = "data"
    save_dir : str
    engine : List[str] = ["llama2"]
    model_base : str
