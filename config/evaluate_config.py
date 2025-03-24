from dataclasses import dataclass
from typing import List
@dataclass
class EvaluateConfig:
    ntrain : int = 5
    data_dir : str = "data"
    save_dir : str
    engine : List[str]
    model_base : str
