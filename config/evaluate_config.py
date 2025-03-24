from dataclasses import dataclass
from typing import List
@dataclass
class EvaluateConfig:
    ntrain : int
    data_dir : str
    save_dir : str
    engine : List[str]
    model_base : str
