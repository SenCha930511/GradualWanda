from dataclasses import dataclass
from typing import List
@dataclass
class EvaluateConfig:
    ntrain : int = 5
    data_dir : str = "data"
    save_dir : str = "result"
    engine : List[str] = ["llama2"]
    model_base : str = "/media/GradualWanda/merged_model"

    @staticmethod
    def default():
        return EvaluateConfig(
            save_dir = "result" + model_base
        )
