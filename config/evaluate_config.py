from dataclasses import dataclass
from typing import List

@dataclass
class EvaluateConfig:
    ntrain : int = 5    #看你自己要多少的預先prompt
    data_dir : str = "data"
    save_dir : str = "output"   #直接用output，雖然本來是result
    engine : List[str] = ["llama2"]
    model_base : str = "/media/GradualWanda/merged_model"   #要用別的自己改路徑
