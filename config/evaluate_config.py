from dataclasses import dataclass
from typing import List

@dataclass
class EvaluateConfig:
<<<<<<< HEAD
<<<<<<< Updated upstream
    ntrain : int = 5    #看你自己要多少的預先prompt
    data_dir : str = "data"
    save_dir : str = "output"   #直接用output，雖然本來是result
    engine : List[str] = ["llama2"]
<<<<<<< HEAD
=======
    ntrain: int = 5          # 用於提示詞的訓練範例數量
    data_dir: str = "data"    # 包含開發/測試 CSV 檔案的目錄
    save_dir: str = "output"  # 儲存評估結果的目錄（你可以自行調整）
    engine: List[str] = ("merged",)  # 引擎標籤
>>>>>>> Stashed changes
=======
>>>>>>> 0d5d81d (Update data structure)
=======
    ntrain: int = 5          # 用於提示詞的訓練範例數量
    data_dir: str = "data"    # 包含開發/測試 CSV 檔案的目錄
    save_dir: str = "output"  # 儲存評估結果的目錄（你可以自行調整）
    engine: List[str] = ("merged",)  # 引擎標籤
    
>>>>>>> 240450c (Fixs some bugs)
