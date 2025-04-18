from dataclasses import dataclass
from typing import List

@dataclass
class EvaluateConfig:
    ntrain: int = 5          # 用於提示詞的訓練範例數量
    data_dir: str = "data"    # 包含開發/測試 CSV 檔案的目錄
    save_dir: str = "output"  # 儲存評估結果的目錄（你可以自行調整）
    model_base: str = "/media/GradualWanda/merged_model"  # 模型基本路徑
    engine: List[str] = ("llama2",)  # 引擎標籤
    
