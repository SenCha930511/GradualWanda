from dataclasses import dataclass
from typing import List

@dataclass
class EvaluateConfig:
    ntrain: int = 5          # 用於提示詞的訓練範例數量
    lora_path: str = None    # LoRA 權重路徑
    data_dir: str = "data"    # 包含開發/測試 CSV 檔案的目錄
    save_dir: str = "output"  # 儲存評估結果的目錄（你可以自行調整）
    engine: List[str] = ("merged",)  # 引擎標籤
    
