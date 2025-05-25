from dataclasses import dataclass
from typing import List, Literal

@dataclass
class EvaluateConfig:
    ntrain: int = 5          # 用於提示詞的訓練範例數量
    lora_path: str = None    # LoRA 權重路徑
    
    data_dir: str = "data"    # 包含開發/測試 CSV 檔案的目錄
    save_dir: str = "output"  # 儲存評估結果的目錄
    engine: List[str] = ("engine",)  # 引擎標籤
    
    # 評估模式選項
    eval_mode: Literal["core", "full"] = "full"  # 評估模式：core=核心科目，full=完整科目
    custom_subjects: List[str] = None  # 自定義要評估的科目列表，如果設置則優先使用此列表
