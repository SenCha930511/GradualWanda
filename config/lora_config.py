from dataclasses import dataclass, field
from typing import Optional, List
from peft import TaskType

@dataclass
class LoRaConfig:
    """
    與 PruningConfig 相同風格的 LoRA 參數配置示例。
    可依需求自行增減屬性。
    """

    # (1) LoRA 相關參數
    r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM  # 如果是 Encoder/Decoder 模型可改為 SEQ_2_SEQ_LM

    # (2) 一些額外控制參數（可選，視你實際需求增減）
    epochs: int = 2
    per_device_train_batch_size: int = 4
    max_length: int = 256

    # (3) 儲存路徑等參數
    output_dir: Optional[str] = "lora_finetuned_model"
