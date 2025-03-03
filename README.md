# GradualWanda

GradualWanda 是一個以 Python 撰寫的專案，將結合逐步減枝及 wanda 算法，對 LLaMA-2-7b 進行剪枝輕量化。目前專案包含了基本的函式庫和一個主要程式 (`wanda.py`)。

## 安裝

要設置此專案，請先 clone 並安裝所需的依賴套件：

```bash
git clone https://github.com/SenCha930511/GradualWanda.git
cd GradualWanda
pip install -r requirements.txt
```

## 資料結構

PruningConfig 類別是用來設定剪枝過程中的各種參數。以下是其內容：

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class PruningConfig:
    model: str
    seed: int = 0
    nsamples: int = 2
    sparsity_ratio: float = 0.0
    sparsity_type: str = "unstructured"
    cache_dir: str = "llm_weights"
    use_variant: bool = False
    save: Optional[str] = None
    save_model: Optional[str] = None
```



## 使用方法

以 LLaMA-2-7b 為例，進行 wanda 剪枝：

```python
import wanda
from lib.pruning_config import PruningConfig

prune_config = PruningConfig(model="meta-llama/Llama-2-7b-hf", seed=0, nsamples=2, sparsity_ratio=0.5, sparsity_type="unstructured", cache_dir="llm_weights", use_variant=False, save="out/llama2_7b/", save_model="out/llama2_7b/")

wanda.prune_wanda(prune_config)
```

