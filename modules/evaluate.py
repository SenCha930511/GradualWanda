import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import EvaluateConfig  # 移除 CORE_SUBJECTS 的導入
import time
import threading
import gc
import numpy as np
import pandas as pd
import torch
import pynvml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from typing import List


CORE_SUBJECTS = [
    # STEM 領域
    "computer_science",           # 電腦科學（基礎學科）
    "college_computer_science",   # 大學電腦科學（進階）
    "machine_learning",          # 機器學習（熱門領域）
    "mathematics",               # 數學（基礎學科）
    "college_mathematics",       # 大學數學（進階）
    "physics",                   # 物理（基礎學科）
    "college_physics",          # 大學物理（進階）
    "chemistry",                 # 化學（基礎學科）
    "college_chemistry",        # 大學化學（進階）
    "biology",                   # 生物（基礎學科）
    "college_biology",          # 大學生物（進階）
    
    # 醫學相關
    "anatomy",                   # 解剖學（基礎醫學）
    "clinical_knowledge",        # 臨床知識（實用醫學）
    "professional_medicine",     # 專業醫學（進階醫學）
    
    # 社會科學
    "economics",                 # 經濟學（基礎社會科學）
    "psychology",                # 心理學（基礎社會科學）
    "professional_psychology",   # 專業心理學（進階）
    "sociology",                 # 社會學（基礎社會科學）
    
    # 人文學科
    "philosophy",                # 哲學（基礎人文）
    "formal_logic",             # 形式邏輯（基礎思維）
    "jurisprudence",            # 法理學（法律基礎）
    
    # 專業領域
    "business_ethics",          # 商業倫理（實用專業）
    "computer_security",        # 電腦安全（熱門專業）
    "electrical_engineering"    # 電機工程（工程專業）
]

# 學科類別
SUBJECT_CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "biology", "chemistry", "computer_science", "mathematics", "medicine", "physics", "engineering", "statistics"],
    "Humanities": ["philosophy", "history", "world_religions", "law", "ethics"],
    "Social_Sciences": ["psychology", "sociology", "economics", "geography", "politics"],
    "Other": ["business", "health", "miscellaneous"]
}

# Softmax 函數
def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

# GPU 使用量監測線程 - 修改版本
class GPUMonitor:
    def __init__(self, interval=1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.util_list = []
        self.subject_utils = {}
        self.stop_event = threading.Event()
        self.interval = interval
        self.current_subject = None
        self.thread = threading.Thread(target=self._monitor)
        self.thread.daemon = True

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        pynvml.nvmlShutdown()

    def _monitor(self):
        while not self.stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.util_list.append(util.gpu)
            if self.current_subject:
                if self.current_subject not in self.subject_utils:
                    self.subject_utils[self.current_subject] = []
                self.subject_utils[self.current_subject].append(util.gpu)
            time.sleep(self.interval)

    def set_subject(self, subject):
        self.current_subject = subject

    def get_subject_util(self, subject):
        if subject in self.subject_utils and self.subject_utils[subject]:
            return sum(self.subject_utils[subject]) / len(self.subject_utils[subject])
        return 0.0

    def get_average_util(self):
        if self.util_list:
            return sum(self.util_list) / len(self.util_list)
        return 0.0

# 計算每個類別的平均準確率並回傳字典
def calculate_category_accuracies(subject_accuracies):
    category_acc = {cat: [] for cat in SUBJECT_CATEGORIES}
    for subject, acc in subject_accuracies.items():
        for category, sub_list in SUBJECT_CATEGORIES.items():
            if any(sub.lower() in subject.lower() for sub in sub_list):
                category_acc[category].append(acc)
    category_summary = {}
    for category, acc_list in category_acc.items():
        avg_acc = np.mean(acc_list) if acc_list else 0.0
        print(f"{category} 的平均準確率: {avg_acc:.3f}")
        category_summary[category] = avg_acc
    return category_summary

# 取得模型路徑
def get_model_path(model_dir):
    snapshot_dir = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshot_dir):
        snapshots = [os.path.join(snapshot_dir, d) for d in os.listdir(snapshot_dir)
                     if os.path.isdir(os.path.join(snapshot_dir, d))]
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            print(f"使用最新快照: {latest_snapshot}")
            return latest_snapshot
    print(f"未找到快照，使用原始路徑: {model_dir}")
    return model_dir

# 載入本地 Llama-2 模型與分詞器 (GPU版本)
def load_llama_model(model_base, lora_path=None):
    """
    載入本地 Llama-2 模型與分詞器，可選擇性載入 LoRA 權重
    
    Args:
        model_base (str): 基礎模型路徑
        lora_path (str, optional): LoRA 權重路徑
    
    Returns:
        tuple: (model, tokenizer)
    """
    model_path = get_model_path(model_base)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路徑 {model_path} 不存在，請檢查是否下載正確。")
    print(f"使用的模型路徑: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的設備: {device}")
    
    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"設定 pad_token 為 eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "left"
    
    # 設置量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 載入基礎模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # 如果提供了 LoRA 路徑，則載入 LoRA 權重
    if lora_path and os.path.exists(lora_path):
        print(f"正在載入 LoRA 權重: {lora_path}")
        try:
            # 載入 LoRA 配置
            peft_config = PeftConfig.from_pretrained(lora_path)
            print(f"LoRA 配置: {peft_config}")
            
            # 載入 LoRA 權重
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("LoRA 權重載入完成")
            
            # 合併 LoRA 權重到基礎模型
            model = model.merge_and_unload()
            print("LoRA 權重已合併到基礎模型")
            
        except Exception as e:
            print(f"載入 LoRA 權重時發生錯誤: {str(e)}")
            print("將使用基礎模型繼續執行")
    
    model.eval()
    print(f"模型 {model_path} 載入完成。")
    return model, tokenizer

# 若 prompt 太長則裁剪
def crop_prompt(prompt, tokenizer, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs.input_ids.shape[1] > max_length:
        input_ids = inputs.input_ids[:, -max_length:]
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return prompt

# 計算 log 機率
def local_generate_logprobs(model, tokenizer, prompt, answers):
    max_seq_len = model.config.max_position_embeddings
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)
    attention_mask = inputs.attention_mask
    seq_len = inputs.input_ids.shape[-1]
    non_pad_len = int(attention_mask.sum().item())
    position_ids = torch.cat([
        torch.zeros(seq_len - non_pad_len, device=inputs.input_ids.device, dtype=torch.long),
        torch.arange(non_pad_len, device=inputs.input_ids.device, dtype=torch.long)
    ]).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
    logits = outputs.logits[0, -1, :]
    answer_ids = []
    for ans in answers:
        tokenized = tokenizer(" " + ans, add_special_tokens=False)["input_ids"]
        token_id = tokenized[1] if len(tokenized) > 1 else tokenized[0]
        answer_ids.append(token_id)
    return [float(logits[t].item()) for t in answer_ids]

# 推理函數
def eval_local(config, subject, model, tokenizer, dev_df, test_df, gpu_monitor=None):
    if gpu_monitor:
        gpu_monitor.set_subject(subject)

    cors, all_probs, answers = [], [], ["A","B","C","D"]
    max_seq_len = model.config.max_position_embeddings
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(test_df.shape[0]):
        k = config.ntrain
        prompt = gen_prompt(dev_df, subject, k) + format_example(test_df, i, include_answer=False)
        prompt = crop_prompt(prompt, tokenizer, max_seq_len)
        label = test_df.iloc[i, -1]
        lprobs = local_generate_logprobs(model, tokenizer, prompt, answers)
        pred = answers[np.argmax(lprobs)]; probs = softmax(np.array(lprobs))
        cors.append(pred == label); all_probs.append(probs)
    end_event.record(); torch.cuda.synchronize()
    t = start_event.elapsed_time(end_event) / 1000
    acc = np.mean(cors)
    print(f"平均準確率 {acc:.3f} - {subject}")
    print(f"{subject} 的推理時間: {t:.3f} 秒")

    # 印出此科目的 GPU 使用率
    if gpu_monitor:
        subject_gpu_util = gpu_monitor.get_subject_util(subject)
        print(f"{subject} 的 GPU 平均利用率: {subject_gpu_util:.2f}%")

    return cors, acc, np.array(all_probs), t

# 格式化 prompt 函數
def format_subject(s): return " ".join(s.split("_")).strip()
def format_example(df, i, include_answer=True):
    p = df.iloc[i, 0]; k = df.shape[1] - 2
    for j in range(k): p += f"\n{['A','B','C','D'][j]}. {df.iloc[i, j+1]}"
    p += "\n答案：" + (f" {df.iloc[i, -1]}\n\n" if include_answer else "")
    return p
def gen_prompt(df, s, k):
    p = f"以下是關於 {format_subject(s)} 的選擇題（附有答案）。\n\n"
    for i in range(k): p += format_example(df, i)
    return p

def get_subjects_to_evaluate(config: EvaluateConfig, all_subjects: List[str]) -> List[str]:
    """
    根據配置決定要評估的科目列表
    
    Args:
        config (EvaluateConfig): 評估配置
        all_subjects (List[str]): 所有可用的科目列表
    
    Returns:
        List[str]: 要評估的科目列表
    """
    if config.custom_subjects:
        # 驗證自定義科目是否都存在
        valid_subjects = [s for s in config.custom_subjects if s in all_subjects]
        if len(valid_subjects) != len(config.custom_subjects):
            invalid = set(config.custom_subjects) - set(valid_subjects)
            print(f"警告：以下自定義科目不存在，將被忽略：{invalid}")
        return valid_subjects
    
    if config.eval_mode == "core":
        # 只返回核心科目中存在的科目
        return [s for s in CORE_SUBJECTS if s in all_subjects]
    
    return all_subjects

def evaluate(config: EvaluateConfig, model_base):
    """
    評估模型效能
    
    Args:
        config (EvaluateConfig): 評估配置
        model_base (str): 基礎模型路徑
    """
    # 初始化 GPU 監測器
    gpu_monitor = GPUMonitor(interval=1)
    gpu_monitor.start()

    # 載入模型（包含可選的 LoRA 權重）
    model, tokenizer = load_llama_model(model_base, config.lora_path)
    
    # 獲取所有可用科目
    all_subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(config.data_dir, "test")) 
                          if f.endswith("_test.csv")])
    
    # 根據配置獲取要評估的科目
    subjects = get_subjects_to_evaluate(config, all_subjects)
    
    print(f"評估模式: {config.eval_mode}")
    print(f"要評估的科目數量: {len(subjects)}/{len(all_subjects)}")
    print("科目列表：", subjects)
    print("設定：", config)

    os.makedirs(config.save_dir, exist_ok=True)

    subj_acc, subj_time, all_cors = [], {}, []
    subj_gpu_utils = {}  # 儲存每個科目的 GPU 使用率

    for e in config.engine:
        engine_dir = os.path.join(config.save_dir, e)
        results_dir = os.path.join(engine_dir, f"results_{e}")
        summary_dir = os.path.join(engine_dir, f"summary_{e}")
        os.makedirs(engine_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)
        print(f"使用引擎：{e}")
        for s in subjects:
            dev_df = pd.read_csv(os.path.join(config.data_dir, "dev", s + "_dev.csv"), header=None)[:config.ntrain]
            test_df = pd.read_csv(os.path.join(config.data_dir, "test", s + "_test.csv"), header=None)
            cors, acc, probs, t = eval_local(config, s, model, tokenizer, dev_df, test_df, gpu_monitor)
            subj_acc.append((s, acc)); subj_time[s] = t; all_cors.extend(cors)

            # 儲存科目 GPU 使用率
            subj_gpu_utils[s] = gpu_monitor.get_subject_util(s)

            df = test_df.copy(); df[f"{e}_正確"] = cors
            for j in range(probs.shape[1]): df[f"{e}_選項{['A','B','C','D'][j]}_機率"] = probs[:, j]
            df.to_csv(os.path.join(results_dir, f"{s}.csv"), index=False)

        overall_acc = np.mean(all_cors) if all_cors else 0
        cat_summary = calculate_category_accuracies(dict(subj_acc))

        # 儲存科目摘要，並加入 GPU 使用率
        subj_summary_df = pd.DataFrame(subj_acc, columns=["科目", "準確率"])
        subj_summary_df["GPU利用率"] = subj_summary_df["科目"].map(subj_gpu_utils)
        subj_summary_df.to_csv(os.path.join(summary_dir, "subject_summary.csv"), index=False)

        # 更新類別摘要，加入 GPU 使用率
        cat_data = []
        for c, subs in SUBJECT_CATEGORIES.items():
            cat_subs = [s for s in subjects if any(sub.lower() in s.lower() for sub in subs)]
            avg_gpu = np.mean([subj_gpu_utils[s] for s in cat_subs if s in subj_gpu_utils]) if cat_subs else 0
            cat_data.append({
                "類別": c,
                "平均準確率": cat_summary[c],
                "平均推理時間(秒)": np.mean([subj_time[s] for s in cat_subs if s in subj_time]),
                "平均GPU利用率": avg_gpu
            })

        cat_data.append({
            "類別": "總體",
            "平均準確率": overall_acc,
            "平均推理時間(秒)": sum(subj_time.values()),
            "平均GPU利用率": gpu_monitor.get_average_util() if gpu_monitor.util_list else 0.0
        })

        pd.DataFrame(cat_data).to_csv(os.path.join(summary_dir, "category_summary.csv"), index=False)

        print("各科目平均推理時間：", subj_time)
        print("總推理時間：", sum(subj_time.values()))
        print("各科目GPU利用率：", subj_gpu_utils)
        print(f"平均 GPU 利用率 (引擎 {e}): {gpu_monitor.get_average_util():.2f}%")

        # 重置 GPU 監測器，以便為下一個引擎提供獨立的平均值
        gpu_monitor.util_list = []
        gpu_monitor.subject_utils = {}

    # 停止 GPU 監測
    gpu_monitor.stop()
    print(f"最終平均 GPU 利用率: {gpu_monitor.get_average_util():.2f}%")

'''
if __name__ == "__main__":
    config = EvaluateConfig()
    model_base = "/media/GradualWanda/out/llama2_7b"
    lora_path = "/path/to/your/lora/weights"  # 可選的 LoRA 權重路徑
    
    # 使用基礎模型評估
    evaluate(config, model_base)
    
    # 使用 LoRA 模型評估
    # evaluate(config, model_base, lora_path)
'''