import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import EvaluateConfig  # 從 config 資料夾中載入 EvaluateConfig
import time
import threading
import gc
import numpy as np
import pandas as pd
import torch
import pynvml
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig



# 定義 softmax 與領域分類

def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

# GPU 使用量監測線程
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
                self.subject_utils.setdefault(self.current_subject, []).append(util.gpu)
            time.sleep(self.interval)

    def set_subject(self, subject):
        self.current_subject = subject

    def get_subject_util(self, subject):
        vals = self.subject_utils.get(subject, [])
        return sum(vals) / len(vals) if vals else 0.0

    def get_average_util(self):
        return sum(self.util_list) / len(self.util_list) if self.util_list else 0.0

# 學科類別
SUBJECT_CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "biology", "chemistry", "computer_science", "mathematics", "medicine", "physics", "engineering", "statistics"],
    "Humanities": ["philosophy", "history", "world_religions", "law", "ethics"],
    "Social_Sciences": ["psychology", "sociology", "economics", "geography", "politics"],
    "Other": ["business", "health", "miscellaneous"]
}

def calculate_category_accuracies(subject_accuracies):
    category_acc = {cat: [] for cat in SUBJECT_CATEGORIES}
    for subject, acc in subject_accuracies.items():
        for category, subs in SUBJECT_CATEGORIES.items():
            if any(sub.lower() in subject.lower() for sub in subs):
                category_acc[category].append(acc)
    summary = {cat: (np.mean(vals) if vals else 0.0) for cat, vals in category_acc.items()}
    for cat, avg in summary.items():
        print(f"{cat} 的平均準確率: {avg:.3f}")
    return summary

# 取得模型路徑
def get_model_path(model_dir):
    snapshots = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshots):
        dirs = [os.path.join(snapshots, d) for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
        if dirs:
            latest = max(dirs, key=os.path.getmtime)
            print(f"使用最新快照: {latest}")
            return latest
    print(f"未找到快照，使用原始路徑: {model_dir}")
    return model_dir

# 載入本地 Llama-2 模型與分詞器
def load_llama_model(model_base):
    path = get_model_path(model_base)
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型路徑 {path} 不存在。")
    print(f"使用的模型路徑: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的設備: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,   # 允許把部分模組 offload 到 CPU
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        #load_in_8bit=True,
        device_map="auto"
    )
    model.eval()
    print(f"模型載入完成。")
    return model, tokenizer

# 裁剪 prompt
def crop_prompt(prompt, tokenizer, max_len):
    tokens = tokenizer(prompt, return_tensors="pt")
    if tokens.input_ids.size(1) > max_len:
        trimmed = tokens.input_ids[:, -max_len:]
        return tokenizer.decode(trimmed[0], skip_special_tokens=True)
    return prompt

# 計算 log-probs
def local_generate_logprobs(model, tokenizer, prompt, choices):
    max_len = model.config.max_position_embeddings
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
    mask = inp.attention_mask
    seq = inp.input_ids.size(-1)
    non_pad = int(mask.sum().item())
    pos_ids = torch.cat([
        torch.zeros(seq - non_pad, dtype=torch.long, device=inp.input_ids.device),
        torch.arange(non_pad, dtype=torch.long, device=inp.input_ids.device)
    ]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=inp.input_ids, attention_mask=mask, position_ids=pos_ids)
    logit = out.logits[0, -1]
    ids = []
    for ch in choices:
        tok = tokenizer(" " + ch, add_special_tokens=False)["input_ids"]
        ids.append(tok[1] if len(tok) > 1 else tok[0])
    return [float(logit[i].item()) for i in ids]

# 推理函數
def eval_local(config, subj, model, tokenizer, dev_df, test_df, monitor=None):
    if monitor: monitor.set_subject(subj)
    cors, all_probs, timeframe = [], [], ["A","B","C","D"]
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(len(test_df)):
        prompt = gen_prompt(dev_df, subj, config.ntrain) + format_example(test_df, i, False)
        prompt = crop_prompt(prompt, tokenizer, model.config.max_position_embeddings)
        label = test_df.iloc[i, -1]
        lps = local_generate_logprobs(model, tokenizer, prompt, timeframe)
        pred = timeframe[np.argmax(lps)]; probs = softmax(np.array(lps))
        cors.append(pred == label); all_probs.append(probs)
    end.record(); torch.cuda.synchronize()
    duration = start.elapsed_time(end) / 1000
    acc = np.mean(cors)
    print(f"{subj} 準確率 {acc:.3f}, 時間 {duration:.3f}s")
    if monitor:
        print(f"{subj} GPU 利用率: {monitor.get_subject_util(subj):.2f}%")
    return cors, acc, np.array(all_probs), duration

# 格式化輔助函數
def format_subject(s): return " ".join(s.split("_")).strip()
def format_example(df, idx, inc_ans=True):
    txt = df.iloc[idx, 0]
    opts = ['A','B','C','D']
    for j in range(df.shape[1] - 2): txt += f"\n{opts[j]}. {df.iloc[idx, j+1]}"
    return txt + (f"\n答案： {df.iloc[idx, -1]}\n\n" if inc_ans else "")
def gen_prompt(df, subj, k):
    p = f"以下是關於 {format_subject(subj)} 的選擇題（附有答案）。\n\n"
    for i in range(k): p += format_example(df, i)
    return p

# 主流程
def evaluate(config: EvaluateConfig, model_path: str):
    gpu_monitor = GPUMonitor()
    gpu_monitor.start()

    model, tokenizer = load_llama_model(model_path)
    subjects = sorted([f.replace("_test.csv", "") for f in os.listdir(os.path.join(config.data_dir, "test")) if f.endswith("_test.csv")])

    # 依 engine 標籤建立輸出資料夾
    for e in config.engine:
        base_out = os.path.join(config.save_dir, e)
        results_dir = os.path.join(base_out, f"results_{e}")
        summary_dir = os.path.join(base_out, f"summary_{e}")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(summary_dir, exist_ok=True)

        subj_acc, subj_time, all_cors = [], {}, []
        subj_gpu = {}

        # 逐科目推理
        for s in subjects:
            dev = pd.read_csv(os.path.join(config.data_dir, "dev", f"{s}_dev.csv"), header=None)[:config.ntrain]
            test = pd.read_csv(os.path.join(config.data_dir, "test", f"{s}_test.csv"), header=None)
            cors, acc, probs, tm = eval_local(config, s, model, tokenizer, dev, test, gpu_monitor)
            subj_acc.append((s, acc)); subj_time[s] = tm; all_cors.extend(cors)
            subj_gpu[s] = gpu_monitor.get_subject_util(s)

            df = test.copy()
            df[f"{e}_正確"] = cors
            for j in range(probs.shape[1]): df[f"{e}_選項{['A','B','C','D'][j]}_機率"] = probs[:, j]
            df.to_csv(os.path.join(results_dir, f"{s}.csv"), index=False)

        gpu_monitor.stop()
        avg_gpu = gpu_monitor.get_average_util()

        overall_acc = np.mean(all_cors) if all_cors else 0
        cat_summary = calculate_category_accuracies(dict(subj_acc))

        # 科目摘要
        subj_df = pd.DataFrame(subj_acc, columns=["科目","準確率"])
        subj_df["GPU利用率"] = subj_df["科目"].map(subj_gpu)
        subj_df.to_csv(os.path.join(summary_dir, "subject_summary.csv"), index=False)

        # 類別摘要
        cat_data = []
        for cat, subs in SUBJECT_CATEGORIES.items():
            matched = [s for s in subjects if any(x in s for x in subs)]
            avg_tm = np.mean([subj_time[s] for s in matched]) if matched else 0
            avg_gpu_c = np.mean([subj_gpu[s] for s in matched]) if matched else 0
            cat_data.append({"類別": cat, "平均準確率": cat_summary[cat], "平均推理時間(秒)": avg_tm, "平均GPU利用率": avg_gpu_c})
        cat_data.append({"類別": "總體", "平均準確率": overall_acc, "平均推理時間(秒)": sum(subj_time.values()), "平均GPU利用率": avg_gpu})
        pd.DataFrame(cat_data).to_csv(os.path.join(summary_dir, "category_summary.csv"), index=False)

    subject_avg_inference_times = {subject: t for subject, t in subject_inference_times.items()}
    print("\n各科目推理時間：")
    for subject, t in subject_avg_inference_times.items():
        print(f"{subject}: {t:.3f} 秒")
    total_time = sum(subject_inference_times.values())
    print(f"\n總推理時間：{total_time:.3f} 秒")
    print(f"摘要已保存在 {summary_dir}")
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
    evaluate(config)
'''