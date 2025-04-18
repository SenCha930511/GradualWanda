import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import EvaluateConfig  # 從 config 資料夾中載入 EvaluateConfig
import time
import gc
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



# 定義 softmax 與領域分類

def softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


# 學科類別

SUBJECT_CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "biology", "chemistry", "computer_science", "mathematics", "medicine", "physics", "engineering", "statistics"],
    "Humanities": ["philosophy", "history", "world_religions", "law", "ethics"],
    "Social_Sciences": ["psychology", "sociology", "economics", "geography", "politics"],
    "Other": ["business", "health", "miscellaneous"]
}

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
def load_llama_model(model_base):
    model_path = get_model_path(model_base)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路徑 {model_path} 不存在，請檢查是否下載正確。")
    print(f"使用的模型路徑: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的設備: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"設定 pad_token 為 eos_token: {tokenizer.eos_token}")
    tokenizer.padding_side = "left"  # 建議使用 left padding
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,  # 如果要使用 8-bit，可保留這行；如果希望保持原始數值格式，可以移除此參數
        device_map="auto"
    )
    model.eval()
    print(f"模型 {model_path} 載入完成。")
    return model, tokenizer

# 若 prompt 太長則使用此函數從右側保留最大長度 tokens
def crop_prompt(prompt, tokenizer, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    if inputs.input_ids.shape[1] > max_length:
        input_ids = inputs.input_ids[:, -max_length:]
        prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return prompt

# 修改 local_generate_logprobs 函數，使用完整答案 token 序列中的第二個 token 當作候選 token
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
        tokenized = tokenizer(" " + ans, add_special_tokens=False)['input_ids']
        if len(tokenized) == 0:
            raise ValueError(f"無法 tokenize 選項 {ans}")
        token_id = tokenized[1] if len(tokenized) > 1 else tokenized[0]
        answer_ids.append(token_id)
    lprobs = [float(logits[token_id].item()) for token_id in answer_ids]
    return lprobs

def eval_local(config, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = ["A", "B", "C", "D"]
    max_seq_len = model.config.max_position_embeddings

    # 使用 torch.cuda.Event 計算推理時間
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    for i in range(test_df.shape[0]):
        k = config.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompt = crop_prompt(prompt, tokenizer, max_seq_len)
        label = test_df.iloc[i, test_df.shape[1]-1]
        lprobs = local_generate_logprobs(model, tokenizer, prompt, answers)
        pred = answers[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))
        cors.append(pred == label)
        all_probs.append(probs)
    
    end_event.record()
    torch.cuda.synchronize()
    inference_time = start_event.elapsed_time(end_event) / 1000
    acc = np.mean(cors)
    all_probs = np.array(all_probs)
    print("平均準確率 {:.3f} - {}".format(acc, subject))
    print(f"{subject} 的推理時間: {inference_time:.3f} 秒")
    return cors, acc, all_probs, inference_time

def format_subject(subject):
    return " ".join(subject.split("_")).strip()

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(["A", "B", "C", "D"][j], df.iloc[idx, j+1])
    prompt += "\n答案："
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def evaluate(config: EvaluateConfig):
    model, tokenizer = load_llama_model(config.model_base)
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(config.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    summary_dir = os.path.join(config.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    for engine in config.engine:
        os.makedirs(os.path.join(config.save_dir, f"results_{engine}"), exist_ok=True)
    print("科目：", subjects)
    print("設定：", config)

    subject_accuracies = {}
    subject_inference_times = {}
    all_cors_list = []

    for engine in config.engine:
        print(f"使用引擎標籤評估： {engine}")
        for subject in subjects:
            dev_path = os.path.join(config.data_dir, "dev", subject + "_dev.csv")
            test_path = os.path.join(config.data_dir, "test", subject + "_test.csv")
            dev_df = pd.read_csv(dev_path, header=None)[:config.ntrain]
            test_df = pd.read_csv(test_path, header=None)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    cors, acc, probs, inference_time = eval_local(config, subject, model, tokenizer, dev_df, test_df)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA 記憶體不足，科目 {subject}，嘗試 {attempt+1}/{max_retries}。清理快取並重試。")
                        torch.cuda.empty_cache()
                        time.sleep(5)
                    else:
                        raise e
            else:
                print(f"由於記憶體不足，{max_retries} 次嘗試後處理 {subject} 失敗。")
                continue

            subject_accuracies[subject] = acc
            subject_inference_times[subject] = inference_time
            all_cors_list.extend(cors)

            test_df[f"{engine}_正確"] = cors
            for j in range(probs.shape[1]):
                test_df[f"{engine}_選項{['A','B','C','D'][j]}_機率"] = probs[:, j]
            save_path = os.path.join(config.save_dir, f"results_{engine}", f"{subject}.csv")
            test_df.to_csv(save_path, index=False)
            print(f"{subject}.csv 已處理完成，結果儲存於 {save_path}，準確率為 {acc:.3f}")
            torch.cuda.empty_cache()

    overall_acc = np.mean(all_cors_list) if all_cors_list else 0.0
    category_summary = calculate_category_accuracies(subject_accuracies)

    subject_df = pd.DataFrame(list(subject_accuracies.items()), columns=["科目", "準確率"])
    subject_df.to_csv(os.path.join(summary_dir, "subject_summary.csv"), index=False)

    category_inference_times = {}
    for category, subs in SUBJECT_CATEGORIES.items():
        relevant_times = [subject_inference_times[sub] for sub in subs if sub in subject_inference_times]
        category_inference_times[category] = np.mean(relevant_times) if relevant_times else 0.0

    cat_data = []
    for cat, avg in category_summary.items():
        cat_data.append({
            "類別": cat,
            "平均準確率": avg,
            "平均推理時間（秒）": category_inference_times.get(cat, 0.0)
        })
    cat_data.append({
        "類别": "總體",
        "平均準確率": overall_acc,
        "平均推理時間（秒）": sum(subject_inference_times.values())
    })
    category_df = pd.DataFrame(cat_data)
    category_df.to_csv(os.path.join(summary_dir, "category_summary.csv"), index=False)

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


if __name__ == "__main__":
    config = EvaluateConfig.default()
    evalute(config)


