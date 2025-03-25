# modules/evaluate.py
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.crop import crop
from config import EvaluateConfig  


# 定義 softmax 與領域分類
def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


SUBJECT_CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "biology", "chemistry", "computer_science", "mathematics", "medicine", "physics", "engineering", "statistics"],
    "Humanities": ["philosophy", "history", "world_religions", "law", "ethics"],
    "Social_Sciences": ["psychology", "sociology", "economics", "geography", "politics"],
    "Other": ["business", "health", "miscellaneous"]
}

# 計算各領域準確率
def calculate_category_accuracies(subjects, all_cors):
    category_acc = {cat: [] for cat in SUBJECT_CATEGORIES}
    for subject, cors in zip(subjects, all_cors):
        for category, subject_list in SUBJECT_CATEGORIES.items():
            if any(sub in subject for sub in subject_list):
                category_acc[category].extend(cors)
    for category, cors in category_acc.items():
        avg_acc = np.mean(cors) if cors else 0.0
        print(f"Average accuracy for {category}: {avg_acc:.3f}")

# 根據模型目錄取得最新的 snapshot
def get_model_path(model_dir):
    snapshot_dir = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshot_dir):
        snapshots = [os.path.join(snapshot_dir, d) for d in os.listdir(snapshot_dir)
                     if os.path.isdir(os.path.join(snapshot_dir, d))]
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            print(f"使用最新 snapshot: {latest_snapshot}")
            return latest_snapshot
    print(f"未找到 snapshots，使用原始路徑: {model_dir}")
    return model_dir

# 加載本地 Llama-2 模型與分詞器，使用 config.model_base 作為路徑
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,

        device_map="auto"
    )
    model.eval()
    print(f"模型 {model_path} 載入完成。")
    return model, tokenizer


def local_generate_logprobs(model, tokenizer, prompt, answers, max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        num_beams=3,
        do_sample=False
    )
    logits = outputs.scores[0]  # logits shape: (batch_size, vocab_size) ; batch_size=1
    answer_ids = []
    for ans in answers:
        tokenized = tokenizer(" " + ans, add_special_tokens=False)['input_ids']
        if len(tokenized) == 0:
            raise ValueError(f"無法 tokenize 選項 {ans}")
        answer_ids.append(tokenized[0])
    lprobs = [float(logits[0, token_id].item()) for token_id in answer_ids]
    return lprobs


def eval_local(config: EvaluateConfig, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = ["A", "B", "C", "D"]
    for i in range(test_df.shape[0]):
        k = config.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
        label = test_df.iloc[i, test_df.shape[1]-1]
        lprobs = local_generate_logprobs(model, tokenizer, prompt, answers, max_new_tokens=1)
        for idx, lp in enumerate(lprobs):
            if lp is None:
                print(f"Warning: {answers[idx]} not found. Artificially adding log prob of -100.")
                lprobs[idx] = -100
        pred = answers[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))
        cor = (pred == label)
        cors.append(cor)
        all_probs.append(probs)
    acc = np.mean(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(["A", "B", "C", "D"][j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
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


def evalute(config: EvaluateConfig, model_path: str):
    model, tokenizer = load_llama_model(model_path)
    engines = config.engine
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(config.data_dir, "test")) if "_test.csv" in f])
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    for engine in engines:
        engine_dir = os.path.join(config.save_dir, f"results_{engine}")
        if not os.path.exists(engine_dir):
            os.mkdir(engine_dir)
    print("Subjects:", subjects)
    print("Configuration:", config)
    for engine in engines:
        print("Evaluating with engine label:", engine)
        all_cors = []
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(config.data_dir, "dev", subject + "_dev.csv"), header=None)[:config.ntrain]
            test_df = pd.read_csv(os.path.join(config.data_dir, "test", subject + "_test.csv"), header=None)
            cors, acc, probs = eval_local(config, subject, model, tokenizer, dev_df, test_df)
            all_cors.append(cors)
            test_df[f"{engine}_correct"] = cors
            for j in range(probs.shape[1]):
                test_df[f"{engine}_choice{['A', 'B', 'C', 'D'][j]}_probs"] = probs[:, j]
            test_df.to_csv(os.path.join(config.save_dir, f"results_{engine}", f"{subject}.csv"), index=None)
        weighted_acc = np.mean(np.concatenate(all_cors))
        calculate_category_accuracies(subjects, all_cors)
        print("Overall average accuracy: {:.3f}".format(weighted_acc))


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

