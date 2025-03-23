import argparse
import os
import numpy as np
import pandas as pd
import time
import torch
import json
from lib.crop import crop
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 你原有的工具函數
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

# 新增的統計函數
def calculate_category_accuracies(subjects, all_cors):
    category_acc = {cat: [] for cat in SUBJECT_CATEGORIES}

    for subject, cors in zip(subjects, all_cors):
        for category, subject_list in SUBJECT_CATEGORIES.items():
            if any(sub in subject for sub in subject_list):
                category_acc[category].extend(cors)

    # 計算各領域平均準確率
    for category, cors in category_acc.items():
        avg_acc = np.mean(cors) if cors else 0.0
        print(f"Average accuracy for {category}: {avg_acc:.3f}")

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

# 加載本地 Llama-2 模型與分詞器
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

# 用本地模型生成單 token 並返回對應 logits 的 log 機率
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
    # outputs.scores 是一個 tuple，對於只生成一個 token，其長度為1
    logits = outputs.scores[0]  # shape: (batch_size, vocab_size) ; batch_size=1
    # 取得各選項的 token id（注意：通常要在選項前加空格）
    answer_ids = []
    for ans in answers:
        # tokenize 選項時不加入特殊 token
        tokenized = tokenizer(" " + ans, add_special_tokens=False)['input_ids']
        if len(tokenized) == 0:
            raise ValueError(f"無法 tokenize 選項 {ans}")
        answer_ids.append(tokenized[0])
    # 取得對應 token 的 logit
    lprobs = [float(logits[0, token_id].item()) for token_id in answer_ids]
    return lprobs

# 修改 eval() 函數，將 openai API 替換為本地生成
def eval_local(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = ["A", "B", "C", "D"]

    for i in range(test_df.shape[0]):
        # 構造 prompt，這部分保持與原來一致
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        # 若 prompt 太長則 crop（原邏輯保持不變）
        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        # 使用本地生成取得 logprobs
        lprobs = local_generate_logprobs(model, tokenizer, prompt, answers, max_new_tokens=1)
        # 若某選項因 token 化失敗，則補 -100 分
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

# 以下兩個函數保持不變（用於格式化 prompt）
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

def main(args):
    # 這裡指定本地模型所在路徑
    model_base = "/media/GradualWanda/llm_weights/models--meta-llama--Llama-2-7b-hf"
    model, tokenizer = load_llama_model(model_base)

    engines = args.engine  # 由於我們本地只使用一個模型，可以忽略多 engine 設置，或用作標籤
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    for engine in engines:
        engine_dir = os.path.join(args.save_dir, f"results_{engine}")
        if not os.path.exists(engine_dir):
            os.mkdir(engine_dir)

    print("Subjects:", subjects)
    print("Arguments:", args)

    for engine in engines:
        print("Evaluating with engine label:", engine)
        all_cors = []
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
            cors, acc, probs = eval_local(args, subject, model, tokenizer, dev_df, test_df)
            all_cors.append(cors)

            # 將結果保存到 CSV 中
            test_df["{}_correct".format(engine)] = cors
            for j in range(probs.shape[1]):
                test_df["{}_choice{}_probs".format(engine, ["A", "B", "C", "D"][j])] = probs[:, j]
            test_df.to_csv(os.path.join(args.save_dir, f"results_{engine}", f"{subject}.csv"), index=None)

        weighted_acc = np.mean(np.concatenate(all_cors))
        calculate_category_accuracies(subjects, all_cors)
        print("Overall average accuracy: {:.3f}".format(weighted_acc))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of training examples to use for prompt.")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Directory containing dev/test CSV files.")
    parser.add_argument("--save_dir", "-s", type=str, default="results/results_models--meta-llama--Llama-2-7b-hf", help="Directory to save evaluation results.")
    parser.add_argument("--engine", "-e", nargs="+", default=["llama2"], help="Engine label (dummy when using local model).")
    args = parser.parse_args()
    main(args)
