import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import json

def get_model_path(model_dir):
    snapshot_dir = os.path.join(model_dir, "snapshots")
    if os.path.exists(snapshot_dir):
        snapshots = [os.path.join(snapshot_dir, d) for d in os.listdir(snapshot_dir) if os.path.isdir(os.path.join(snapshot_dir, d))]
        if snapshots:
            latest_snapshot = max(snapshots, key=os.path.getmtime)
            print(f"使用最新 snapshot: {latest_snapshot}")
            return latest_snapshot
    print(f"未找到 snapshots，使用原始路徑: {model_dir}")
    return model_dir

model_base = "/media/GradualWanda/llm_weights/models--meta-llama--Llama-2-7b-hf"
model_path = get_model_path(model_base)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=True
)
model.eval()

dataset = load_dataset("cnn_dailymail", "3.0.0", split="test").select(range(100))
print(f"the type of dataset is {type(dataset)}")

if isinstance(dataset[0], dict) and "article" in dataset[0] and "highlights" in dataset[0]:
    print("Dataset structure is valid.")
else:
    print("Error: Dataset structure is invalid.")
    exit()

def preprocess_sample(sample):
    processed = {
        "article": sample["article"],
        "highlights": sample["highlights"]
    }
    return processed

dataset = dataset.map(preprocess_sample)

print(f"Dataset format: {type(dataset)}")
print(f"Sample structure: {dataset[0]}")

bleu = load("bleu")

def generate_summary(article):
    inputs = tokenizer(article, return_tensors="pt", truncation=True, max_length=512).to(device) # 保留 tokenizer 的 max_length
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,         # 限制新生成的 token 數量
        num_beams=2,                # 使用 beam search 進行搜索
        no_repeat_ngram_size=3,     # 防止生成重複 n-gram
        length_penalty=2.0,         # 激勵生成合適長度
        early_stopping=True,        # 達到結束條件即停止生成
        temperature=0.7,            # 如果使用採樣，控制隨機性
        top_p=0.9                   # 採樣時選擇概率質量為 0.9 的 token
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

references = []
predictions = []


for sample in tqdm(dataset):
    if isinstance(sample, dict):
        article = sample["article"]
        reference = sample["highlights"]
        prediction = generate_summary(article)
        references.append([reference.split()])
        predictions.append(prediction.split())
    else:
        print(f"Warning: sample is not a dictionary, skipping: {sample}")

if references and predictions:
    score = bleu.compute(predictions=predictions, references=references)
    print("\nBLEU 分數:", score)

    results = {
        "bleu_score": score,
        "samples": [
            {
                "article": dataset[i]["article"][:300] + "...",
                "reference": dataset[i]["highlights"],
                "prediction": " ".join(predictions[i])
            }
            for i in range(min(10, len(dataset)))
        ]
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("評估結果已保存至 evaluation_results.json")
else:
    print("Error: No valid samples found. Evaluation skipped.")
