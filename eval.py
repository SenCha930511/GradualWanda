import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import json
# 假设 evaluate_utils.py 文件已经修改为支持 10-shot
from evaluate_utils import create_few_shot_prompt, extract_answer_from_response

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

# 設定模型基本路徑與設備
model_base = "/media/GradualWanda/llm_weights/models--meta-llama--Llama-2-7b-hf"
model_path = get_model_path(model_base)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 tokenizer，並確保有 pad_token
tokenizer = AutoTokenizer.from_pretrained(model_base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 載入 MMLU 數據集，這裡以 "abstract_algebra" 為例，選取前 100 個樣本
dataset = load_dataset("cais/mmlu", "abstract_algebra", split="test").select(range(100))
print(f"Dataset structure: {dataset[0]}")  # 應包含 "question", "choices", "answer"

# 這裡我們不再進行額外的預處理，保留原始結構以便 few-shot 提示使用

# 載入 MMLU 評估指標（這裡用 accuracy 作為評分）
mmlu_metric = load("accuracy")

def generate_answer(prompt: str) -> str:
    """
    根據提示生成模型回答文本
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=10,  # 保持 max_new_tokens 为 10 (或尝试 5-15 之间的值)
        do_sample=True,    # 启用 sampling
        temperature=0.7,  # 设置 temperature，例如 0.7 (可尝试 0.5-0.9 范围)
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

all_predictions = []
all_references = []
prompts = [] #  用于保存每个样本的 prompt
generated_texts = [] # 用于保存每个样本的生成文本
mapping = {"A": 0, "B": 1, "C": 2, "D": 3}

for sample in tqdm(dataset):
    # 使用 few-shot 提示構造器生成 10-shot 提示，這裡 subject 為 "abstract_algebra"
    prompt = create_few_shot_prompt(sample, "abstract_algebra", num_shots=10) # 明确指定 num_shots=10
    prompts.append(prompt) # 保存 prompt
    generated_text = generate_answer(prompt)
    generated_texts.append(generated_text) # 保存生成文本
    pred = extract_answer_from_response(generated_text)
    all_predictions.append(pred)
    # 將參考答案轉換為數值
    reference = sample["answer"].strip() if isinstance(sample["answer"], str) else str(sample["answer"])
    ref_val = mapping.get(reference, -1)
    all_references.append(ref_val)

    # 輸出調試信息
    print("\n---")
    print(f"Prompt:\n{prompt}")
    print(f"Generated text: {generated_text}")
    print(f"Predicted label: {pred}, Reference label: {ref_val}")

# 計算 MMLU 評估指標（以 accuracy 為基準）
if all_predictions and all_references:
    score = mmlu_metric.compute(predictions=all_predictions, references=all_references)
    print("\nMMLU Accuracy:", score)
else:
    print("Error: No valid predictions or references.")

# 儲存評估結果
results = {
    "mmlu_accuracy": score,
    "samples": [
        {
            "question": dataset[i]["question"],
            "choices": dataset[i]["choices"],
            "reference": dataset[i]["answer"],
            "prediction": all_predictions[i],
            "prompt": prompts[i], # 保存 prompt 到 JSON
            "generated_text": generated_texts[i] # 保存生成文本到 JSON
        }
        for i in range(min(10, len(dataset)))
    ]
}

with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("評估結果已保存至 evaluation_results.json")