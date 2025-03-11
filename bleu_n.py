import os
import torch
import evaluate
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from datasets import load_dataset

def get_model_path(model_dir):
    """
    檢查模型目錄，如果存在 snapshots 資料夾，則使用最新的 snapshot。
    否則，使用原始模型路徑。
    """
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

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路徑 {model_path} 不存在，請檢查是否下載正確。")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的設備: {device}")

# 載入 tokenizer，並確保有 pad_token. use model_base here.
tokenizer = AutoTokenizer.from_pretrained(model_base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"設定 pad_token 為 eos_token: {tokenizer.eos_token}")

# 載入模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"模型 {model_path} 載入完成，開始評估...")

# 設定 text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=None,  # 明確設定 temperature 為 None
    top_p=None,        # 明確設定 top_p 為 None
)

# 載入摘要資料集，例如 CNN/DailyMail 用於摘要生成
dataset = load_dataset("cnn_dailymail", "3.0.0")  # 載入 CNN/DailyMail 資料集
# 隨機選取 10 筆資料
sampled_data = dataset["test"].shuffle(seed=42).select([i for i in range(10)])

# 設定 BLEU 分數計算
metric = evaluate.load("bleu")

# 生成文本並計算 BLEU 分數
predictions = []
references = []

for example in tqdm(sampled_data):
    prompt = example["article"]  # 輸入文章
    reference = example["highlights"]  # 參考摘要

    # 減少生成的最大 tokens 長度，並分批處理
    max_tokens = 20  # 減少 max_new_tokens 的長度
    generated_text = ""
    
    # 逐步生成，避免一次性生成過多
    for i in range(0, len(prompt), 500):  # 每次處理500字
        chunk = prompt[i:i+500]
        # 使用 pipeline 生成文本
        part_generated = pipe(chunk, max_new_tokens=max_tokens, do_sample=False)[0]["generated_text"]
        generated_text += part_generated[len(chunk):]  # 拼接每一部分生成的文本
    
    # BLEU 計算需要 predictions 和 references 兩個部分
    predictions.append(generated_text.strip())  # 移除文本兩端的空白字符
    references.append([reference.strip()])  # 參考文本需要是列表形式

    print(f"提示: {prompt[:50]}...")  # 顯示文章開頭
    print(f"生成摘要: {generated_text[:50]}...")  # 顯示生成的摘要
    print(f"參考摘要: {reference[:50]}...\n")  # 顯示參考摘要

    # 清理 GPU 記憶體，避免 CUDA out of memory
    torch.cuda.empty_cache()

# 計算 BLEU 分數
results = metric.compute(predictions=predictions, references=references)

print(f"評估結果: {results}")
