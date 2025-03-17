import os
import torch
import evaluate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def get_model_path(model_dir):
    """
    檢查模型目錄，如果存在 snapshots 資料夾，則使用最新的 snapshot。
    否則，使用原始模型路徑。
    """
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

# 設定模型基本路徑與設備
model_base = "/media/GradualWanda/out/llama2_7b"
model_path = get_model_path(model_base)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路徑 {model_path} 不存在，請檢查是否下載正確。")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的設備: {device}")

# 載入 tokenizer，確保 pad_token 存在
tokenizer = AutoTokenizer.from_pretrained(model_base)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"設定 pad_token 為 eos_token: {tokenizer.eos_token}")

# 載入 LLaMA 2 模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"模型 {model_path} 載入完成，開始評估...")

# 建立 text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    truncation=True
)

# 從 datasets 載入摘要數據集
# 使用 CNN/DailyMail 數據集作為示例，如需其他數據集可替換
dataset = load_dataset("MIMIC-CXR", "3.0.0", split="test[:10]")  # 僅使用前10個樣本進行測試
print(f"已載入 {len(dataset)} 個測試樣本")

# 載入 ROUGE 評估器
rouge = evaluate.load("rouge")

predictions = []
references = []

for i, sample in enumerate(dataset):
    # 構建提示詞
    article = sample["article"]
    reference = sample["highlights"]
    
    prompt = f"""請為以下文章生成一個簡潔的摘要:

{article}

摘要:"""
    
    print(f"正在處理第 {i+1}/{len(dataset)} 個樣本...")
    
    # 生成摘要
    try:
        output = pipe(prompt, return_full_text=False)[0]["generated_text"].strip()
        
        # 只保留生成內容的第一段作為摘要
        # 這裡假設第一段是最相關的摘要內容
        if "\n\n" in output:
            generated_summary = output.split("\n\n")[0]
        else:
            generated_summary = output
            
        predictions.append(generated_summary)
        references.append(reference)
        
        # 打印進度和示例
        if i < 2:  # 只顯示前兩個樣本的詳細信息
            print(f"原文: {article[:200]}...")
            print(f"生成摘要: {generated_summary}")
            print(f"參考摘要: {reference}\n")
    
    except Exception as e:
        print(f"處理樣本 {i} 時出錯: {e}")
    
    # 清理 GPU 記憶體
    torch.cuda.empty_cache()

# 計算 ROUGE 分數
results = rouge.compute(
    predictions=predictions, 
    references=references,
    use_stemmer=True
)

# 打印詳細結果
print("\n評估結果:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")

# 計算平均分數
avg_rouge = sum(results.values()) / len(results)
print(f"\n平均 ROUGE 分數: {avg_rouge:.4f}")

# 保存結果到文件
with open("llama2_7b_summarization_results2.txt", "w") as f:
    f.write("LLaMA-2-7B 摘要評估結果\n")
    f.write("-----------------------\n\n")
    f.write(f"模型: {model_path}\n")
    f.write(f"測試樣本數: {len(predictions)}\n\n")
    f.write("ROUGE 分數:\n")
    for metric, score in results.items():
        f.write(f"{metric}: {score:.4f}\n")
    f.write(f"\n平均 ROUGE 分數: {avg_rouge:.4f}\n")
    
    # 保存一些示例
    f.write("\n示例摘要對比:\n")
    for i in range(min(3, len(predictions))):
        f.write(f"\n示例 {i+1}:\n")
        f.write(f"參考摘要: {references[i]}\n")
        f.write(f"生成摘要: {predictions[i]}\n")
        f.write("-----------------------\n")

print(f"\n結果已保存到 llama2_7b_summarization_results.txt")