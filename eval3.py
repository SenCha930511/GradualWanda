import os
import torch
import evaluate
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

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

# 自定義文本，作為測試資料
test_prompts = [
    "The future of AI in education is",
    "Once upon a time, in a faraway land",
    "The key to success in technology is",
    "In the year 2050, the world will be",
    "The most important lesson in life is"
]

# 人工生成的參考文本，作為對比
reference_texts = [
    "The future of AI in education is bright, with many new technologies that help students learn.",
    "Once upon a time, in a faraway land, there lived a king who ruled justly and wisely.",
    "The key to success in technology is hard work, innovation, and the ability to adapt to change.",
    "In the year 2050, the world will be much more connected, with advanced technologies and global collaboration.",
    "The most important lesson in life is to be kind, patient, and always strive for improvement."
]

# 設定 BLEU 分數計算
metric = evaluate.load("bleu")

# 生成文本並計算 BLEU 分數
predictions = []
references = []

for prompt, reference in zip(test_prompts, reference_texts):
    # 讓模型生成文本
    generated_text = pipe(prompt, max_new_tokens=50, do_sample=False)[0]["generated_text"]
    
    # BLEU 計算需要 predictions 和 references 兩個部分
    predictions.append(generated_text.strip())  # 移除文本兩端的空白字符
    references.append(reference.strip())  # 移除參考文本兩端的空白字符

    print(f"提示: {prompt}")
    print(f"生成文本: {generated_text[:50]}...")
    print(f"參考文本: {reference[:50]}...\n")

# 計算 BLEU 分數
results = metric.compute(predictions=predictions, references=references)

print(f"評估結果: {results}")
