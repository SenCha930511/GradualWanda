import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import json
from datetime import datetime
from mmlufun import construct_mmlu_prompt, extract_answer, generate_answer

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

def main():
    # model_base = "/media/GradualWanda/llm_weights/models--meta-llama--Llama-2-7b-hf"
    model_base = "/media/GradualWanda/merged_model"
    model_path = get_model_path(model_base)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路徑 {model_path} 不存在，請檢查是否下載正確。")
    print(f"使用的模型路徑: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的設備: {device}")

    # 確保輸出目錄存在
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 加載分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_base)
    if tokenizer.pad_token is None:
        # 如果 pad_token 和 eos_token 相同，可能需要指定其他 pad_token（若可能）\n
        tokenizer.pad_token = tokenizer.eos_token
        print(f"設定 pad_token 為 eos_token: {tokenizer.eos_token}")

    # 加載模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"模型 {model_path} 載入完成，開始進行 MMLU 評估...")

    # 修改生成参数，使之更接近官方评估：确定性解码，低温度，短生成长度
    gen_pipeline = torch.hub.load('huggingface/transformers', 'pipeline', 'text-generation',
        model=model, tokenizer=tokenizer) if False else None
    # 直接使用我们自定义的 generate_answer 函数，这里我们在其中设置 max_tokens=10
    # 如官方评估只生成一个 token 作为答案，则可设置为 10 tokens 保证不会生成额外内容

    # 創建輸出日誌文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"mmlu_evaluation_logs_{timestamp}.jsonl")
    summary_file = os.path.join(output_dir, f"mmlu_summary_{timestamp}.json")
    
    # 加載數據集：這裡使用 "cais/mmlu" 的 "college_physics" 配置，取前 100 筆樣本
    dataset = load_dataset("cais/mmlu", "college_physics", split="test").select(range(100))
    test_examples = dataset.select(range(15))  # 直接選取前15個例子進行測試

    predictions = []
    references = []
    all_results = []

    for i, example in enumerate(test_examples):
        # 構造 prompt，並加入明確指令，確保只生成答案（注意示例部分可根據需要調整）：
        prompt = construct_mmlu_prompt(example)
        prompt += "\n\nNow, answer ONLY the following question by providing a single uppercase letter (A, B, C, or D) as your final answer. Do not include any additional text."
        print(f"處理第 {i+1} 個例子...")
        
        try:
            # 生成答案，設定 max_tokens 為 10 避免多餘內容
            result = generate_answer(prompt, tokenizer, model, device, max_tokens=10)
            generated_text = result["generated_text"]
            # 若生成文本中包含額外的「\\nQ:」則截斷只保留目標答案部分
            if "\nQ:" in generated_text:
                generated_text = generated_text.split("\nQ:")[0].strip()
            pred_answer = extract_answer(generated_text)
            predictions.append(pred_answer)
            
            # 處理參考答案
            ref = chr(65 + example["answer"]) if isinstance(example["answer"], int) else str(example["answer"]).strip()
            references.append(ref)
            
            example_result = {
                "example_id": i + 1,
                "question": example["question"],
                "choices": example["choices"],
                "correct_answer": ref,
                "predicted_answer": pred_answer,
                "is_correct": pred_answer == ref,
                "prompt": prompt,
                "generated_text": generated_text,
                "full_output": result["full_output"]
            }
            all_results.append(example_result)
            
            # 將結果寫入日誌文件
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(example_result, ensure_ascii=False) + "\n")
            
            print(f"例子 {i+1}: 預測={pred_answer}, 正確答案={ref}, 正確={pred_answer==ref}")
        
        except Exception as e:
            print(f"處理第 {i+1} 個例子時出錯: {e}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"example_id": i + 1, "error": str(e), "prompt": prompt}, ensure_ascii=False) + "\n")
        
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 計算準確率：將字母轉換為整數索引
    valid_predictions = [ord(pred) - 65 for pred in predictions if pred in "ABCD"]
    valid_references = [ord(ref) - 65 for ref in references if ref in "ABCD"]

    if valid_predictions:
        accuracy_metric = evaluate.load("accuracy")
        result = accuracy_metric.compute(predictions=valid_predictions, references=valid_references)
        accuracy = result["accuracy"]
    else:
        accuracy = 0

    summary = {
        "total_examples": len(test_examples),
        "valid_predictions": len(valid_predictions),
        "accuracy": accuracy,
        "model_path": model_path,
        "timestamp": timestamp
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果已保存至 {log_file} 和 {summary_file}")
    return all_results[:5]

if __name__ == "__main__":
    detailed_results = main()
