import evaluate
import torch
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import re

def evaluate_accuracy(predictions: List[int], references: List[int]) -> float:
    """
    使用 Hugging Face Evaluate 庫計算分類任務的 Accuracy
    """
    accuracy_metric = evaluate.load("accuracy")
    result = accuracy_metric.compute(predictions=predictions, references=references)
    return result["accuracy"]

def evaluate_f1(predictions: List[int], references: List[int], average: str = "macro") -> Dict[str, float]:
    """
    使用 Hugging Face Evaluate 庫計算 F1 分數（可指定 average 方法）
    """
    f1_metric = evaluate.load("f1")
    result = f1_metric.compute(predictions=predictions, references=references, average=average)
    return result

def evaluate_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """
    使用 Hugging Face Evaluate 庫計算 BLEU 分數，適用於生成任務
    """
    bleu_metric = evaluate.load("bleu")
    result = bleu_metric.compute(predictions=predictions, references=references)
    return result["bleu"]

def evaluate_mmlu(predictions: List[int], references: List[int]) -> float:
    """
    假設 MMLU 評估以 Accuracy 衡量，這裡直接使用 accuracy 指標作為 MMLU 評分
    """
    mmlu_metric = evaluate.load("accuracy")
    result = mmlu_metric.compute(predictions=predictions, references=references)
    return result["accuracy"]

def evaluate_recall(predictions: List[int], references: List[int], average: str = "macro") -> Dict[str, float]:
    """
    使用 Hugging Face Evaluate 庫計算 Recall 分數（可指定 average 方法）
    """
    recall_metric = evaluate.load("recall")
    result = recall_metric.compute(predictions=predictions, references=references, average=average)
    return result

def create_few_shot_prompt(sample, subject):
    """
    建立帶有 few-shot 示例的提示

    Args:
        sample: 當前要評估的樣本
        subject: 科目名稱，用於選擇合適的示例

    Returns:
        包含 few-shot 示例的完整提示
    """
    few_shot_examples = get_examples_by_subject(subject)
    prompt = "以下是一些多選題示例，請選擇正確答案：\n\n"
    for i, example in enumerate(few_shot_examples, 1):
        prompt += f"問題 {i}: {example['question']}\n"
        prompt += "選項："
        for j, choice in enumerate(example['choices']):
            option_letter = chr(65 + j)
            prompt += f"[選項{option_letter}] {choice} "
        prompt += f"\n正確答案：選項{example['answer']}\n\n"
    prompt += f"現在請回答以下問題：\n\n"
    prompt += f"問題: {sample['question']}\n選項："
    choices = sample["choices"]
    if len(choices) == 4:
        prompt += f"[選項A] {choices[0]} [選項B] {choices[1]} [選項C] {choices[2]} [選項D] {choices[3]}\n"
    else:
        prompt += " ".join([f"[選項{chr(65+i)}] {c}" for i, c in enumerate(choices)]) + "\n"
    # 關鍵修改：在末尾加入換行和「答案：」標記
    prompt += "請選擇正確答案，並僅回覆答案字母：\n答案："
    return prompt

def get_examples_by_subject(subject):
    """
    根據科目提供適合的 few-shot 示例

    Args:
        subject: 科目名稱

    Returns:
        包含示例的列表
    """
    default_examples = [
        {
            "question": "若 f(x) = 3x^2 + 2x - 5，則 f(2) = ?",
            "choices": ["9", "13", "15", "17"],
            "answer": "C"  # 15 是正確答案
        },
        {
            "question": "集合 {1, 2, 3} 的所有子集數量是多少？",
            "choices": ["4", "6", "8", "16"],
            "answer": "C"  # 8 是正確答案
        },
        {
            "question": "若 x^2 - 6x + k = 0 有一重根，則 k = ?",
            "choices": ["6", "9", "12", "-9"],
            "answer": "B"  # 9 是正確答案
        }
    ]
    subject_examples = {
        "abstract_algebra": [
            {
                "question": "群 G 中，若元素 a 的階為 5，則 a^{-1} 的階為多少？",
                "choices": ["1", "5", "25", "不確定"],
                "answer": "B"
            },
            {
                "question": "令 R 為具有單位元的交換環，若 a, b ∈ R 且 ab = 0，則稱 a 和 b 互為零因子。以下哪一項陳述正確？",
                "choices": ["整數環中沒有零因子", "任何環都有零因子", "域中有零因子", "零環中沒有零因子"],
                "answer": "A"
            },
            {
                "question": "若 H 是群 G 的子群，則 G 中左陪集的個數等於：",
                "choices": ["H 的階", "G 的階除以 H 的階", "G 的階", "無法確定"],
                "answer": "B"
            }
        ],
    }
    return subject_examples.get(subject, default_examples)

def extract_answer_from_response(response: str) -> int:
    """
    從模型回應中提取答案選項

    Args:
        response: 模型生成的回應文本

    Returns:
        提取的選項索引 (0=A, 1=B, 2=C, 3=D)
    """
    # 先嘗試從 "答案：" 後提取答案字母
    m = re.search(r"答案[:：]?\s*([ABCD])", response, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        return ord(letter) - ord('A')
    # 再嘗試從 "請選擇正確答案" 之後提取
    m = re.search(r"請選擇正確答案[:：]?\s*([ABCD])", response, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        return ord(letter) - ord('A')
    # 若還找不到，則在整個回應中尋找第一個獨立的字母
    m = re.search(r'\b([ABCD])\b', response, re.IGNORECASE)
    if m:
        return ord(m.group(1).upper()) - ord('A')
    return 0  # 預設返回 A

def evaluate_model_on_dataset(model, tokenizer, dataset, metric_list: List[str], device: torch.device, batch_size: int = 1) -> Dict[str, Any]:
    """
    修改模型預測方式：使用 generate 生成文本，並從文本中解析答案
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = torch.tensor(batch["input_ids"]).to(device) if isinstance(batch["input_ids"], list) else batch["input_ids"].to(device)
            labels = batch["label"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = torch.ones_like(input_ids).to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,       # 使用 sampling
                max_new_tokens=30,     # 給足生成空間
                top_p=0.9,
                temperature=0.7
            )
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("Generated texts:", generated_texts)
            batch_predictions = []
            for text in generated_texts:
                predicted_label = extract_answer_from_response(text)
                batch_predictions.append(predicted_label)
            print("Batch labels:", labels)
            print("Batch predictions:", batch_predictions)
            valid_predictions = [pred for pred in batch_predictions if pred != -1]
            valid_references = [labels[i] for i, pred in enumerate(batch_predictions) if pred != -1]
            all_predictions.extend(valid_predictions)
            all_references.extend(valid_references)
    results = {}
    if "accuracy" in metric_list:
        results["accuracy"] = evaluate_accuracy(all_predictions, all_references) if all_predictions else 0.0
    if "f1" in metric_list:
        results["f1"] = evaluate_f1(all_predictions, all_references) if all_predictions else {"f1": 0.0}
    if "bleu" in metric_list:
        results["bleu"] = evaluate_bleu([], [])
    if "mmlu" in metric_list:
        results["mmlu"] = evaluate_mmlu(all_predictions, all_references) if all_predictions else 0.0
    if "recall" in metric_list:
        results["recall"] = evaluate_recall(all_predictions, all_references) if all_predictions else {"recall": 0.0}
    return results

if __name__ == "__main__":
    print("Evaluate Utils 模組載入成功！")
