# mmlufun.py
import torch

def construct_mmlu_prompt(target_example):
    prompt = (
        "You are an expert in college physics. Please carefully read the following multiple-choice question and select the best answer. "
        "Provide ONLY a single uppercase letter (A, B, C, or D) as your final answer, followed by a brief explanation.\n\n"
    )
    prompt += f"Q: {target_example['question'].strip()}\n"
    for i, choice in enumerate(target_example["choices"]):
        prompt += f"{chr(65 + i)}. {choice.strip()}\n"
    prompt += "\nFinal Answer: "
    return prompt

def extract_answer(generated_text):
    """
    从生成的文本中提取第一个出现的答案字母 (A, B, C, or D) 作为预测答案。
    如果生成文本中包含 '\\nQ:'，则只保留其前面的部分。
    """
    if "\nQ:" in generated_text:
        generated_text = generated_text.split("\nQ:")[0].strip()
    for ch in generated_text:
        if ch in "ABCD":
            return ch
    print(f"警告: 无法从文本中提取答案。返回默认值 'X'。生成文本: {generated_text[:50]}...")
    return "X"


def generate_answer(prompt, tokenizer, model, device, max_tokens):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    prompt_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,      
            temperature=0.1,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            #early_stopping=True,       # 當生成達到 eos_token 或滿足 early stopping 條件時提前停止
            no_repeat_ngram_size=2      # 不允許重複2-gram，減少重複生成
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_tokens = output_ids[0][prompt_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "generated_text": generated_text,
        "full_output": full_output,
        "prompt": prompt
    }

