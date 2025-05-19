# modules/lora.py
import gc
import torch
from config import LoRaConfig  
from datasets import load_dataset
from peft import get_peft_model, LoraConfig as PeftLoraConfig, TaskType
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    EarlyStoppingCallback
)

def lora_finetune(config: LoRaConfig, model_path: str):
    """
    使用自定義 LoRaConfig (dataclass) 的 LoRA 微調函式。
    """

    print(f"Loading model and tokenizer from {model_path}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))

    gc.collect()
    torch.cuda.empty_cache()

    peft_lora_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type
    )

    model = get_peft_model(model, peft_lora_config)
    print("LoRA applied to model.")

    dataset = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train'
    )
    
    # 分割訓練集和驗證集
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_length)
    
    remove_columns = [col for col in train_dataset.column_names if col != "text"]
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=min(config.epochs, 3),  # 限制最大訓練輪數為3
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=500,  # 每500步評估一次
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,  # 載入最佳模型
        metric_for_best_model="eval_loss",  # 使用驗證集loss作為評估指標
        greater_is_better=False  # loss越小越好
    )
    
    # 創建早停回調
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # 如果3次評估沒有改善就停止
        early_stopping_threshold=0.01  # 如果改善小於0.01就認為沒有改善
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping_callback]
    )
    
    print("Starting LoRA fine-tuning...")
    trainer.train()

    output_dir = config.output_dir if config.output_dir else "lora_finetuned_model"
    print(f"Saving LoRA fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    print("LoRA fine-tuning completed.\n")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()