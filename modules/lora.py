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

    try:
        # 首先嘗試從本地載入
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
    except Exception as e:
        print(f"Local tokenizer loading failed: {e}")
        print("Attempting to download tokenizer from Hugging Face...")
        try:
            # 如果本地載入失敗，嘗試從 Hugging Face 下載
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                force_download=True,
                local_files_only=False
            )
        except Exception as e:
            print(f"Tokenizer download failed: {e}")
            raise Exception("無法載入 tokenizer，請檢查模型路徑和網絡連接")

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
        split='train[:5%]'
    )
    
    # 分割訓練集和驗證集
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=config.max_length,
            padding="max_length"
        )
    
    remove_columns = [col for col in train_dataset.column_names if col != "text"]
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=min(config.epochs, 2), 
        per_device_train_batch_size=config.per_device_train_batch_size,  
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=16,
        eval_strategy="steps",  # 更新為新的參數名稱
        eval_steps=200,  # 減少評估頻率
        save_steps=200,
        save_total_limit=2,
        logging_steps=50,
        fp16=True,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused",  # 使用融合優化器
        lr_scheduler_type="cosine",  # 使用餘弦學習率調度
        warmup_ratio=0.1,  # 添加熱身階段
        learning_rate=2e-4  # 調整學習率
    )
    
    # 修改早停策略
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,  # 減少耐心值
        early_stopping_threshold=0.02  # 調整閾值
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