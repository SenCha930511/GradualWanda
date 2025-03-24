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
    Trainer
)

def lora_finetune(config: LoRaConfig):
    """
    使用自定義 LoRaConfig (dataclass) 的 LoRA 微調函式。
    """

    print(f"Loading model and tokenizer from {config.model}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        device_map="auto",
        quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model)
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
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=config.max_length)
    
    remove_columns = [col for col in dataset.column_names if col != "text"]
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=config.output_dir,      # 預設輸出資料夾
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,         # 由 config 讀取
        per_device_train_batch_size=config.per_device_train_batch_size,  # 由 config 讀取
        gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        fp16=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting LoRA fine-tuning...")
    trainer.train()

    output_dir = config.save_model if config.save_model else "lora_finetuned_model"
    print(f"Saving LoRA fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    print("LoRA fine-tuning completed.\n")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
