import gc
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

def lora_finetune(model_path, epochs=2, per_device_train_batch_size=1, max_length=256):
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

    # 1. If no pad token is set, use EOS as the pad token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 2. Resize the model embeddings in case the tokenizer has been updated.
    model.resize_token_embeddings(len(tokenizer))

    gc.collect()
    torch.cuda.empty_cache()
    
    lora_config = LoraConfig(
        r=2,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    print("LoRA applied to model.")

    # Example dataset
    dataset = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)
    
    remove_columns = [col for col in dataset.column_names if col != "text"]
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=remove_columns)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
         output_dir="lora_finetuned_model",
         overwrite_output_dir=True,
         num_train_epochs=epochs,
         per_device_train_batch_size=per_device_train_batch_size,
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
    
    print("Saving LoRA fine-tuned model...")
    model.save_pretrained("lora_finetuned_model")
    print("LoRA fine-tuning completed.\n")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
