import os
import torch
import sys
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import model_config, training_config, TRAIN_DATA_PATH, LORA_WEIGHTS_DIR

def train():
    print(f"Loading configuration...")
    print(f"Base Model: {model_config.base_model}")
    print(f"Output Dir: {training_config.output_dir}")
    
    # 1. Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.use_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, model_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=model_config.use_nested_quant,
    )

    # 2. Load Base Model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Silence warnings during training
    model.config.pretraining_tp = 1
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16 training

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_config.target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load Dataset
    print(f"Loading dataset from {TRAIN_DATA_PATH}")
    dataset = load_dataset("json", data_files=str(TRAIN_DATA_PATH), split="train")
    
    # Formatting function
    def format_instruction(sample):
        return f"""<|im_start|>user
{sample['instruction']}
{sample.get('input', '')}
<|im_end|>
<|im_start|>assistant
{sample['output']}
<|im_end|>
"""

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        max_grad_norm=0.3,
        warmup_ratio=training_config.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=training_config.optim, # Correct scheduler/optim mapping if needed, else "cosine"
        report_to="tensorboard",
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        save_total_limit=training_config.save_total_limit,
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",  # We will use formatting_func instead
        max_seq_length=training_config.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        formatting_func=format_instruction,
    )

    # 8. Start Training
    print("Starting training...")
    trainer.train()

    # 9. Save Model
    print(f"Saving model to {training_config.output_dir}")
    trainer.model.save_pretrained(training_config.output_dir)
    tokenizer.save_pretrained(training_config.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train()
