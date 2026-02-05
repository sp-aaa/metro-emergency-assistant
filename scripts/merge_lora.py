import torch
import sys
import os
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import model_config, LORA_WEIGHTS_DIR, MERGED_MODEL_DIR

def merge_model():
    print("Starting model merging...")
    
    # 1. Load Base Model (Full precision or fp16 for merging, NOT 4-bit)
    print(f"Loading base model: {model_config.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.base_model,
        trust_remote_code=True
    )

    # 2. Load LoRA Adapter
    print(f"Loading LoRA adapters from: {LORA_WEIGHTS_DIR}")
    try:
        model = PeftModel.from_pretrained(base_model, str(LORA_WEIGHTS_DIR))
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Make sure you have trained the model first!")
        return

    # 3. Merge weights
    print("Merging weights...")
    model = model.merge_and_unload()

    # 4. Save Merged Model
    output_dir = str(MERGED_MODEL_DIR)
    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Merge complete! You can now convert this model to GGUF format.")

if __name__ == "__main__":
    merge_model()
