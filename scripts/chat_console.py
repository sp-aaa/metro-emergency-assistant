import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from src.config import model_config, MERGED_MODEL_DIR

def chat_console():
    print(f"Loading merged model from: {MERGED_MODEL_DIR}")
    print("Note: This requires the model to be merged first (scripts/merge_lora.py)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_DIR, 
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_DIR,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you run scripts/merge_lora.py?")
        return

    print("\n--- UrbanTransit-Assistant Console Chat ---")
    print("Type 'quit' or 'exit' to end session.\n")

    messages = [
        {"role": "system", "content": "你是城市轨道交通应急处置助手，专门为地铁运营工作人员提供《地铁突发事件应急预案》相关的咨询服务。"}
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chat_console()
