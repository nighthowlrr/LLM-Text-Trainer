from pathlib import Path

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from code_util.file_helpers import pick_folder

def main():
    print("=== Merge LoRA ===")

    parser = argparse.ArgumentParser(description='Merge LoRA adapter into base model and save merged model')
    parser.add_argument("--base_model", type=str, help="Path to local base model or HuggingFace Id of model")
    parser.add_argument("--lora_dir", type=str, help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, help="Where to save the merged model")
    args = parser.parse_args()

    base_model = (args.base_model or input("Enter local path to base model or HuggingFace ID of model (e.g. Qwen/Qwen2.5-1.5B-Instruct): ")
                  .strip())
    lora_adapter = args.lora_dir or pick_folder("Select folder with trained LoRA adapter...")
    given_output_dir = args.output_dir or pick_folder("Select output folder...")

    model_name_dir = Path(base_model).name if Path(base_model).exists() else base_model.split("/")[-1]
    output_dir = given_output_dir + f"/{model_name_dir}"

    if Path(base_model).exists():
        print("Loading base model from local path: ", base_model)
    else:
        print("Downloading base model from HuggingFace Hub: ", base_model)
    print("LoRA adapter:", lora_adapter)
    print("Output directory:", output_dir)
    print("==================")

    print("1/4: Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16,
        device_map="cpu"
    )

    print("2/4: Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_adapter)

    print("3/4: Merging LoRA into base model...")
    model = model.merge_and_unload()

    print("4/4: Saving merged model...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_dir)

    print(f"\nLoRA merged successfully! Saved to {output_dir}")
    print("==================")

if __name__ == "__main__":
    main()