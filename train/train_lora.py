from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import argparse

from code_util.file_helpers import pick_folder, pick_file
from code_util.get_instructions import get_model_instructions

def main():
    print("=== Train LoRA ===")

    parser = argparse.ArgumentParser(description="Train LoRA")
    parser.add_argument("--model-name", type=str, required=False, help="HuggingFace ID of model")
    parser.add_argument("--train-file", type=str, required=False, help="Location of training data file")
    parser.add_argument("--output-dir", type=str, required=False, help="Where to save the LoRA adapter")
    args = parser.parse_args()

    model_name = args.model_name or input("Enter HuggingFace ID of model (e.g. Qwen/Qwen2.5-1.5B-Instruct): ")
    train_file = args.train_file or pick_file("Select training data file...", True,
                                              [("JSONL Files", "*.jsonl")])
    given_output_dir = args.output_dir or pick_folder("Select output directory...")
    output_dir = given_output_dir + f"/{model_name.split("/")[0]}/{model_name.split("/")[1]}"

    print("Model name:", model_name)
    print("Training file:", train_file)
    print("Output directory:", output_dir)

    print("1/7: Loading dataset...")
    training_dataset = load_dataset("json", data_files=train_file, split="train")
    print(f"Dataset loaded with {len(training_dataset)} samples")

    print("2/7: Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("3/7: Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(model, lora_config)

    def preprocess(batch):
        prompt = get_model_instructions() + f":\n{batch['input']}\n\nSummary:"
        tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=1024)
        labels = tokenizer(batch["output"], truncation=True, padding="max_length", max_length=1024)
        tokenized["labels"] = labels["input_ids"]
        return tokenized

    print("4/7: Tokenizing dataset...")
    tokenized_dataset = training_dataset.map(preprocess, remove_columns=list(training_dataset.column_names))

    print("5/7: Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=False,
        bf16=False # only fp32
    )

    print("6/7: Starting training...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Detect if MPS backend is available for pin_memory setting
    if torch.backends.mps.is_available():
        pin_memory = False
        print("Detected MPS backend - disabling pin_memory to avoid warnings.") # TODO: Remove this log in end version
    else:
        pin_memory = True

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    trainer.args.dataloader_pin_memory = pin_memory;

    trainer.train()

    print("7/7: Saving model and tokenizer...")
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nTraining complete! Model + tokenizer saved to: {output_dir}")

if __name__ == "__main__":
    main()