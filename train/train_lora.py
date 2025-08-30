from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from code_util.get_instructions import get_model_instructions

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" # TODO: Get model name from user
model_name_split = MODEL_NAME.split("/")
OUTPUT_DIR = f"../output/lora/{model_name_split[0]}/{model_name_split[1]}"
TRAIN_FILE = "../data/summary_data.jsonl" # TODO: Get training data file from user

def main():
    print("=== Train LoRA ===")

    print("1/7: Loading dataset...")
    training_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    print(f"Dataset loaded with {len(training_dataset)} samples")

    print("2/7: Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
        output_dir=OUTPUT_DIR,
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

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    trainer.train()

    print("7/7: Saving model and tokenizer...")
    peft_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete! Model + tokenizer saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()