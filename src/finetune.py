import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import json
import os
from pathlib import Path

print("Loading model and tokenizer....")

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

# Save base model for future use
print("\nSaving base model...")
base_model_save_path = Path.home() / "models" / "llama-3.1-8b-instruct"
base_model_save_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(base_model_save_path))
tokenizer.save_pretrained(str(base_model_save_path))

print(f"Base model saved to: {base_model_save_path}")
print(f"To load for other project use: AutoModelForCausalLM.from_pretrained({'base_model_save_path'})")
print()

print("Model loaded successfully!")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

# LoRA configuration
print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load and format dataset
print(f"\nLoading training data...")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl('data/processed/train.jsonl')
val_data = load_jsonl('data/processed/val.jsonl')

print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")

# Format data for training
def format_instructions(sample):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    You are a medical triage assistant.  Based on the patient information, provide the medical specialty and clinical assessment.

    {sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    {sample['output']}<|eot_id|>
    """

train_formatted = [{"text": format_instructions(x)} for x in train_data]
val_formatted = [{"text": format_instructions(x) for x in val_data}]

train_dataset = Dataset.from_list(train_formatted)
val_dataset = Dataset.from_list(val_formatted)

print("\n" + "="*80)
print("DEBUG: Full formatted example:")
print("="*80)
print(train_formatted[0]["text"])
print("="*80)
print(f"\nLength: {len(train_formatted[0]['text'])} characters")

# Training arguments
print("\nSetting up training...")
training_args = SFTConfig(
    output_dir="../models/llama-medical-triage-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_steps=50,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    warmup_steps=10,
    max_grad_norm=0.3,
    max_length=2048,
    dataset_text_field="text",
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
)

print(f"Beginning training cycle...")
print("="*80)

# Train model
trainer.train()

print("Training complete...")
print("Saving model...")

model.save_pretrained("models/llama-medical-triage-lora")
tokenizer.save_pretrained("models/llama-medical-triage-lora")

print("Model saved to models/llama-medical-triage-lora")