import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

# --- Configuration ---
# You can replace this with "mistralai/Mistral-7B-v0.1" or any other supported model
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit" # Using a pre-quantized 4bit model for efficiency, or use base model with config below
# Note: For standard HF models, use "meta-llama/Meta-Llama-3-8B" and the bnb_config below will handle quantization.
# For this example, let's use a standard path but assume access or use a smaller public model for testing like:
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

# Directory to save the adapter
OUTPUT_DIR = "./lora-adapter"

# --- 1. Load Model & Tokenizer ---
print(f"Loading model: {MODEL_NAME}")

# BitsAndBytes Config for 4-bit quantization (QLoRA) - saves memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Fix for models that don't have a pad token
tokenizer.padding_side = "right" # Fix for fp16

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# --- 2. LoRA Configuration ---
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64, # Rank: higher = more parameters to train, but better results (usually 8, 16, 32, 64)
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] # Modules to apply LoRA to
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 3. Dataset Preparation ---
# Load your dataset. It should be a jsonl file or similar. 
# For this example, we'll try to load a local file 'dataset.json' if it exists, otherwise use a demo dataset.
dataset_file = "dataset.json"

if os.path.exists(dataset_file):
    print(f"Loading local dataset from {dataset_file}")
    dataset = load_dataset("json", data_files=dataset_file, split="train")
else:
    print("Local 'dataset.json' not found. Loading demo dataset (guanaco-1k) ...")
    dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# Verify dataset structure
print("Sample data:", dataset[0])

# --- 4. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2, # Reduce if OOM
    gradient_accumulation_steps=4, # Increase if batch size is small
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False, # Set to True if using A100/H100
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none" # disable wandb/tensorboard for simple run
)

# --- 5. Trainer ---
# SFTTrainer handles the formatting and data processing convenience
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # The name of the column containing the text to train on
    max_seq_length=512, # Adjust based on your VRAM
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# --- 6. Train ---
print("Starting training...")
trainer.train()

# --- 7. Save Adapter ---
print(f"Saving adapter to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! You can now load this adapter on top of the base model.")
