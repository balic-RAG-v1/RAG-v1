import os
import json
import hashlib
import time
import pandas as pd
from app.ingestion.loader import load_file
from app.llm.ollama_client import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Constants
LORA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(LORA_DIR, "dataset.json")
PROCESSED_FILES_LOG = os.path.join(LORA_DIR, "processed_files.json")
OUTPUT_DIR = os.path.join(LORA_DIR, "lora-adapter")

# Ensure directories exist
os.makedirs(LORA_DIR, exist_ok=True)

def get_file_hash(file_path):
    """Calculates MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def is_file_processed(file_hash):
    """Checks if the file hash exists in the processed files log."""
    if not os.path.exists(PROCESSED_FILES_LOG):
        return False
    try:
        with open(PROCESSED_FILES_LOG, "r") as f:
            processed = json.load(f)
            return file_hash in processed
    except (json.JSONDecodeError, FileNotFoundError):
        return False

def mark_file_processed(file_hash, file_name):
    """Marks a file as processed."""
    processed = {}
    if os.path.exists(PROCESSED_FILES_LOG):
        try:
            with open(PROCESSED_FILES_LOG, "r") as f:
                processed = json.load(f)
        except json.JSONDecodeError:
            pass
    
    processed[file_hash] = {"file_name": file_name, "timestamp": time.time()}
    
    with open(PROCESSED_FILES_LOG, "w") as f:
        json.dump(processed, f, indent=4)

def generate_training_data(text_content):
    """
    Generates Q&A pairs and extracts entities from text using LLM.
    Returns a list of dictionaries.
    """
    llm = get_llm()
    
    prompt_template = """
    You are an expert at creating training data for fine-tuning LLMs.
    Given the following text, extract key information and generate 3-5 high-quality Question-Answer pairs.
    Also extract key entities (names, dates, specific values) mentioned.
    
    Return the output as a JSON object with a list of "qa_pairs" (each having "instruction", "input" (optional), "output") and "entities" (list of strings).
    
    Text:
    {text}
    
    JSON Output:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm | JsonOutputParser()
    
    try:
        # Split text if too long (simplified for now, taking first 4000 chars)
        # In production, you'd iterate over chunks.
        truncated_text = text_content[:4000] 
        response = chain.invoke({"text": truncated_text})
        if response is None:
             print("Warning: LLM returned None for training data generation.")
             return {"qa_pairs": [], "entities": []}
        return response
    except Exception as e:
        print(f"Error generating data: {e}")
        return {"qa_pairs": [], "entities": []}

def append_to_dataset(new_data):
    """Appends new Q&A pairs to the dataset.json file."""
    dataset = []
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r") as f:
                dataset = json.load(f)
        except json.JSONDecodeError:
            pass
            
    # Transform to Alpaca/Instruction format
    formatted_data = []
    for item in new_data.get("qa_pairs", []):
        entry = {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        formatted_data.append(entry)
        
    dataset.extend(formatted_data)
    
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=4)
        
    return formatted_data

def train_model():
    """
    Executes the LoRA training process.
    This mimics the logic from the standalone train_lora.py script.
    """
    print("Starting LoRA training...")
    
    # Import here to avoid loading heavy libraries on module import if not needed
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
    
    # Configuration
    MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
    
    # BitsAndBytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True # Enables CPU offloading for quantized weights when VRAM is full
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": "cpu"}, # Force map offload layers explicitly
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
        )
        
        model = get_peft_model(model, peft_config)
        
        if not os.path.exists(DATASET_FILE):
            print("No dataset found. Skipping training.")
            return "No dataset found.", pd.DataFrame(columns=["Step", "Loss", "Metric Type"])
            
        dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
        
        # Split dataset if we have enough data to ensure at least 1 eval example
        if len(dataset) > 4:
            split = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            do_eval = True
        else:
            train_dataset = dataset
            eval_dataset = None
            do_eval = False
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=1, # Granular logging for UI updates
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=50, # Limited steps for demo/speed
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",
            eval_strategy="steps" if do_eval else "no",
            eval_steps=5 if do_eval else None
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            dataset_text_field="output", # Using 'output' or formatting to text would be better, but standard is text column
            # For simplicity, we assume dataset has 'text' field or we format it. 
            # Let's add a formatting function.
            formatting_func=lambda x: [f"### Instruction: {i}\n### Input: {inp}\n### Output: {o}" for i, inp, o in zip(x['instruction'], x['input'], x['output'])],
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_args,
            packing=False,
        )
        
        trainer.train()
        
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Extract and format metrics for visualization
        history = trainer.state.log_history
        metrics_data = []
        for log in history:
            step = log.get("step")
            if "loss" in log:
                metrics_data.append({"Step": step, "Loss": log["loss"], "Metric Type": "Train Loss"})
            if "eval_loss" in log:
                metrics_data.append({"Step": step, "Loss": log["eval_loss"], "Metric Type": "Validation Loss"})
                
        metrics_df = pd.DataFrame(metrics_data) if metrics_data else pd.DataFrame(columns=["Step", "Loss", "Metric Type"])
        
        return "Training completed successfully.", metrics_df
        
    except Exception as e:
        print(f"Training failed: {e}")
        return f"Training failed: {str(e)}", pd.DataFrame(columns=["Step", "Loss", "Metric Type"])

def process_file_for_lora(file_path):
    """
    Main entry point.
    1. Check deduplication.
    2. Generate data.
    3. Update dataset.
    4. Train model.
    """
    file_hash = get_file_hash(file_path)
    file_name = os.path.basename(file_path)
    
    if is_file_processed(file_hash):
        return "File already processed. Skipping training.", pd.DataFrame(), pd.DataFrame(columns=["Step", "Loss", "Metric Type"])
    
    # Load content
    docs = load_file(file_path)
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # Generate Data
    generated_data = generate_training_data(full_text)
    
    # Append to dataset
    new_entries = append_to_dataset(generated_data)
    
    # Mark as processed
    mark_file_processed(file_hash, file_name)
    
    # Create DataFrame for UI
    df_preview = pd.DataFrame(new_entries)
    if "entities" in generated_data:
         # Add entities as a column or separate ref?
         # For simplicity, just adding to the first row or returning strictly Q&A for now.
         # Let's add an 'Entities' column to the first row just for visibility
         df_preview["Extracted Entities"] = ""
         if len(df_preview) > 0:
             df_preview.at[0, "Extracted Entities"] = ", ".join(generated_data["entities"])
    
    # Trigger Training
    training_status, df_metrics = train_model()
    
    return training_status, df_preview, df_metrics

import shutil

def clear_lora_data():
    """Clears all LoRA datasets, logs, and adapter files."""
    try:
        if os.path.exists(DATASET_FILE):
            os.remove(DATASET_FILE)
        if os.path.exists(PROCESSED_FILES_LOG):
            os.remove(PROCESSED_FILES_LOG)
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        return True, "LoRA data cleared successfully."
    except Exception as e:
        return False, f"Error clearing LoRA data: {e}"
