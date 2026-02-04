# How to Create LoRA Adapters

This guide explains how to fine-tune a Large Language Model (LLM) using LoRA (Low-Rank Adaptation) adapters. This method allows you to train a small number of parameters (adapters) on top of a frozen base model, making it much faster and more memory-efficient than full fine-tuning.

## Prerequisites

1.  **Hardware**: NVIDIA GPU with at least 8GB VRAM (for 7B models in 4-bit).
2.  **Software**: Python 3.10+, CUDA installed.

## Setup

1.  Open your terminal/command prompt.
2.  Navigate to this folder: `cd C:\Users\mayur\.gemini\antigravity\scratch\lora_training`
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Preparing Data

You need a dataset. The simplest format is a JSON or JSONL file where each line is a training example. The script expects a field named `text` which contains the full prompt and completion.

Example `dataset.json`:
```json
[
  { "text": "### Human: What is LoRA?\n### Assistant: LoRA stands for Low-Rank Adaptation. It is a PEFT technique..." },
  { "text": "### Human: How do I train?\n### Assistant: You can use the SFTTrainer from HuggingFace..." }
]
```

Create this file in the same directory as the script.

## Running Training

Run the python script:

```bash
python train_lora.py
```

### Configuration in `train_lora.py`

You can modify the script to change:
- `MODEL_NAME`: The base model ID from Hugging Face (e.g., `meta-llama/Meta-Llama-3-8B`).
- `OUTPUT_DIR`: Where to save the fine-tuned adapter.
- `num_train_epochs`: How many times to iterate over the dataset.
- `dataset_text_field`: The key in your JSON that holds the text.

## Using the Adapter

Once trained, `lora-adapter` folder will be created. You can load it like this in another script:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("base_model_name")
model = PeftModel.from_pretrained(base_model, "path/to/lora-adapter")
```
