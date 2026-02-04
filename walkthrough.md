# LoRA Training Walkthrough

I have set up the environment and scripts necessary to create LoRA adapters.

## Changes Created

I created a new directory `lora_training` in your scratch space containing:

1.  **[lora_guide.md](file:///C:/Users/mayur/.gemini/antigravity/scratch/lora_training/lora_guide.md)**: A step-by-step guide to understanding and running the training.
2.  **[train_lora.py](file:///C:/Users/mayur/.gemini/antigravity/scratch/lora_training/train_lora.py)**: The Python script that performs the actual training using `peft` and `transformers`.
3.  **[requirements.txt](file:///C:/Users/mayur/.gemini/antigravity/scratch/lora_training/requirements.txt)**: List of necessary libraries.

## How to Verify

1.  Open your terminal.
2.  Navigate to the directory:
    ```bash
    cd C:\Users\mayur\.gemini\antigravity\scratch\lora_training
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the training script (ensure you have a GPU):
    ```bash
    python train_lora.py
    ```

> [!NOTE]
> The script defaults to finding a `dataset.json` file. If none is found, it will attempt to download a demo dataset from Hugging Face.
