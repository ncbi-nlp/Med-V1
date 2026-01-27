# SFT Training Examples (Accelerate + TRL)

This repository provides example commands for running supervised fine-tuning (SFT) using
Hugging Face **Accelerate** and **TRL's SFTTrainer**.

Two example setups are shown:

1. **Large model with LoRA (PEFT)**  
   Fine-tuning *Llama-3.3-70B-Instruct* using LoRA on 4 GPUs.

2. **Small model full fine-tuning**  
   Fine-tuning *Qwen2.5-3B-Instruct* on a single node with gradient checkpointing.

---

## Prerequisites

Install required packages:

```bash
pip install -U transformers datasets accelerate trl peft bitsandbytes
```

## Dataset Format (Chat Messages JSONL)

The training script expects the dataset to be in **JSON or JSONL** format.  
Each line should contain a single training example with a `messages` field that follows a chat-style structure.

### Required Structure

Each example must look like this:

```json
{
  "messages": [
    { "role": "system", "content": "<system message text>" },
    { "role": "user", "content": "<user message text>" },
    { "role": "assistant", "content": "<assistant message text>" }
  ]
}
```


### Field Definitions

- **messages**  
  A list of chat turns that make up one training example.

- **role**  
  Must be one of:
  - `"system"` – instructions or context that define assistant behavior  
  - `"user"` – the user’s input or question  
  - `"assistant"` – the desired model response (training target)

- **content**  
  A string containing the text for that role.

### Notes

- The order of messages matters.  
  The expected order is: `system → user → assistant`.

- The `assistant` message is treated as the **label** during supervised fine-tuning.

- You may omit the `system` message if your data does not use system prompts, for example:

```json
{
  "messages": [
    { "role": "user", "content": "Explain gradient accumulation." },
    { "role": "assistant", "content": "Gradient accumulation simulates larger batch sizes by..." }
  ]
}
```

- Each JSON object should be on its own line if using `.jsonl`:

```
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]}
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Explain LoRA."},{"role":"assistant","content":"LoRA is a parameter-efficient fine-tuning method..."}]}
```


## Training Arguments (What They Do)

Below are the key CLI arguments used in the provided training commands, grouped by purpose.

### Model & Dataset

- `--model_name_or_path`  
  Hugging Face model id (or local path) to start from.  
  Examples: `meta-llama/Llama-3.3-70B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`.

- `--dataset_name`  
  Path to your dataset file(s). Commonly a `.jsonl` file.  
  Examples: `./data_0314.jsonl`, `./datasets/sft_data_1497981.jsonl`.

- `--dataset_num_proc`  
  Number of CPU processes for dataset preprocessing/loading.  
  Increasing this can speed up data preparation (up to your CPU limits).

### Optimization / Learning

- `--learning_rate`  
  Base learning rate for training (e.g. `2e-5`).

- `--num_train_epochs`  
  How many full passes over the training dataset.

- `--per_device_train_batch_size`  
  Batch size **per GPU** (or per process). For large models this is usually small.

- `--gradient_accumulation_steps`  
  Accumulate gradients for N steps before applying an optimizer update.  
  This increases the **effective batch size** without increasing GPU memory.

  **Effective batch size** (common approximation):
  ```
  effective_batch = per_device_train_batch_size
                   × num_gpus
                   × gradient_accumulation_steps
  ```

### Memory / Speed

- `--gradient_checkpointing`  
  Trades extra compute for lower memory by checkpointing activations.
  Helpful when you are close to OOM (out-of-memory).

- `--max_seq_length`  
  Maximum token length for each training example after tokenization/truncation.
  Example: `--max_seq_length 1024`.

### Logging & Checkpointing

- `--logging_steps`  
  How often to log metrics (in training steps).

- `--save_strategy`  
  When to save checkpoints. Common values:
  - `epoch` (save at the end of each epoch)
  - `steps` (save every N steps, if configured)

- `--output_dir`  
  Directory where checkpoints and logs are written.

### PEFT / LoRA (Parameter-Efficient Fine-Tuning)

- `--use_peft`  
  Enables PEFT (e.g., LoRA adapters) instead of full fine-tuning.

- `--lora_r`  
  LoRA rank. Higher rank = more adapter parameters and capacity.

- `--lora_alpha`  
  LoRA scaling factor. Commonly paired with `lora_r`.

---

## Example Commands

### Example 1 — Llama-3.3-70B with LoRA (Multi-GPU + nohup)

Uses 4 GPUs and runs in the background, writing logs to `training.out` and `error.log`.

```bash
OMP_NUM_THREADS=8 nohup \
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file ../default_config.yaml sft.py \
  --model_name_or_path meta-llama/Llama-3.3-70B-Instruct \
  --dataset_name ./data_0314.jsonl \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --logging_steps 50 \
  --output_dir ./models/llama33_70BI_0314_v2 \
  --use_peft \
  --lora_r 32 \
  --lora_alpha 16 \
  --dataset_num_proc 20 \
  --save_strategy epoch \
  > training.out 2> error.log &
```

**Effective batch size estimate** (4 GPUs):
```
2 × 4 × 8 = 64
```

---

### Example 2 — Qwen2.5-3B Full Fine-Tuning (with Gradient Checkpointing)

```bash
OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=true \
accelerate launch --config_file ../default_config.yaml sft.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset_name ./datasets/sft_data_1497981.jsonl \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --logging_steps 50 \
  --output_dir ./models/qwen25_3B_I2507_1497981_v1 \
  --save_strategy epoch \
  --max_seq_length 1024 \
  --gradient_checkpointing \
  --dataset_num_proc 20
```

---

### Example 3 — Minimal Debug Run (Quick Sanity Check)

Use a tiny dataset and 1 epoch to confirm everything works before launching big runs:

```bash
accelerate launch --config_file ../default_config.yaml sft.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --dataset_name ./datasets/debug.jsonl \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 10 \
  --output_dir ./models/debug_run \
  --save_strategy epoch \
  --max_seq_length 512
```

